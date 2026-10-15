"""
Simulador mínimo de la línea de empaque piloto (entorno).

Genera las percepciones descritas en la sección 5 del diseño:
  * imagen RGB de cada fruta (sintética, procedural),
  * señal de disparo (presencia de unidad),
  * lectura (ruidosa) de la velocidad de la banda,
  * identificador de lote / proveedor.

Respeta las propiedades de ambiente de la sección 3:
  * parcialmente observable: la etiqueta real NO viaja en el percepto (queda en ``verdad``),
  * estocástico: iluminación, ruido, posición, rotación, desenfoque y variedad son aleatorios,
  * dinámico: la banda avanza en tiempo simulado, con deriva de velocidad y plazo de decisión,
  * agente único y episódico: cada unidad se clasifica de forma independiente.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .config import CLASES, IMG_SIZE

# --------------------------------------------------------------------------- #
# Variedades de fruta (apariencia base en HSV de OpenCV: H 0-179, S/V 0-255)
# --------------------------------------------------------------------------- #
VARIEDADES: Dict[str, Dict[str, Any]] = {
    "manzana": dict(h=3.0, s=190.0, v=185.0, ejes=(0.36, 0.34)),
    "naranja": dict(h=14.0, s=225.0, v=230.0, ejes=(0.35, 0.35)),
    "banano": dict(h=26.0, s=205.0, v=225.0, ejes=(0.42, 0.15)),
    "pera": dict(h=36.0, s=150.0, v=195.0, ejes=(0.38, 0.28)),  # no se usa en el entrenamiento
}
VARIEDADES_ENTRENAMIENTO = ("manzana", "naranja", "banano")

BANDA_HSV = (108.0, 130.0, 110.0)     # banda transportadora azul (grado alimentario)


# --------------------------------------------------------------------------- #
# Condiciones de captura (todas ajustables por escenario)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Condiciones:
    ganancia_media: float = 1.0        # iluminación media de la escena
    ganancia_deriva: float = 0.03      # desviación de la deriva lenta de iluminación
    parpadeo: float = 0.04             # fluctuación aleatoria por cuadro
    temp_color: float = 0.05           # desbalance aleatorio entre canales
    ruido_sigma: float = 4.0           # ruido gaussiano del sensor (niveles de gris)
    desenfoque_px: int = 0             # longitud del desenfoque de movimiento (a lo largo de la banda)
    jitter_px: float = 14.0            # desplazamiento aleatorio del centro
    escala: Tuple[float, float] = (0.90, 1.10)
    rotar: bool = True
    frontera: bool = False             # casos límite: clases próximas entre sí
    brillo_especular: float = 0.9


@dataclass
class Percepto:
    """Lo único que el agente recibe del entorno en cada paso."""
    t_s: float                         # tiempo simulado de llegada a la cámara
    unit_id: int
    lot_id: str
    proveedor: str
    imagen: Any                        # ndarray BGR uint8; en casos de falla puede ser cualquier cosa
    velocidad_banda: Optional[float]   # m/s (encoder con ruido) o None
    disparo: bool = True


@dataclass
class Verdad:
    """Información oculta al agente (solo para evaluar)."""
    clase: Optional[str]
    variedad: Optional[str]
    falla: Optional[str] = None
    velocidad_real: float = 0.5


# --------------------------------------------------------------------------- #
# Utilidades de dibujo
# --------------------------------------------------------------------------- #
def _ruido_suave(rng: np.random.Generator, shape: Tuple[int, int], sigma: float) -> np.ndarray:
    n = rng.standard_normal(shape).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigma)
    return n / (float(n.std()) + 1e-6)


def _parametros_estado(rng: np.random.Generator, clase: str, frontera: bool) -> Dict[str, float]:
    """Variables latentes de cada clase. Hay solapamiento deliberado entre clases."""
    if clase == "fresca":
        r = rng.uniform(0.10, 0.34) if frontera else rng.uniform(0.0, 0.30)
        d, k, tiny = 0.0, 0, int(rng.choice([0, 0, 0, 1, 2]))
    elif clase == "maduracion":
        r = rng.uniform(0.30, 0.50) if frontera else rng.uniform(0.34, 0.95)
        d, k, tiny = 0.0, 0, int(rng.choice([0, 0, 1]))
    else:  # danada
        r = rng.uniform(0.0, 0.9)
        d = rng.uniform(0.025, 0.07) if frontera else rng.uniform(0.07, 0.40)
        k, tiny = int(rng.integers(1, 6)), 0
    return dict(r=float(r), d=float(d), k=k, tiny=tiny)


def _dibujar_fruta(hsv: np.ndarray, rng: np.random.Generator, clase: str, variedad: str,
                   cond: Condiciones, desplazamiento: Tuple[float, float] = (0.0, 0.0),
                   escala_extra: float = 1.0) -> None:
    H, W = hsv.shape[:2]
    v = VARIEDADES[variedad]
    est = _parametros_estado(rng, clase, cond.frontera)

    cx = W / 2 + rng.uniform(-cond.jitter_px, cond.jitter_px) + desplazamiento[0]
    cy = H / 2 + rng.uniform(-cond.jitter_px, cond.jitter_px) + desplazamiento[1]
    ang = rng.uniform(0, 180) if cond.rotar else 0.0
    sc = rng.uniform(*cond.escala) * escala_extra
    ax, ay = int(v["ejes"][0] * W * sc), int(v["ejes"][1] * H * sc)

    mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(mask, (int(cx), int(cy)), (max(ax, 4), max(ay, 4)), ang, 0, 360, 255, -1)
    m = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 0.9)
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    shade = dt / (dt.max() + 1e-6)

    r, d = est["r"], est["d"]
    h0 = v["h"] + (14.0 - v["h"]) * 0.5 * r
    s0 = v["s"] * (1 - 0.30 * r)
    v0 = v["v"] * (1 - 0.20 * r)

    tex = _ruido_suave(rng, (H, W), 2.0)
    mott = _ruido_suave(rng, (H, W), 6.0)
    fh = h0 + 1.0 * tex
    fs = s0 + 8.0 * tex
    fv = v0 * (0.72 + 0.38 * shade) + 6.0 * tex
    fv = fv * (1 - 0.22 * r * np.clip(mott, 0, 2) / 2.0)          # moteado por maduración

    # brillo especular
    hx, hy = cx - 0.15 * ax, cy - 0.20 * ay
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    spec = np.exp(-(((xx - hx) ** 2 + (yy - hy) ** 2) / (2 * (0.07 * W) ** 2)))
    fv = fv + 70 * cond.brillo_especular * spec
    fs = fs - 90 * cond.brillo_especular * spec

    capa = np.stack([fh, fs, fv], axis=-1)

    def pintar(alpha: np.ndarray, color: Tuple[float, float, float], ruido: float = 8.0) -> None:
        a = (alpha * m)[..., None]
        c = np.array(color, np.float32)[None, None, :] + ruido * rng.standard_normal((H, W, 3)).astype(np.float32) * 0.4
        capa[:] = capa * (1 - a) + c * a

    # pecas de maduración y manchas leves
    n_pecas = int(30 * r * rng.uniform(0.5, 1.5)) if r > 0.33 else 0
    n_pecas += est["tiny"]
    if n_pecas:
        al = np.zeros((H, W), np.float32)
        for _ in range(n_pecas):
            ang_p, rad_p = rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.8)
            px = int(cx + np.cos(ang_p) * rad_p * ax * 0.8)
            py = int(cy + np.sin(ang_p) * rad_p * ay * 0.8)
            cv2.circle(al, (px, py), int(rng.integers(1, 3)), 1.0, -1)
        al = cv2.GaussianBlur(al, (0, 0), 0.8)
        pintar(np.clip(al * 1.5, 0, 1), (12.0, 120.0, 70.0))

    # daño: manchas oscuras irregulares + posible moho
    if clase == "danada" and d > 0:
        area_fruta = float(mask.sum() / 255.0)
        k = est["k"]
        bl = np.zeros((H, W), np.float32)
        ruido_irr = _ruido_suave(rng, (H, W), 4.0)
        for _ in range(k):
            rad = np.sqrt(d * area_fruta / k / np.pi) * rng.uniform(0.7, 1.3)
            ang_b, rb = rng.uniform(0, 2 * np.pi), rng.uniform(0, 0.7)
            bx = cx + np.cos(ang_b) * rb * ax * 0.7
            by = cy + np.sin(ang_b) * rb * ay * 0.7
            bump = np.exp(-(((xx - bx) ** 2 + (yy - by) ** 2) / (2 * (rad * 0.75) ** 2)))
            bl = np.maximum(bl, bump)
        alpha = np.clip((bl + 0.25 * ruido_irr * bl - 0.35) * 4.0, 0, 1)
        pintar(alpha, (10.0 + rng.uniform(-4, 4), 110.0 + rng.uniform(-30, 30), 32.0 + rng.uniform(-12, 12)))
        if rng.random() < 0.35:
            bm = np.exp(-(((xx - (cx + rng.uniform(-0.4, 0.4) * ax)) ** 2 +
                           (yy - (cy + rng.uniform(-0.4, 0.4) * ay)) ** 2) / (2 * (0.09 * W) ** 2)))
            pintar(np.clip((bm - 0.4) * 3.0, 0, 1) * 0.9, (45.0, 40.0, 215.0), ruido=14.0)

    sel = (m > 0.02)[..., None]
    hsv[:] = np.where(sel, hsv * (1 - m[..., None]) + capa * m[..., None], hsv)


def renderizar_unidad(rng: np.random.Generator, clase: Optional[str], variedad: Optional[str],
                      cond: Condiciones, ganancia: float, n_frutas: int = 1) -> np.ndarray:
    """Imagen BGR uint8 de una unidad (o de la banda vacía si clase es None)."""
    H = W = IMG_SIZE
    hsv = np.zeros((H, W, 3), np.float32)
    hsv[..., 0] = BANDA_HSV[0]
    hsv[..., 1] = BANDA_HSV[1]
    yy = np.arange(H, dtype=np.float32)[:, None]
    hsv[..., 2] = BANDA_HSV[2] + 10 * np.sin(yy / 3.0 + rng.uniform(0, 6.28)) + 6 * _ruido_suave(rng, (H, W), 1.5)

    if clase is not None:
        if n_frutas == 1:
            _dibujar_fruta(hsv, rng, clase, variedad, cond)
        else:
            _dibujar_fruta(hsv, rng, clase, variedad, cond, desplazamiento=(-40, -8), escala_extra=0.85)
            _dibujar_fruta(hsv, rng, clase, variedad, cond, desplazamiento=(40, 10), escala_extra=0.85)

    hsv8 = np.clip(hsv, 0, 255)
    hsv8[..., 0] = np.mod(hsv[..., 0], 180)
    bgr = cv2.cvtColor(hsv8.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    # ---- etapa del sensor ----
    canal = 1.0 + cond.temp_color * rng.standard_normal(3).astype(np.float32)
    bgr = bgr * ganancia * canal[None, None, :]
    yy2, xx2 = np.mgrid[0:H, 0:W].astype(np.float32)
    vign = 1.0 - 0.18 * (((xx2 - W / 2) ** 2 + (yy2 - H / 2) ** 2) / ((W / 2) ** 2 * 2))
    bgr = bgr * vign[..., None]
    if cond.desenfoque_px > 1:
        k = np.ones((1, int(cond.desenfoque_px)), np.float32) / float(cond.desenfoque_px)
        bgr = cv2.filter2D(bgr, -1, k)
    bgr = bgr + cond.ruido_sigma * rng.standard_normal(bgr.shape).astype(np.float32)
    return np.clip(bgr, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Fallas de percepción
# --------------------------------------------------------------------------- #
TIPOS_FALLA = ("sin_dato", "vacia", "formato", "nan", "tipo", "negra", "saturada", "banda_vacia", "solapadas")


def _aplicar_falla(rng: np.random.Generator, tipo: str, cond: Condiciones, ganancia: float,
                   clase: str, variedad: str) -> Any:
    if tipo == "sin_dato":
        return None
    if tipo == "vacia":
        return np.zeros((0, 0, 3), np.uint8)
    if tipo == "formato":
        img = renderizar_unidad(rng, clase, variedad, cond, ganancia)
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if tipo == "nan":
        img = renderizar_unidad(rng, clase, variedad, cond, ganancia).astype(np.float32)
        img[40:80, 40:80, :] = np.nan
        return img
    if tipo == "tipo":
        return "imagen_0001.jpg"
    if tipo == "negra":
        return np.clip(rng.normal(2.0, 1.5, (IMG_SIZE, IMG_SIZE, 3)), 0, 255).astype(np.uint8)
    if tipo == "saturada":
        return np.full((IMG_SIZE, IMG_SIZE, 3), 255, np.uint8)
    if tipo == "banda_vacia":
        return renderizar_unidad(rng, None, None, cond, ganancia)
    if tipo == "solapadas":
        return renderizar_unidad(rng, clase, variedad, cond, ganancia, n_frutas=2)
    raise ValueError(f"Tipo de falla desconocido: {tipo}")


# --------------------------------------------------------------------------- #
# Escenario y línea de empaque
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Lote:
    lot_id: str
    proveedor: str
    priors: Tuple[float, float, float]     # P(fresca), P(maduracion), P(danada)


LOTES_POR_DEFECTO = (
    Lote("L-001", "Proveedor A", (0.70, 0.22, 0.08)),
    Lote("L-002", "Proveedor B", (0.55, 0.25, 0.20)),
    Lote("L-003", "Proveedor C", (0.40, 0.30, 0.30)),
)


@dataclass(frozen=True)
class Escenario:
    nombre: str
    descripcion: str = ""
    cond: Condiciones = Condiciones()
    lotes: Tuple[Lote, ...] = LOTES_POR_DEFECTO
    variedades: Tuple[str, ...] = VARIEDADES_ENTRENAMIENTO
    velocidad_ms: float = 0.50            # velocidad nominal de la banda (m/s)
    deriva_velocidad: float = 0.02        # desviación del paseo aleatorio de la velocidad
    ruido_encoder: float = 0.02
    separacion_m: Tuple[float, float] = (0.15, 0.40)
    plan_fallas: Tuple[Tuple[int, str], ...] = ()    # (índice de unidad, tipo de falla)
    prob_falla: float = 0.0


class LineaEmpaqueSimulada:
    """Entorno: produce percepciones y conserva la verdad oculta de cada unidad."""

    def __init__(self, escenario: Escenario, seed: int):
        self.esc = escenario
        self.rng = np.random.default_rng(seed)
        self.verdad: Dict[int, Verdad] = {}
        self._ganancia = escenario.cond.ganancia_media
        self._vel = escenario.velocidad_ms

    def flujo(self, n_unidades: int) -> Iterator[Percepto]:
        esc, rng = self.esc, self.rng
        t = 0.0
        plan = dict(esc.plan_fallas)
        bloque = max(1, n_unidades // max(1, len(esc.lotes)))
        for i in range(n_unidades):
            lote = esc.lotes[min(i // bloque, len(esc.lotes) - 1)]
            clase = str(rng.choice(CLASES, p=lote.priors))
            variedad = str(rng.choice(esc.variedades))

            # dinámica de la banda: deriva de velocidad y de iluminación (paseo aleatorio acotado)
            self._vel = float(np.clip(self._vel + esc.deriva_velocidad * rng.standard_normal()
                                      + 0.1 * (esc.velocidad_ms - self._vel), 0.05, 3.0))
            self._ganancia = float(np.clip(self._ganancia + esc.cond.ganancia_deriva * rng.standard_normal()
                                           + 0.15 * (esc.cond.ganancia_media - self._ganancia), 0.2, 2.0))
            g = float(np.clip(self._ganancia * (1 + esc.cond.parpadeo * rng.standard_normal()), 0.15, 2.2))
            sep = float(rng.uniform(*esc.separacion_m))
            t += sep / self._vel

            falla = plan.get(i)
            if falla is None and esc.prob_falla > 0 and rng.random() < esc.prob_falla:
                falla = str(rng.choice(TIPOS_FALLA))
            if falla is None:
                img = renderizar_unidad(rng, clase, variedad, esc.cond, g)
                verdad = Verdad(clase, variedad, None, self._vel)
            else:
                img = _aplicar_falla(rng, falla, esc.cond, g, clase, variedad)
                verdad_clase = None if falla == "banda_vacia" else clase
                verdad = Verdad(verdad_clase, variedad, falla, self._vel)

            self.verdad[i] = verdad
            lectura = self._vel * (1 + esc.ruido_encoder * rng.standard_normal())
            yield Percepto(t_s=t, unit_id=i, lot_id=lote.lot_id, proveedor=lote.proveedor,
                           imagen=img, velocidad_banda=float(lectura), disparo=True)


def generar_dataset(n_por_clase: int, seed: int, cond: Optional[Condiciones] = None,
                    variedades: Tuple[str, ...] = VARIEDADES_ENTRENAMIENTO,
                    aleatorizar_condiciones: bool = True) -> Tuple[List[np.ndarray], List[str]]:
    """Imágenes etiquetadas para entrenar o validar el modelo (distribución de entrenamiento)."""
    rng = np.random.default_rng(seed)
    base = cond or Condiciones()
    imgs: List[np.ndarray] = []
    etiquetas: List[str] = []
    for clase in CLASES:
        for _ in range(n_por_clase):
            c = base
            g = base.ganancia_media
            if aleatorizar_condiciones:
                g = float(rng.uniform(0.70, 1.35))
                c = replace(base, ruido_sigma=float(rng.uniform(2, 8)),
                            desenfoque_px=int(rng.choice([0, 0, 0, 3, 5])),
                            frontera=bool(rng.random() < 0.15))
            var = str(rng.choice(variedades))
            imgs.append(renderizar_unidad(rng, clase, var, c, g))
            etiquetas.append(clase)
    return imgs, etiquetas
