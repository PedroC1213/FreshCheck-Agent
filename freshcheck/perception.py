"""
Percepción: valida la imagen recibida, estima la iluminación a partir de la banda visible,
segmenta la fruta y calcula el vector de características que consume el modelo.

Todas las funciones son puras (sin estado). El estado (iluminación suavizada, salud de la
cámara) vive en ``agent.ModeloDelMundo``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

# Luminancia nominal de la banda en condiciones de calibración (se recalcula al entrenar).
BELT_L_NOMINAL_POR_DEFECTO = 85.0
BORDE_PX = 8                                  # franja de borde usada para observar la banda
NOMBRES_CARACTERISTICAS = (
    "area_ratio", "elongacion", "solidez",
    "L_med", "a_med", "b_med", "L_std", "a_std", "b_std",
    "H_sin", "H_cos", "S_med", "V_med", "S_std", "V_std",
    *[f"hist_h{i}" for i in range(12)],
    "frac_oscuro", "frac_marron", "frac_claro_bajo_s",
    "anom_frac", "anom_n", "anom_mayor", "anom_dL", "anom_da", "anom_db",
    "pecas_n", "textura_lap",
)


@dataclass
class ResultadoValidacion:
    ok: bool
    codigo: str            # OK | SENSOR_SIN_DATO | TIPO_INVALIDO | VACIA | FORMATO_INVALIDO | NAN | OSCURA | SATURADA
    brillo: float = float("nan")


def validar_imagen(img: Any) -> ResultadoValidacion:
    """Comprobaciones de integridad. Nunca lanza excepciones."""
    try:
        if img is None:
            return ResultadoValidacion(False, "SENSOR_SIN_DATO")
        if not isinstance(img, np.ndarray):
            return ResultadoValidacion(False, "TIPO_INVALIDO")
        if img.size == 0:
            return ResultadoValidacion(False, "VACIA")
        if img.ndim != 3 or img.shape[2] != 3:
            return ResultadoValidacion(False, "FORMATO_INVALIDO")
        if img.dtype.kind == "f" and not np.isfinite(img).all():
            return ResultadoValidacion(False, "NAN")
        if img.shape[0] < 32 or img.shape[1] < 32:
            return ResultadoValidacion(False, "FORMATO_INVALIDO")
        brillo = float(np.mean(img))
        if brillo < 8:
            return ResultadoValidacion(False, "OSCURA", brillo)
        if brillo > 245:
            return ResultadoValidacion(False, "SATURADA", brillo)
        return ResultadoValidacion(True, "OK", brillo)
    except Exception:  # pragma: no cover - defensa adicional
        return ResultadoValidacion(False, "TIPO_INVALIDO")


def _a_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)


def observar_banda(img_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
    """Color Lab mediano y luminancia de la franja de borde (la banda es el fondo conocido)."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = BORDE_PX
    franja = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    ref = np.median(franja, axis=0)
    return ref, float(ref[0])


def normalizar_iluminacion(img_bgr: np.ndarray, luminancia_banda: float,
                           belt_l_nominal: float) -> Tuple[np.ndarray, float]:
    """Divide por la ganancia estimada de iluminación (relación entre banda observada y nominal)."""
    ganancia = float(np.clip(luminancia_banda / max(belt_l_nominal, 1e-3), 0.35, 2.5))
    out = np.clip(img_bgr.astype(np.float32) / ganancia, 0, 255).astype(np.uint8)
    return out, ganancia


def segmentar_fruta(img_bgr: np.ndarray, ref_lab: np.ndarray, umbral: float = 24.0
                    ) -> Tuple[np.ndarray, int, float]:
    """Máscara de la fruta por distancia de color a la banda (Lab) + componente conexa mayor.

    Devuelve (máscara uint8 0/255, n_componentes_relevantes, fracción de la 2.ª componente).
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_s = cv2.GaussianBlur(lab, (0, 0), 1.2)
    dist = np.linalg.norm(lab_s - ref_lab[None, None, :], axis=2)
    m = (dist > umbral).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(m), 0, 0.0
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)
    orden = np.argsort(-areas)
    total = float(m.shape[0] * m.shape[1])
    relevantes = int(np.sum(areas > 0.03 * total))
    mayor = int(orden[0]) + 1
    mask = (etiquetas == mayor).astype(np.uint8) * 255
    # rellenar huecos internos (manchas oscuras que se parecen a la banda)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    relleno = np.zeros_like(mask)
    cv2.drawContours(relleno, cnts, -1, 255, -1)
    segunda = float(areas[orden[1]] / total) if len(orden) > 1 else 0.0
    return relleno, relevantes, segunda


def extraer_caracteristicas(img_bgr: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """Vector de características del interior de la máscara (independiente de la variedad)."""
    sel = mask > 0
    n = int(sel.sum())
    total = float(mask.shape[0] * mask.shape[1])
    if n < 50:
        return None
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    pix_lab = lab[sel]
    pix_hsv = hsv[sel]

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    hull_area = max(cv2.contourArea(cv2.convexHull(c)), 1.0)
    solidez = float(cv2.contourArea(c) / hull_area)
    if len(c) >= 5:
        (_, _), (ma, mi), _ = cv2.fitEllipse(c)
        elong = float(max(ma, mi) / max(min(ma, mi), 1.0))
    else:
        elong = 1.0

    # recorte hacia el interior para evitar el halo del borde
    interior = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    if interior.sum() < 50:
        interior = sel
    lab_i = lab[interior]
    hsv_i = hsv[interior]

    med_lab = np.median(lab_i, axis=0)
    std_lab = lab_i.std(axis=0)
    h_rad = hsv_i[:, 0] * (2 * np.pi / 180.0)
    h_sin, h_cos = float(np.mean(np.sin(h_rad))), float(np.mean(np.cos(h_rad)))
    hist, _ = np.histogram(hsv_i[:, 0], bins=12, range=(0, 180))
    hist = hist.astype(np.float32) / max(hist.sum(), 1)

    v_med = float(np.median(hsv_i[:, 2]))
    frac_oscuro = float(np.mean(hsv_i[:, 2] < 0.45 * max(v_med, 1.0)))
    frac_marron = float(np.mean((hsv_i[:, 0] > 4) & (hsv_i[:, 0] < 24) & (hsv_i[:, 1] > 60) & (hsv_i[:, 2] < 0.5 * max(v_med, 1.0))))
    frac_claro = float(np.mean((hsv_i[:, 1] < 70) & (hsv_i[:, 2] > 1.1 * max(v_med, 1.0))))

    # anomalías: píxeles que se desvían del color mediano de la propia fruta
    lab_s = cv2.GaussianBlur(lab, (0, 0), 1.0)
    d = np.linalg.norm(lab_s - med_lab[None, None, :], axis=2)
    anom = ((d > 26.0) & interior).astype(np.uint8)
    anom = cv2.morphologyEx(anom, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n_c, _, st, _ = cv2.connectedComponentsWithStats(anom, connectivity=8)
    areas = st[1:, cv2.CC_STAT_AREA].astype(np.float32) if n_c > 1 else np.zeros(0, np.float32)
    area_i = float(max(interior.sum(), 1))
    grandes = areas[areas > 0.004 * area_i] if areas.size else areas
    pecas = float(np.sum((areas >= 2) & (areas <= 0.004 * area_i))) if areas.size else 0.0
    anom_frac = float(anom.sum() / area_i)
    anom_mayor = float(areas.max() / area_i) if areas.size else 0.0
    if anom.sum() > 0:
        pl = lab[anom > 0]
        dl, da, db = (pl.mean(axis=0) - med_lab)
    else:
        dl = da = db = 0.0

    gris = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gris, cv2.CV_32F)
    textura = float(lap[interior].var())

    vec = np.array([
        n / total, elong, solidez,
        med_lab[0], med_lab[1], med_lab[2], std_lab[0], std_lab[1], std_lab[2],
        h_sin, h_cos, float(np.median(hsv_i[:, 1])), v_med, float(hsv_i[:, 1].std()), float(hsv_i[:, 2].std()),
        *hist.tolist(),
        frac_oscuro, frac_marron, frac_claro,
        anom_frac, float(len(grandes)), anom_mayor, float(dl), float(da), float(db),
        pecas, textura,
    ], dtype=np.float32)
    assert vec.shape[0] == len(NOMBRES_CARACTERISTICAS)
    return vec


@dataclass
class Observacion:
    """Resultado de procesar una imagen válida."""
    caracteristicas: Optional[np.ndarray]
    n_frutas: int
    area_ratio: float
    ganancia: float
    luminancia_banda: float
    ref_lab: np.ndarray


def procesar_imagen(img: np.ndarray, belt_l_nominal: float,
                    ref_lab_suavizada: Optional[np.ndarray] = None,
                    mezcla_actual: float = 0.6) -> Observacion:
    """Pipeline completo sobre una imagen ya validada.

    ``ref_lab_suavizada`` es la memoria de la banda mantenida por el agente (estado interno); si existe,
    se mezcla con la observación actual para obtener una referencia más estable.
    """
    img_u8 = _a_uint8(img)
    ref_actual, lum = observar_banda(img_u8)
    ref = ref_actual if ref_lab_suavizada is None else (mezcla_actual * ref_actual + (1 - mezcla_actual) * ref_lab_suavizada)
    lum_ref = float(ref[0])
    img_n, ganancia = normalizar_iluminacion(img_u8, lum_ref, belt_l_nominal)
    ref_n, _ = observar_banda(img_n)
    mask, n_rel, segunda = segmentar_fruta(img_n, ref_n)
    area_ratio = float((mask > 0).sum() / mask.size)
    vec = extraer_caracteristicas(img_n, mask) if area_ratio >= 0.02 else None
    return Observacion(vec, n_rel, area_ratio, ganancia, lum, ref_actual)
