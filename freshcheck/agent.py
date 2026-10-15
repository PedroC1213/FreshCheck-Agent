"""
FreshCheck-Agent: agente reflejo basado en modelo.

Ciclo explícito (función del agente):

    percepto ──► percibir() ──► decidir() ──► actuar() ──► efecto sobre la banda
                    │ actualiza        │ lee
                    └──── ModeloDelMundo (estado interno) ────┘

* ``percibir``  recibe el percepto del entorno, lo valida, estima la iluminación y segmenta la
  fruta; además ACTUALIZA el modelo del mundo (salud de la cámara, banda de referencia, velocidad).
* ``decidir``   infiere el estado de la fruta con el modelo entrenado y aplica la regla
  condición-acción, condicionada al estado interno (por ejemplo, cámara en falla).
* ``actuar``    envía la orden al desviador (interfaz de herramienta) y reporta el efecto.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from .actuator import InterfazDesviador, ResultadoActuador
from .config import CLASES, DISTANCIA_CAMARA_DESVIADOR_M, MARGEN_SEGURIDAD
from .model import ClasificadorEstado
from .perception import Observacion, procesar_imagen, validar_imagen
from .simulator import Percepto


# --------------------------------------------------------------------------- #
# Estado interno (modelo del mundo)
# --------------------------------------------------------------------------- #
@dataclass
class ModeloDelMundo:
    belt_lab_ema: Optional[np.ndarray] = None       # color de la banda suavizado en el tiempo
    estado_camara: str = "OK"                       # OK | DEGRADADA | EN_FALLA
    invalidos_consecutivos: int = 0
    validos_consecutivos: int = 0
    velocidad_est: float = 0.5                      # m/s, suavizada
    recientes: Deque[str] = field(default_factory=lambda: deque(maxlen=20))
    contadores_lote: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    ultimo_unit_id: int = -1
    alerta_lote: bool = False                       # proporción alta de daño en las últimas 20 clasificaciones

    # --- reglas de actualización ---
    def actualizar_percepcion(self, valida: bool, ref_lab: Optional[np.ndarray],
                              velocidad: Optional[float]) -> None:
        if velocidad is not None and np.isfinite(velocidad) and velocidad > 0:
            self.velocidad_est = 0.7 * self.velocidad_est + 0.3 * float(velocidad)
        if valida:
            self.invalidos_consecutivos = 0
            self.validos_consecutivos += 1
            if ref_lab is not None:
                self.belt_lab_ema = ref_lab.copy() if self.belt_lab_ema is None else 0.7 * self.belt_lab_ema + 0.3 * ref_lab
            if self.estado_camara == "EN_FALLA":
                if self.validos_consecutivos >= 2:          # histéresis de recuperación
                    self.estado_camara = "OK"
            else:
                self.estado_camara = "OK"
        else:
            self.validos_consecutivos = 0
            self.invalidos_consecutivos += 1
            self.estado_camara = "EN_FALLA" if self.invalidos_consecutivos >= 3 else "DEGRADADA"

    def registrar_decision(self, lot_id: str, proveedor: str, clase: Optional[str], accion: str,
                           ejecutado: bool) -> None:
        c = self.contadores_lote.setdefault(lot_id, dict(
            proveedor=proveedor, n=0, fresca=0, maduracion=0, danada=0,
            desvios_ejecutados=0, desvios_fallidos=0, revision_manual=0, sin_fruta=0))
        c["n"] += 1
        if clase in CLASES:
            c[clase] += 1
            self.recientes.append(clase)
        if accion == "DESVIAR":
            c["desvios_ejecutados" if ejecutado else "desvios_fallidos"] += 1
        elif accion == "REVISION_MANUAL":
            c["revision_manual"] += 1
        elif accion == "NINGUNA":
            c["sin_fruta"] += 1
        if len(self.recientes) >= 10:
            self.alerta_lote = (sum(1 for x in self.recientes if x == "danada") / len(self.recientes)) > 0.5


# --------------------------------------------------------------------------- #
# Estructuras del ciclo
# --------------------------------------------------------------------------- #
@dataclass
class Percepcion:
    codigo: str                          # OK o código de falla de validación
    brillo: float = float("nan")
    obs: Optional[Observacion] = None


@dataclass
class Decision:
    clase: Optional[str]
    proba: Optional[np.ndarray]
    accion: str                          # CONTINUAR | DESVIAR | REVISION_MANUAL | NINGUNA
    motivo: str


@dataclass
class Efecto:
    ejecutado: bool
    motivo: str
    destino: str                         # descarte | empaque_final | inspeccion_manual | ninguno


# --------------------------------------------------------------------------- #
# Agente
# --------------------------------------------------------------------------- #
class FreshCheckAgent:
    def __init__(self, clasificador: ClasificadorEstado, desviador: InterfazDesviador,
                 usar_modelo_del_mundo: bool = True):
        self.clasificador = clasificador
        self.desviador = desviador
        self.mundo = ModeloDelMundo()
        self.usar_modelo_del_mundo = usar_modelo_del_mundo   # False = agente reflejo simple (ablación)

    # ---------------- percibir ----------------
    def percibir(self, p: Percepto) -> Percepcion:
        val = validar_imagen(p.imagen)
        obs: Optional[Observacion] = None
        if val.ok:
            try:
                ref = self.mundo.belt_lab_ema if self.usar_modelo_del_mundo else None
                obs = procesar_imagen(p.imagen, self.clasificador.belt_l_nominal, ref)
            except Exception:                          # nunca debe caerse la línea
                val = type(val)(False, "FORMATO_INVALIDO", val.brillo)
        if self.usar_modelo_del_mundo:
            self.mundo.actualizar_percepcion(val.ok, obs.ref_lab if obs else None, p.velocidad_banda)
        else:
            self.mundo.velocidad_est = p.velocidad_banda or self.mundo.velocidad_est
        self.mundo.ultimo_unit_id = p.unit_id
        return Percepcion(codigo=val.codigo if not val.ok else "OK", brillo=val.brillo, obs=obs)

    # ---------------- decidir ----------------
    def decidir(self, per: Percepcion) -> Decision:
        if per.codigo != "OK":
            return Decision(None, None, "REVISION_MANUAL", per.codigo)
        if self.usar_modelo_del_mundo and self.mundo.estado_camara == "EN_FALLA":
            return Decision(None, None, "REVISION_MANUAL", "CAMARA_EN_FALLA")
        obs = per.obs
        if obs is None or obs.area_ratio < 0.02:
            return Decision(None, None, "NINGUNA", "SIN_FRUTA")
        solidez = float(obs.caracteristicas[2]) if obs.caracteristicas is not None else 1.0
        if obs.n_frutas >= 2 or obs.area_ratio > 0.60 or solidez < 0.94:   # frutas en contacto / forma no convexa
            return Decision(None, None, "REVISION_MANUAL", "SOLAPAMIENTO")
        if obs.caracteristicas is None:
            return Decision(None, None, "REVISION_MANUAL", "SEGMENTACION_FALLIDA")
        proba = self.clasificador.predecir_proba(obs.caracteristicas)
        clase = CLASES[int(np.argmax(proba))]
        accion = "DESVIAR" if clase == "danada" else "CONTINUAR"       # regla condición-acción
        return Decision(clase, proba, accion, "CLASIFICADA")

    # ---------------- actuar ----------------
    def actuar(self, d: Decision, p: Percepto, latencia_ms: float, deadline_ms: float) -> Efecto:
        if d.accion == "DESVIAR":
            t_act = p.t_s + DISTANCIA_CAMARA_DESVIADOR_M / max(self.mundo.velocidad_est, 0.05)
            r: ResultadoActuador = self.desviador.accionar(p.unit_id, t_act, latencia_ms, deadline_ms)
            return Efecto(r.ejecutado, r.motivo, "descarte" if r.ejecutado else "empaque_final")
        if d.accion == "CONTINUAR":
            return Efecto(True, "OK", "empaque_final")
        if d.accion == "REVISION_MANUAL":
            return Efecto(True, "OK", "inspeccion_manual")
        return Efecto(True, "OK", "ninguno")

    # ---------------- ciclo completo ----------------
    def paso(self, p: Percepto) -> Dict[str, Any]:
        t0 = time.perf_counter()
        per = self.percibir(p)
        dec = self.decidir(per)
        latencia_ms = (time.perf_counter() - t0) * 1000.0
        deadline_ms = DISTANCIA_CAMARA_DESVIADOR_M / max(self.mundo.velocidad_est, 0.05) * 1000.0 * MARGEN_SEGURIDAD
        efecto = self.actuar(dec, p, latencia_ms, deadline_ms)
        self.mundo.registrar_decision(p.lot_id, p.proveedor, dec.clase, dec.accion, efecto.ejecutado)
        pr = dec.proba if dec.proba is not None else [np.nan] * 3
        return dict(
            t_sim_s=round(p.t_s, 4), unit_id=p.unit_id, lot_id=p.lot_id, proveedor=p.proveedor,
            estado_percepcion=per.codigo, brillo=round(per.brillo, 2) if per.brillo == per.brillo else "",
            velocidad_lectura=round(p.velocidad_banda, 4) if p.velocidad_banda else "",
            estado_camara=self.mundo.estado_camara,
            ganancia_est=round(per.obs.ganancia, 3) if per.obs else "",
            n_frutas=per.obs.n_frutas if per.obs else "",
            clase_predicha=dec.clase or "", p_fresca=round(float(pr[0]), 4), p_maduracion=round(float(pr[1]), 4),
            p_danada=round(float(pr[2]), 4), accion=dec.accion, motivo_decision=dec.motivo,
            ejecutado=efecto.ejecutado, motivo_actuador=efecto.motivo, destino=efecto.destino,
            latencia_ms=round(latencia_ms, 3), deadline_ms=round(deadline_ms, 1),
            tardia=bool(latencia_ms > deadline_ms), alerta_lote=self.mundo.alerta_lote,
        )
