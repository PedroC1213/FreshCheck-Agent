"""Ejecución de escenarios sobre el simulador y cálculo de las medidas de desempeño (M1-M6)."""
from __future__ import annotations

from datetime import datetime
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .actuator import DesviadorSimulado
from .agent import FreshCheckAgent
from .model import ClasificadorEstado
from .simulator import Escenario, LineaEmpaqueSimulada


def ejecutar_escenario(clasificador: ClasificadorEstado, esc: Escenario, n_unidades: int, seed: int,
                       rep: int = 0, usar_modelo_del_mundo: bool = True,
                       ) -> Tuple[pd.DataFrame, FreshCheckAgent]:
    """Corre el agente sobre un flujo simulado y devuelve el log (percepción, decisión, acción, verdad)."""
    linea = LineaEmpaqueSimulada(esc, seed)
    agente = FreshCheckAgent(clasificador, DesviadorSimulado(), usar_modelo_del_mundo)
    filas = []
    run_id = f"{esc.nombre}_rep{rep}_seed{seed}"
    for percepto in linea.flujo(n_unidades):
        fila = agente.paso(percepto)
        v = linea.verdad[percepto.unit_id]
        fila.update(run_id=run_id, escenario=esc.nombre, rep=rep, seed=seed,
                    ts_ejecucion=datetime.now().isoformat(timespec="milliseconds"),
                    clase_real=v.clase or "", variedad=v.variedad or "", falla_inyectada=v.falla or "")
        filas.append(fila)
    return pd.DataFrame(filas), agente


# --------------------------------------------------------------------------- #
# Medidas
# --------------------------------------------------------------------------- #
def _validas(df: pd.DataFrame) -> pd.DataFrame:
    """Unidades con fruta única, sin falla inyectada (base de M1-M3)."""
    return df[(df["clase_real"] != "") & (df["falla_inyectada"] == "")]


def m1_exactitud(df: pd.DataFrame) -> float:
    v = _validas(df)
    return float((v["clase_predicha"] == v["clase_real"]).mean()) if len(v) else float("nan")


def m2_reduccion_defectuosas(df: pd.DataFrame) -> Tuple[float, int, int]:
    """R = (D0 - D1) / D0.  D0: dañadas que llegarían al empaque sin agente; D1: con agente."""
    v = _validas(df)
    d = v[v["clase_real"] == "danada"]
    d0 = len(d)
    d1 = int((d["destino"] == "empaque_final").sum())
    return ((d0 - d1) / d0 if d0 else float("nan")), d0, d1


def m3_descarte_indebido(df: pd.DataFrame) -> float:
    v = _validas(df)
    nd = v[v["clase_real"] != "danada"]
    return float((nd["destino"] == "descarte").mean()) if len(nd) else float("nan")


def m4_latencia(df: pd.DataFrame) -> Tuple[float, int]:
    """(p95 de la latencia en ms, número de decisiones fuera de plazo)."""
    return float(np.percentile(df["latencia_ms"], 95)), int(df["tardia"].sum())


def tasa_defecto_empaque(df: pd.DataFrame) -> Tuple[float, float]:
    """Tasa de unidades dañadas entre las que llegan al empaque final: (sin agente, con agente)."""
    v = _validas(df)
    sin = float((v["clase_real"] == "danada").mean())
    llegan = v[v["destino"] == "empaque_final"]
    con = float((llegan["clase_real"] == "danada").mean()) if len(llegan) else 0.0
    return sin, con
