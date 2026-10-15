"""Reportes de calidad por lote y proveedor (componente de comunicación)."""
from __future__ import annotations

from typing import Dict

import pandas as pd

from .agent import FreshCheckAgent

COLUMNAS_REPORTE = ["lot_id", "proveedor", "unidades", "fresca", "maduracion", "danada",
                    "desvios_ejecutados", "desvios_fallidos", "revision_manual", "sin_fruta"]


def reporte_desde_agente(agente: FreshCheckAgent) -> pd.DataFrame:
    """Reporte construido con los contadores internos del agente (memoria de largo plazo)."""
    filas = []
    for lot_id, c in sorted(agente.mundo.contadores_lote.items()):
        filas.append(dict(lot_id=lot_id, proveedor=c["proveedor"], unidades=c["n"], fresca=c["fresca"],
                          maduracion=c["maduracion"], danada=c["danada"],
                          desvios_ejecutados=c["desvios_ejecutados"], desvios_fallidos=c["desvios_fallidos"],
                          revision_manual=c["revision_manual"], sin_fruta=c["sin_fruta"]))
    return pd.DataFrame(filas, columns=COLUMNAS_REPORTE)


def reporte_desde_log(log: pd.DataFrame) -> pd.DataFrame:
    """Recalcula el mismo reporte a partir del registro de ejecución (verificación independiente)."""
    filas = []
    for (lot_id, prov), g in log.groupby(["lot_id", "proveedor"], sort=True):
        acc = g["accion"]
        filas.append(dict(
            lot_id=lot_id, proveedor=prov, unidades=len(g),
            fresca=int((g["clase_predicha"] == "fresca").sum()),
            maduracion=int((g["clase_predicha"] == "maduracion").sum()),
            danada=int((g["clase_predicha"] == "danada").sum()),
            desvios_ejecutados=int(((acc == "DESVIAR") & (g["ejecutado"] == True)).sum()),   # noqa: E712
            desvios_fallidos=int(((acc == "DESVIAR") & (g["ejecutado"] == False)).sum()),    # noqa: E712
            revision_manual=int((acc == "REVISION_MANUAL").sum()),
            sin_fruta=int((acc == "NINGUNA").sum())))
    return pd.DataFrame(filas, columns=COLUMNAS_REPORTE)


def consistencia(rep_agente: pd.DataFrame, rep_log: pd.DataFrame) -> float:
    """Fracción de celdas (lote × métrica) idénticas entre ambos reportes."""
    a = rep_agente.sort_values("lot_id").reset_index(drop=True)
    b = rep_log.sort_values("lot_id").reset_index(drop=True)
    if list(a["lot_id"]) != list(b["lot_id"]):
        return 0.0
    cols = [c for c in COLUMNAS_REPORTE if c not in ("lot_id", "proveedor")]
    iguales = int((a[cols].values == b[cols].values).sum())
    return iguales / float(a[cols].size)
