"""Particion 70/15/15 estratificada por (clase, fruta) y por grupo de duplicados.

Se hace sobre imagenes ORIGINALES y antes de cualquier aumento (Hito 1). Semilla fija: 42.
Uso:  python -m src.split
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .config import Rutas


def asignar(manifiesto: pd.DataFrame, fracciones=config.FRACCIONES, semilla=config.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    m = manifiesto.sort_values("id")
    # un grupo tiene UN estrato (el de su primer miembro) y UNA particion para todos sus miembros
    grupos = m.groupby("grupo").agg(clase=("clase", "first"), fruta=("fruta", "first"), n=("id", "size")).reset_index()
    destino = {}
    for _, sub in grupos.groupby(["clase", "fruta"], sort=True):
        sub = sub.iloc[rng.permutation(len(sub))]
        objetivo = np.array(fracciones) * sub["n"].sum()
        conteo = np.zeros(3)
        for _, g in sub.iterrows():
            s = int(np.argmax(objetivo - conteo))
            destino[g["grupo"]] = config.PARTICIONES[s]
            conteo[s] += g["n"]
    out = m[["id", "grupo"]].copy()
    out["particion"] = out["grupo"].map(destino)
    assert out["particion"].notna().all()
    assert (out.groupby("grupo")["particion"].nunique() == 1).all(), "un grupo quedo en dos particiones"
    return out[["id", "particion"]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", type=Path, default=config.ROOT)
    a = ap.parse_args(argv)
    rutas = Rutas(a.raiz)
    m = pd.read_csv(rutas.manifiesto)
    s = asignar(m)
    s.to_csv(rutas.particiones, index=False)
    tabla = m.merge(s, on="id").pivot_table(index=["fruta", "clase"], columns="particion", values="id",
                                            aggfunc="count", fill_value=0)
    tabla = tabla.reindex(columns=list(config.PARTICIONES), fill_value=0)
    print(tabla.to_string())
    vacios = tabla[(tabla == 0).any(axis=1)]
    if len(vacios):
        print("\nAviso: estratos con particion vacia (necesitan mas imagenes):")
        print(vacios.to_string())
    sha = hashlib.sha256(rutas.particiones.read_bytes()).hexdigest()[:16]
    print(f"\n-> {rutas.particiones}  (sha256 {sha})")


if __name__ == "__main__":
    main()
