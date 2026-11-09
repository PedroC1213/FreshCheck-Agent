"""Construye data/manifest.csv desde data/raw/fruitvision/<clase>/<fruta>__<id>.<ext>.

Pasos: (1) verifica que cada imagen abra, (2) calcula un dHash y agrupa duplicados casi identicos
(para que nunca queden en particiones distintas).

Uso:  python -m src.make_manifest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import config
from .config import Rutas


def dhash(img: Image.Image, n: int = 8) -> np.uint64:
    g = img.convert("L").resize((n + 1, n), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


def _popcount(x: np.ndarray) -> np.ndarray:
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    return np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)


def agrupar_duplicados(hashes: np.ndarray, umbral: int, max_grupo: int = config.MAX_GRUPO):
    """Une imagenes con distancia de Hamming <= umbral. Un grupo con mas de max_grupo miembros se considera un
    falso positivo del hash (fondos lisos, encuadres parecidos) y se deshace. Devuelve (etiquetas, n_deshechos)."""
    n = len(hashes)
    padre = np.arange(n)

    def raiz(i):
        while padre[i] != i:
            padre[i] = padre[padre[i]]
            i = padre[i]
        return i

    for i in range(n - 1):
        d = _popcount(hashes[i + 1:] ^ hashes[i])
        for j in np.nonzero(d <= umbral)[0] + i + 1:
            ri, rj = raiz(i), raiz(j)
            if ri != rj:
                padre[rj] = ri
    raices = np.array([raiz(i) for i in range(n)])
    _, etiquetas = np.unique(raices, return_inverse=True)
    etiquetas = etiquetas.astype(np.int64)
    tam = np.bincount(etiquetas)
    grandes = np.nonzero(tam > max_grupo)[0]
    siguiente = int(etiquetas.max()) + 1 if n else 0
    for g in grandes:
        for i in np.nonzero(etiquetas == g)[0]:
            etiquetas[i] = siguiente
            siguiente += 1
    return etiquetas, int(len(grandes))


def construir(rutas: Rutas) -> pd.DataFrame:
    filas, corruptas = [], []
    for clase in config.CLASES:
        carpeta = rutas.fruitvision / clase
        if not carpeta.is_dir():
            continue
        for p in sorted(carpeta.iterdir()):
            if p.suffix.lower() not in config.EXTENSIONES:
                continue
            rel = p.relative_to(rutas.raw).as_posix()
            try:
                with Image.open(p) as im:
                    im.load()
                    ancho, alto = im.size
                    h = dhash(im)
            except Exception as e:  # imagen ilegible
                corruptas.append((rel, str(e)))
                continue
            fruta = p.stem.split("__")[0] if "__" in p.stem else "desconocida"
            if fruta not in config.FRUTAS:
                fruta = "desconocida"
            filas.append({"id": rel, "clase": clase, "fruta": fruta, "ancho": ancho, "alto": alto, "dhash": int(h)})
    if not filas:
        sys.exit(f"No se encontraron imagenes en {rutas.fruitvision}. Ejecute primero src.import_fruitvision (ver data/README.md).")
    df = pd.DataFrame(filas)

    etiquetas, deshechos = agrupar_duplicados(df["dhash"].to_numpy(dtype=np.uint64), config.UMBRAL_HAMMING)
    df["grupo"] = [f"g{g:06d}" for g in etiquetas]
    df["dhash"] = df["dhash"].map(lambda v: f"{v:016x}")

    if corruptas:
        print(f"Aviso: {len(corruptas)} imagenes ilegibles omitidas (ej. {corruptas[0][0]}).")
    if deshechos:
        print(f"Aviso: {deshechos} grupos de mas de {config.MAX_GRUPO} imagenes se deshicieron (hash demasiado permisivo "
              "para estas imagenes); los duplicados dentro de ellos no estan controlados.")
    desconocidas = int((df["fruta"] == "desconocida").sum())
    if desconocidas:
        print(f"Aviso: {desconocidas} imagenes con fruta 'desconocida' (el nombre no empieza con una fruta valida).")
    tam = df.groupby("grupo").size()
    conflicto = df.groupby("grupo")["clase"].nunique()
    print(f"Imagenes: {len(df)} | grupos de duplicados: {(tam > 1).sum()} | grupo mas grande: {tam.max()}")
    if (conflicto > 1).any():
        print(f"Aviso: {(conflicto > 1).sum()} grupos con etiquetas contradictorias; revisar a mano:")
        print(df[df.grupo.isin(conflicto[conflicto > 1].index)].sort_values("grupo")[["grupo", "id", "clase"]].head(20).to_string(index=False))
    return df.sort_values("id").reset_index(drop=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", type=Path, default=config.ROOT)
    a = ap.parse_args(argv)
    rutas = Rutas(a.raiz)
    df = construir(rutas)
    rutas.manifiesto.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(rutas.manifiesto, index=False)
    print(df.pivot_table(index="fruta", columns="clase", values="id", aggfunc="count", fill_value=0).to_string())
    print(f"-> {rutas.manifiesto}")


if __name__ == "__main__":
    main()
