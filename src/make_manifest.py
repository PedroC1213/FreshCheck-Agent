"""Construye data/manifest.csv desde data/raw/<fuente>/<clase>/<archivo>.

Pasos: (1) verifica que cada imagen abra, (2) exige fila con licencia para cada imagen de internet,
(3) calcula un dHash y agrupa duplicados casi identicos (para que nunca queden en particiones distintas).

Uso:  python -m src.make_manifest
"""
from __future__ import annotations

import argparse
import io
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


def agrupar_duplicados(hashes: np.ndarray, umbral: int) -> np.ndarray:
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
    return etiquetas


def leer_csv(ruta: Path) -> pd.DataFrame:
    """Lee un CSV tolerando BOM, separador coma/punto y coma/tabulador (Excel en espanol) y UTF-8 o cp1252."""
    crudo = ruta.read_bytes()
    try:
        texto = crudo.decode("utf-8-sig")
    except UnicodeDecodeError:
        texto = crudo.decode("cp1252")
    primera = texto.splitlines()[0] if texto.strip() else ""
    sep = max((",", ";", "\t"), key=primera.count)
    df = pd.read_csv(io.StringIO(texto), sep=sep, dtype=str).fillna("")
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def construir(rutas: Rutas, permitir_sin_licencia: bool = False) -> pd.DataFrame:
    filas, corruptas = [], []
    for fuente in config.FUENTES:
        for clase in config.CLASES:
            carpeta = rutas.raw / fuente / clase
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
                filas.append({"id": rel, "fuente": fuente, "clase": clase, "fruta": fruta,
                              "ancho": ancho, "alto": alto, "dhash": int(h)})
    if not filas:
        sys.exit(f"No se encontraron imagenes en {rutas.raw}. Ver data/README.md para la estructura esperada.")
    df = pd.DataFrame(filas)

    # Licencia obligatoria para internet
    internet = df[df.fuente == "internet"]
    if len(internet):
        if not rutas.fuentes_internet.exists():
            sys.exit(f"Falta {rutas.fuentes_internet} (URL, autor y licencia de cada imagen de internet).")
        fuentes = leer_csv(rutas.fuentes_internet)
        if not {"archivo", "licencia"} <= set(fuentes.columns):
            sys.exit(f"{rutas.fuentes_internet.name} debe tener las columnas 'archivo' y 'licencia'. Encontradas: {list(fuentes.columns)}")
        fuentes["archivo"] = fuentes["archivo"].str.strip()
        con_licencia = set(fuentes.loc[fuentes["licencia"].str.strip() != "", "archivo"])
        sin = [i for i in internet["id"] if Path(i).name not in con_licencia]
        if sin and not permitir_sin_licencia:
            print(f"{len(sin)} imagenes de internet sin fila con licencia en sources_internet.csv, por ejemplo:")
            for s in sin[:10]:
                print("  ", s)
            sys.exit("Complete la licencia de cada imagen (columna 'archivo' = nombre del archivo) o use --permitir-sin-licencia solo para pruebas.")

    hashes = df["dhash"].to_numpy(dtype=np.uint64)
    df["grupo"] = [f"g{g:06d}" for g in agrupar_duplicados(hashes, config.UMBRAL_HAMMING)]
    df["dhash"] = df["dhash"].map(lambda v: f"{v:016x}")

    if corruptas:
        print(f"Aviso: {len(corruptas)} imagenes ilegibles omitidas (ej. {corruptas[0][0]}).")
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
    ap.add_argument("--permitir-sin-licencia", action="store_true", help="solo para pruebas locales")
    a = ap.parse_args(argv)
    rutas = Rutas(a.raiz)
    df = construir(rutas, a.permitir_sin_licencia)
    rutas.manifiesto.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(rutas.manifiesto, index=False)
    print(df.pivot_table(index=["fuente", "clase"], values="id", aggfunc="count").rename(columns={"id": "n"}).to_string())
    print(f"-> {rutas.manifiesto}")


if __name__ == "__main__":
    main()
