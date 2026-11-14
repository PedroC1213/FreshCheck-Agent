"""Redimensiona todas las imagenes del manifiesto (lado menor -> LADO, recorte central LADOxLADO) y las guarda
en un unico .npy uint8. Acelera el ajuste fino: las fotos originales son muy grandes (hasta 4160 px).

Uso:  python -m src.cache [--lado 256]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import config
from .config import Rutas


def construir(rutas: Rutas, lado: int = 256, salida: Path | None = None) -> Path:
    m = pd.read_csv(rutas.manifiesto)
    salida = salida or rutas.cache_imgs(lado)
    salida.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(salida, mode="w+", dtype=np.uint8, shape=(len(m), lado, lado, 3))
    for i, rel in enumerate(m["id"]):
        with Image.open(rutas.raw / rel) as im:
            if im.format == "JPEG":
                im.draft("RGB", (lado * 2, lado * 2))   # decodificacion reducida (mucho mas rapida)
            im = im.convert("RGB")
            w, h = im.size
            e = lado / min(w, h)
            im = im.resize((max(lado, round(w * e)), max(lado, round(h * e))), Image.BILINEAR)
            w, h = im.size
            l, t = (w - lado) // 2, (h - lado) // 2
            arr[i] = np.asarray(im.crop((l, t, l + lado, t + lado)))
        if i % 500 == 0:
            print(f"  {i}/{len(m)}", flush=True)
    arr.flush()
    salida.with_suffix(".ids.json").write_text(json.dumps(m["id"].tolist()), encoding="utf-8")
    print(f"-> {salida}  {arr.shape}")
    return salida


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", type=Path, default=config.ROOT)
    ap.add_argument("--lado", type=int, default=256)
    a = ap.parse_args(argv)
    construir(Rutas(a.raiz), a.lado)


if __name__ == "__main__":
    main()
