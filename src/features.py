"""Extrae vectores de 2048 dimensiones con ResNet-50 (ImageNet) CONGELADO para todas las imagenes del manifiesto.

Linea base del Hito 1: extractor congelado + cabeza lineal. Los pesos se descargan la primera vez
(torchvision, IMAGENET1K_V2). Uso:  python -m src.features
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .config import Rutas


def extraer(rutas: Rutas, lote: int = 64, trabajadores: int = 0) -> Path:
    import torch
    import torchvision
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision.models import ResNet50_Weights, resnet50

    torch.manual_seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    pesos = ResNet50_Weights.IMAGENET1K_V2
    modelo = resnet50(weights=pesos)
    modelo.fc = torch.nn.Identity()
    modelo.eval().to(dispositivo)
    transformar = pesos.transforms()

    m = pd.read_csv(rutas.manifiesto)
    ids = m["id"].tolist()

    class Imgs(Dataset):
        def __len__(self):
            return len(ids)

        def __getitem__(self, i):
            with Image.open(rutas.raw / ids[i]) as im:
                return transformar(im.convert("RGB"))

    salidas = []
    with torch.inference_mode():
        for k, x in enumerate(DataLoader(Imgs(), batch_size=lote, num_workers=trabajadores)):
            salidas.append(modelo(x.to(dispositivo)).cpu().numpy().astype(np.float32))
            if k % 10 == 0:
                print(f"  {min((k + 1) * lote, len(ids))}/{len(ids)}", flush=True)
    X = np.concatenate(salidas)
    meta = {"modelo": "resnet50", "pesos": "IMAGENET1K_V2", "torch": torch.__version__,
            "torchvision": torchvision.__version__, "dispositivo": dispositivo, "n": int(len(ids))}
    rutas.features.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(rutas.features, ids=np.array(ids), X=X, meta=json.dumps(meta))
    print(f"-> {rutas.features}  {X.shape}")
    return rutas.features


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", type=Path, default=config.ROOT)
    ap.add_argument("--lote", type=int, default=64)
    a = ap.parse_args(argv)
    extraer(Rutas(a.raiz), a.lote)


if __name__ == "__main__":
    main()
