"""Busqueda de hiperparametros en VALIDACION (semilla 42, sin tocar prueba): tasa de aprendizaje x capas descongeladas.
La cantidad de epocas se resuelve con la mejor epoca en validacion dentro del maximo --epocas.

Uso:  python -m src.tune [--arqs resnet50 efficientnet_b0] [--epocas 6]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import config
from .config import Rutas
from .finetune import Cfg, cargar_datos, dispositivo, entrenar

GRID_LR = (1e-4, 3e-4)
GRID_DESCONGELAR = ("ultimo_bloque", "todo")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", type=Path, default=config.ROOT)
    ap.add_argument("--arqs", nargs="+", default=list(config.ARQUITECTURAS))
    ap.add_argument("--epocas", type=int, default=6)
    ap.add_argument("--lado", type=int, default=256)
    ap.add_argument("--tam", type=int, default=224)
    ap.add_argument("--sin-pesos", action="store_true", help="solo para pruebas")
    a = ap.parse_args(argv)
    rutas = Rutas(a.raiz)
    dev = dispositivo()
    datos = cargar_datos(rutas, a.lado)
    salida = rutas.final / "tuning"
    salida.mkdir(parents=True, exist_ok=True)
    filas, mejores = [], {}
    for arq in a.arqs:
        for lr in GRID_LR:
            for desc in GRID_DESCONGELAR:
                cfg = Cfg(arq=arq, semilla=config.SEMILLAS[0], lr=lr, epocas=a.epocas, descongelar=desc,
                          lado=a.lado, tam=a.tam, sin_pesos=a.sin_pesos)
                r = entrenar(cfg, datos, dev)
                filas.append({"arq": arq, "lr": lr, "descongelar": desc, "macro_f1_val": r["val"]["macro_f1"],
                              "exactitud_val": r["val"]["exactitud"], "mejor_epoca": r["mejor_epoca"], "segundos": r["segundos"]})
                pd.DataFrame(filas).to_csv(salida / "grid_val.csv", index=False)
    df = pd.DataFrame(filas)
    for arq in a.arqs:
        sub = df[df.arq == arq].sort_values(["macro_f1_val", "exactitud_val"], ascending=False).iloc[0]
        mejores[arq] = {"lr": float(sub["lr"]), "descongelar": sub["descongelar"], "epocas": a.epocas,
                        "mejor_epoca": int(sub["mejor_epoca"]), "macro_f1_val": float(sub["macro_f1_val"])}
    (salida / "mejor_config.json").write_text(json.dumps(mejores, indent=2), encoding="utf-8")
    print(df.to_string(index=False))
    print(json.dumps(mejores, indent=2))


if __name__ == "__main__":
    main()
