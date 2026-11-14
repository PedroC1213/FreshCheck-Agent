"""Corridas FINALES del Hito 3 (despues de src.tune): 3 semillas por arquitectura con la mejor configuracion,
evaluacion UNICA en prueba, y generalizacion a fruta no vista de la mejor arquitectura (elegida por validacion).

Salidas: experiments/final/<arq>/s<semilla>/..., experiments/final/<arq>/fnv_<fruta>_s<semilla>.csv, models_v2/<arq>.pt
Uso:  python -m src.run_final
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import numpy as np

from . import config
from .config import Rutas
from .finetune import Cfg, cargar_datos, dispositivo, entrenar, guardar_corrida, guardar_modelo


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", type=Path, default=config.ROOT)
    ap.add_argument("--arqs", nargs="+", default=list(config.ARQUITECTURAS))
    ap.add_argument("--lado", type=int, default=256)
    ap.add_argument("--tam", type=int, default=224)
    ap.add_argument("--sin-pesos", action="store_true", help="solo para pruebas")
    ap.add_argument("--sin-fruta-no-vista", action="store_true")
    a = ap.parse_args(argv)
    rutas = Rutas(a.raiz)
    cfgs = json.loads((rutas.final / "tuning" / "mejor_config.json").read_text(encoding="utf-8"))
    dev = dispositivo()
    datos = cargar_datos(rutas, a.lado)

    val_media = {}
    for arq in a.arqs:
        c = cfgs[arq]
        f1s, epocas = [], []
        for s in config.SEMILLAS:
            cfg = Cfg(arq=arq, semilla=s, lr=c["lr"], epocas=c["epocas"], descongelar=c["descongelar"], lado=a.lado,
                      tam=a.tam, sin_pesos=a.sin_pesos, evaluar_test=True)
            r = entrenar(cfg, datos, dev)
            guardar_corrida(r, rutas.final / arq / f"s{s}")
            if s == config.SEMILLAS[0]:
                guardar_modelo(r["net"], rutas.modelos / f"{arq}.pt", cfg)
            f1s.append(r["val"]["macro_f1"])
            epocas.append(r["mejor_epoca"])
            del r
        val_media[arq] = {"macro_f1_val_media": float(np.mean(f1s)), "mejor_epoca_mediana": int(np.median(epocas))}

    mejor = max(val_media, key=lambda k: val_media[k]["macro_f1_val_media"])   # por VALIDACION, nunca por prueba
    import pandas
    import sklearn
    import torch
    import torchvision
    entorno = {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__,
               "numpy": np.__version__, "pandas": pandas.__version__, "scikit_learn": sklearn.__version__,
               "dispositivo": dev, "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else None}
    info = {"val": val_media, "mejor_arquitectura": mejor, "dispositivo": dev, "entorno": entorno}
    if not a.sin_fruta_no_vista:
        c = cfgs[mejor]
        ep = max(1, val_media[mejor]["mejor_epoca_mediana"])
        for fruta in config.FRUTAS:
            for s in config.SEMILLAS:
                cfg = Cfg(arq=mejor, semilla=s, lr=c["lr"], epocas=ep, descongelar=c["descongelar"], lado=a.lado,
                          tam=a.tam, sin_pesos=a.sin_pesos, modo="fruta_no_vista", fruta=fruta)
                r = entrenar(cfg, datos, dev)
                r["pred_val"].to_csv(rutas.final / mejor / f"fnv_{fruta}_s{s}.csv", index=False)
                del r
    rutas.modelos.mkdir(parents=True, exist_ok=True)
    (rutas.modelos / "meta.json").write_text(json.dumps({"mejor": mejor, "tam": a.tam, "clases": list(config.CLASES)}), encoding="utf-8")
    (rutas.final / "resumen_ejecucion.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
