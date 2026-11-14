"""Ejecuta TODO el Hito 3 en orden: cache -> linea base final -> busqueda de hiperparametros -> corridas finales ->
evaluacion final. Requiere haber ejecutado antes python -m src.run_baseline (Hito 2). Se recomienda GPU (Colab).

Uso:  python -m src.run_hito3 [--saltar-tuning]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import baseline_final, cache, config, evaluar_final, run_final, tune
from .config import Rutas


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", type=Path, default=config.ROOT)
    ap.add_argument("--saltar-tuning", action="store_true", help="reutiliza experiments/final/tuning/mejor_config.json")
    a = ap.parse_args(argv)
    raiz = ["--raiz", str(a.raiz)]
    rutas = Rutas(a.raiz)
    if not rutas.cache_imgs(256).exists():
        cache.main(raiz)
    baseline_final.main(raiz)
    if not a.saltar_tuning:
        tune.main(raiz)
    run_final.main(raiz)
    evaluar_final.main(raiz)


if __name__ == "__main__":
    main()
