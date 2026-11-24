"""Ejecuta TODO el Hito 3 en orden: cache -> linea base final -> busqueda de hiperparametros -> corridas finales ->
evaluacion final. Requiere haber ejecutado antes python -m src.run_baseline (Hito 2). Se recomienda GPU (Colab).

--rapido: solo EfficientNet-B0 y 4 epocas (menos tiempo; sigue cumpliendo ajuste fino frente a extractor congelado).
Uso:  python -m src.run_hito3 [--rapido] [--saltar-tuning]
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
    ap.add_argument("--rapido", action="store_true", help="solo EfficientNet-B0 y 4 epocas")
    ap.add_argument("--arqs", nargs="+", default=None, choices=config.ARQUITECTURAS)
    ap.add_argument("--epocas", type=int, default=None)
    a = ap.parse_args(argv)
    arqs = a.arqs or (["efficientnet_b0"] if a.rapido else list(config.ARQUITECTURAS))
    epocas = a.epocas or (4 if a.rapido else 6)
    raiz = ["--raiz", str(a.raiz)]
    rutas = Rutas(a.raiz)
    if not rutas.cache_imgs(256).exists():
        cache.main(raiz)
    baseline_final.main(raiz)
    if not a.saltar_tuning:
        tune.main(raiz + ["--arqs", *arqs, "--epocas", str(epocas)])
    run_final.main(raiz + ["--arqs", *arqs])
    evaluar_final.main(raiz)


if __name__ == "__main__":
    main()
