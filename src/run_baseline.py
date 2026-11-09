"""Ejecuta toda la linea base: manifiesto -> particion -> features -> entrenamiento/evaluacion en validacion.

Uso:  python -m src.run_baseline            (usa experiments/features/ si ya existe)
      python -m src.run_baseline --recalcular-features
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import config, features, make_manifest, split, train_baseline
from .config import Rutas


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", type=Path, default=config.ROOT)
    ap.add_argument("--recalcular-features", action="store_true")
    a = ap.parse_args(argv)
    raiz = str(a.raiz)
    make_manifest.main(["--raiz", raiz])
    split.main(["--raiz", raiz])
    rutas = Rutas(a.raiz)
    if a.recalcular_features or not rutas.features.exists():
        features.extraer(rutas)
    train_baseline.main(["--raiz", raiz])


if __name__ == "__main__":
    main()
