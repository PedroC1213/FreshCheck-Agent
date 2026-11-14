"""Linea base FINAL (Hito 3): ResNet-50 congelado + cabeza lineal con cuadricula de alfa AMPLIADA (el optimo del
Hito 2, 1e-2, quedo en el borde). Alfa se elige en validacion; la prueba se evalua una vez por semilla.
Tambien mide la generalizacion a fruta no vista y guarda la cabeza para el prototipo.

Requiere experiments/features/resnet50_imagenet.npz (python -m src.run_baseline).
Uso:  python -m src.baseline_final
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .config import Rutas
from .metricas import metricas
from .train_baseline import ajustar, cargar


def _pred(m, idx, clf, esc, X):
    proba = clf.predict_proba(esc.transform(X[idx]))
    p_danada = proba[:, list(clf.classes_).index("danada")]
    d = m.iloc[idx][["id", "fruta"]].copy()
    d["y_real"] = m.iloc[idx]["clase"].to_numpy()
    d["p_danada"] = p_danada
    d["y_pred"] = np.where(p_danada >= 0.5, "danada", "fresca")
    return d[["id", "fruta", "y_real", "y_pred", "p_danada"]].reset_index(drop=True)


def correr(rutas: Rutas, ruta_features: Path | None = None, semillas=config.SEMILLAS, alfas=config.ALFAS_AMPLIADA):
    m, X, meta = cargar(rutas, ruta_features)
    part = m["particion"].to_numpy()
    tr, va, te = (np.nonzero(part == p)[0] for p in config.PARTICIONES)
    ytr, yva = m.iloc[tr]["clase"].to_numpy(), m.iloc[va]["clase"].to_numpy()
    salida = rutas.final / "baseline"
    salida.mkdir(parents=True, exist_ok=True)

    filas = []
    for alfa in alfas:
        for s in semillas:
            esc, clf = ajustar(X[tr], ytr, alfa, s)
            r = metricas(yva, clf.predict(esc.transform(X[va])))
            filas.append({"alfa": alfa, "semilla": s, "exactitud_val": r["exactitud"], "macro_f1_val": r["macro_f1"]})
    grid = pd.DataFrame(filas)
    grid.to_csv(salida / "grid_val.csv", index=False)
    mejor_alfa = float(grid.groupby("alfa")["macro_f1_val"].mean().idxmax())

    for s in semillas:
        esc, clf = ajustar(X[tr], ytr, mejor_alfa, s)
        d = salida / f"s{s}"
        d.mkdir(exist_ok=True)
        _pred(m, va, clf, esc, X).to_csv(d / "pred_val.csv", index=False)
        _pred(m, te, clf, esc, X).to_csv(d / "pred_test.csv", index=False)
        if s == semillas[0]:
            rutas.modelos.mkdir(parents=True, exist_ok=True)
            joblib.dump({"scaler": esc, "clf": clf, "clases": list(clf.classes_), "alfa": mejor_alfa},
                        rutas.modelos / "baseline_head.joblib")

    # fruta no vista: entrena sin la fruta f y evalua en f (train+val de f)
    for fruta in config.FRUTAS:
        sel = (m["fruta"] == fruta).to_numpy()
        ent = np.nonzero((part == "train") & ~sel)[0]
        ev = np.nonzero(sel & np.isin(part, ["train", "val"]))[0]
        if not len(ent) or not len(ev):
            continue
        for s in semillas:
            esc, clf = ajustar(X[ent], m.iloc[ent]["clase"].to_numpy(), mejor_alfa, s)
            _pred(m, ev, clf, esc, X).to_csv(salida / f"s{s}" / f"fnv_{fruta}.csv", index=False)

    info = {"alfa": mejor_alfa, "alfas": list(alfas), "semillas": list(semillas), "features": meta}
    (salida / "resumen.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", type=Path, default=config.ROOT)
    ap.add_argument("--features", type=Path, default=None)
    a = ap.parse_args(argv)
    info = correr(Rutas(a.raiz), a.features)
    print(f"Linea base final: alfa={info['alfa']:g} -> {a.raiz / 'experiments' / 'final' / 'baseline'}")


if __name__ == "__main__":
    main()
