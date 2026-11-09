"""Linea base (Hito 1): ResNet-50 congelado + cabeza lineal (SGD, perdida logistica), 3 semillas.

Reporta: piso (clase mayoritaria), metricas en VALIDACION (global y por fruta) y generalizacion a una
fruta no vista (se entrena con 4 frutas y se evalua en la quinta). La particion de prueba NO se usa:
queda intacta hasta la evaluacion final (Hito 3).
Uso:  python -m src.train_baseline
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from . import config
from .config import Rutas
from .metricas import estado_umbrales, guardar_matriz_confusion, metricas


def cargar(rutas: Rutas, ruta_features: Path | None = None):
    f = np.load(ruta_features or rutas.features, allow_pickle=False)
    ids = list(f["ids"])
    X = f["X"]
    m = pd.read_csv(rutas.manifiesto).merge(pd.read_csv(rutas.particiones), on="id")
    orden = {i: k for k, i in enumerate(ids)}
    faltan = [i for i in m["id"] if i not in orden]
    if faltan:
        raise SystemExit(f"{len(faltan)} imagenes del manifiesto no tienen features (ej. {faltan[0]}). Re-ejecute src.features.")
    idx = m["id"].map(orden).to_numpy()
    return m, X[idx], json.loads(str(f["meta"])) if "meta" in f.files else {}


def ajustar(Xtr, ytr, alfa, semilla):
    esc = StandardScaler().fit(Xtr)
    clf = SGDClassifier(loss="log_loss", alpha=alfa, class_weight="balanced", max_iter=2000, tol=1e-4,
                        random_state=semilla)
    clf.fit(esc.transform(Xtr), ytr)
    return esc, clf


def _num(v):
    return np.nan if v is None else float(v)


def correr(rutas: Rutas, ruta_features: Path | None = None, semillas=config.SEMILLAS, alfas=config.ALFAS):
    m, X, meta = cargar(rutas, ruta_features)
    tr, va = (m["particion"] == "train").to_numpy(), (m["particion"] == "val").to_numpy()
    if not tr.any() or not va.any():
        raise SystemExit("Faltan imagenes en train o val. Revise src.split.")
    ytr, yva = m.loc[tr, "clase"].to_numpy(), m.loc[va, "clase"].to_numpy()

    # piso: clase mayoritaria
    piso = DummyClassifier(strategy="most_frequent").fit(X[tr], ytr)
    m_piso = metricas(yva, piso.predict(X[va]))

    # cuadricula alfa x semilla, seleccion por macro-F1 medio en validacion
    filas = []
    for alfa in alfas:
        for s in semillas:
            esc, clf = ajustar(X[tr], ytr, alfa, s)
            r = metricas(yva, clf.predict(esc.transform(X[va])))
            filas.append({"alfa": alfa, "semilla": s, "exactitud_val": r["exactitud"], "macro_f1_val": r["macro_f1"]})
    grid = pd.DataFrame(filas)
    mejor_alfa = float(grid.groupby("alfa")["macro_f1_val"].mean().idxmax())

    # corridas finales con el mejor alfa (validacion)
    por_semilla, errores, ref = [], None, {}
    mv = m.loc[va].reset_index(drop=True)
    for s in semillas:
        esc, clf = ajustar(X[tr], ytr, mejor_alfa, s)
        Xv = esc.transform(X[va])
        pred, proba = clf.predict(Xv), clf.predict_proba(Xv)
        r = metricas(yva, pred)
        r["semilla"] = s
        por_semilla.append(r)
        if s == semillas[0]:
            cls = list(clf.classes_)
            err = mv.assign(y_real=yva, y_pred=pred, p_pred=proba.max(axis=1),
                            p_real=[proba[i, cls.index(c)] for i, c in enumerate(yva)])
            errores = err[err.y_real != err.y_pred][["id", "fruta", "y_real", "y_pred", "p_pred", "p_real"]]
            ref = {"global": r,
                   "por_fruta": {f: metricas(yva[(mv["fruta"] == f).to_numpy()], pred[(mv["fruta"] == f).to_numpy()])
                                 for f in config.FRUTAS if (mv["fruta"] == f).any()}}

    # generalizacion a fruta no vista: entrena sin la fruta f y evalua en f (train+val de f; nunca prueba)
    gen_filas = []
    for fruta in config.FRUTAS:
        sel = (m["fruta"] == fruta).to_numpy()
        entren = ((m["particion"] == "train").to_numpy()) & ~sel
        evalu = sel & m["particion"].isin(["train", "val"]).to_numpy()
        if not sel.any() or not entren.any() or not evalu.any():
            continue
        yent = m.loc[entren, "clase"].to_numpy()
        if len(set(yent)) < 2:
            continue
        for s in semillas:
            esc, clf = ajustar(X[entren], yent, mejor_alfa, s)
            r = metricas(m.loc[evalu, "clase"].to_numpy(), clf.predict(esc.transform(X[evalu])))
            gen_filas.append({"fruta": fruta, "semilla": s, "n": r["n"], "exactitud": r["exactitud"],
                              "macro_f1": r["macro_f1"], "recall_danada": _num(r["recall_danada"])})
    gen = pd.DataFrame(gen_filas)
    gen_por_fruta = gen.groupby("fruta")[["exactitud", "macro_f1", "recall_danada"]].mean() if len(gen) else None
    macro_f1_nv = float(gen_por_fruta["macro_f1"].mean()) if gen_por_fruta is not None else None

    estado = estado_umbrales(ref["global"], macro_f1_nv, config.cargar_umbrales(rutas))
    f1 = np.array([r["macro_f1"] for r in por_semilla])
    ex = np.array([r["exactitud"] for r in por_semilla])
    sd = (lambda a: float(a.std(ddof=1)) if len(a) > 1 else 0.0)
    resultado = {
        "conjunto": "validacion (la particion de prueba no se uso)",
        "clases": list(config.CLASES),
        "n_train": int(tr.sum()), "n_val": int(va.sum()),
        "piso_clase_mayoritaria": {"exactitud": m_piso["exactitud"], "macro_f1": m_piso["macro_f1"]},
        "linea_base": {"modelo": "ResNet-50 congelado + cabeza lineal (SGD log_loss, class_weight=balanced)",
                       "alfa": mejor_alfa, "semillas": list(semillas),
                       "macro_f1_media": float(f1.mean()), "macro_f1_desv": sd(f1),
                       "exactitud_media": float(ex.mean()), "exactitud_desv": sd(ex)},
        "semilla_referencia": {"semilla": semillas[0], "validacion_global": ref["global"], "validacion_por_fruta": ref["por_fruta"]},
        "generalizacion_fruta_no_vista": {
            "macro_f1_media_5_frutas": macro_f1_nv,
            "por_fruta": None if gen_por_fruta is None else gen_por_fruta.round(4).to_dict(orient="index")},
        "estado_umbrales_indicativo": estado,
        "por_semilla": [{k: r[k] for k in ("semilla", "exactitud", "macro_f1", "recall_danada")} for r in por_semilla],
    }
    sal = rutas.baseline
    sal.mkdir(parents=True, exist_ok=True)
    (sal / "resultados_val.json").write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    grid.to_csv(sal / "grid_val.csv", index=False)
    errores.to_csv(sal / "errores_val.csv", index=False)
    gen.to_csv(sal / "generalizacion_fruta_no_vista.csv", index=False)
    guardar_matriz_confusion(ref["global"]["matriz_confusion"], config.CLASES, sal / "matriz_confusion_val.png",
                             f"Linea base, validacion (semilla {semillas[0]})")
    entorno = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
               "scikit_learn": sklearn.__version__, "semillas": list(semillas), "alfas": list(alfas),
               "particion_semilla": config.SEED, "features": meta,
               "sha256_splits": hashlib.sha256(rutas.particiones.read_bytes()).hexdigest()}
    (sal / "entorno.json").write_text(json.dumps(entorno, indent=2, ensure_ascii=False), encoding="utf-8")
    return resultado


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", type=Path, default=config.ROOT)
    ap.add_argument("--features", type=Path, default=None)
    a = ap.parse_args(argv)
    r = correr(Rutas(a.raiz), a.features)
    lb, piso = r["linea_base"], r["piso_clase_mayoritaria"]
    print(f"Piso (mayoritaria): exactitud {piso['exactitud']:.3f} | macro-F1 {piso['macro_f1']:.3f}")
    print(f"Linea base: macro-F1 {lb['macro_f1_media']:.3f} +/- {lb['macro_f1_desv']:.3f} | "
          f"exactitud {lb['exactitud_media']:.3f} +/- {lb['exactitud_desv']:.3f} (alfa={lb['alfa']}, validacion)")
    g = r["generalizacion_fruta_no_vista"]
    if g["por_fruta"]:
        print("Fruta no vista (macro-F1): " + " | ".join(f"{k} {v['macro_f1']:.3f}" for k, v in g["por_fruta"].items()))
    for e in r["estado_umbrales_indicativo"]:
        v = "n/d" if e["valor"] is None else f"{e['valor']:.3f}"
        print(f"  {e['metrica']:<28} {v:>6}  (umbral {e['umbral']})  {'cumple' if e['cumple'] else 'no cumple' if e['cumple'] is False else 'n/d'}")
    print(f"-> {a.raiz / 'experiments' / 'baseline'}")


if __name__ == "__main__":
    main()
