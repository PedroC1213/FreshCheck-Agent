"""Agrega las corridas finales (linea base, ResNet-50, EfficientNet-B0): tablas, subgrupos por fruta, fruta no vista,
umbrales del Hito 1, prueba de McNemar, figuras y lista de errores para el analisis (>= 10 casos).

Salida: experiments/final/resumen.json, figuras/*.png, errores_test_<modelo>.csv, docs/analisis_errores.csv (plantilla)
Uso:  python -m src.evaluar_final
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, precision_recall_curve

from . import config
from .config import Rutas
from .metricas import guardar_matriz_confusion, metricas

NOMBRES = {"baseline": "Línea base (ResNet-50 congelado)", "resnet50": "ResNet-50 ajuste fino",
           "efficientnet_b0": "EfficientNet-B0 ajuste fino"}


def _sd(v):
    return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def _por_semilla(carpeta: Path, archivo: str):
    out = {}
    for d in sorted(carpeta.glob("s*")):
        f = d / archivo
        if d.is_dir() and f.exists():
            out[int(d.name[1:])] = pd.read_csv(f)
    return out


def _fnv(carpeta: Path):
    filas = []
    for f in sorted(carpeta.rglob("fnv_*.csv")):
        nombre = f.stem                       # fnv_<fruta>_s<semilla>  o  fnv_<fruta> (en s<semilla>/)
        if f.parent.name.startswith("s") and f.parent != carpeta:
            fruta, semilla = nombre[4:], int(f.parent.name[1:])
        else:
            fruta, semilla = nombre[4:].rsplit("_s", 1)[0], int(nombre.rsplit("_s", 1)[1])
        d = pd.read_csv(f)
        r = metricas(d["y_real"], d["y_pred"])
        filas.append({"fruta": fruta, "semilla": semilla, "n": r["n"], "exactitud": r["exactitud"],
                      "macro_f1": r["macro_f1"], "recall_danada": np.nan if r["recall_danada"] is None else r["recall_danada"]})
    if not filas:
        return None
    df = pd.DataFrame(filas)
    gm = df.groupby("fruta")[["exactitud", "macro_f1", "recall_danada"]].mean()
    return {"por_fruta": {k: {a: float(b) for a, b in v.items()} for k, v in gm.to_dict(orient="index").items()},
            "macro_f1_media": float(gm["macro_f1"].mean()), "exactitud_media": float(gm["exactitud"].mean())}


def resumir(rutas: Rutas) -> dict:
    R = {"modelos": {}}
    for nombre in NOMBRES:
        carpeta = rutas.final / nombre
        tests = _por_semilla(carpeta, "pred_test.csv")
        if not tests:
            continue
        vals = _por_semilla(carpeta, "pred_val.csv")
        met = {s: metricas(d["y_real"], d["y_pred"]) for s, d in tests.items()}
        claves = ("macro_f1", "exactitud", "recall_danada")
        info = {"semillas": sorted(met),
                "test": {k: [met[s][k] for s in sorted(met)] for k in claves},
                "test_media": {k: float(np.mean([met[s][k] for s in met])) for k in claves},
                "test_desv": {k: _sd([met[s][k] for s in met]) for k in claves},
                "precision_danada": float(np.mean([met[s]["por_clase"]["danada"]["precision"] for s in met])),
                "por_clase": {c: {k: float(np.mean([met[s]["por_clase"][c][k] for s in met])) for k in ("precision", "recall", "f1")} for c in config.CLASES},
                "val_macro_f1_media": float(np.mean([metricas(d["y_real"], d["y_pred"])["macro_f1"] for d in vals.values()])) if vals else None,
                "matriz_confusion_primera_semilla": met[sorted(met)[0]]["matriz_confusion"]}
        todo = pd.concat(tests.values())
        pf = {}
        for fruta, g in todo.groupby("fruta"):
            rs = [metricas(d[d.fruta == fruta]["y_real"], d[d.fruta == fruta]["y_pred"]) for d in tests.values()]
            pf[fruta] = {"n": int(len(g) / len(tests)), "macro_f1": float(np.mean([r["macro_f1"] for r in rs])),
                         "exactitud": float(np.mean([r["exactitud"] for r in rs])),
                         "recall_danada": float(np.mean([r["recall_danada"] for r in rs if r["recall_danada"] is not None]))}
        info["por_fruta"] = pf
        info["fruta_no_vista"] = _fnv(carpeta)
        rs = carpeta / "s42" / "resumen.json"
        if rs.exists():
            j = json.loads(rs.read_text(encoding="utf-8"))
            info["segundos_s42"], info["mejor_epoca_s42"], info["config"] = j.get("segundos"), j.get("mejor_epoca"), j.get("config")
        R["modelos"][nombre] = info

    ft = [k for k in ("resnet50", "efficientnet_b0") if k in R["modelos"] and R["modelos"][k]["val_macro_f1_media"] is not None]
    if "baseline" in R["modelos"] and ft:
        mejor = max(ft, key=lambda k: R["modelos"][k]["val_macro_f1_media"])      # elegido por VALIDACION
        R["mejor_modelo"] = mejor
        b, m = R["modelos"]["baseline"], R["modelos"][mejor]
        err_b, err_m = 1 - b["test_media"]["macro_f1"], 1 - m["test_media"]["macro_f1"]
        delta = m["test_media"]["macro_f1"] - b["test_media"]["macro_f1"]
        R["comparacion"] = {"error_base": err_b, "error_modelo": err_m, "delta_macro_f1": delta,
                            "reduccion_error": (err_b - err_m) / err_b if err_b > 0 else None,
                            "desv_max": max(b["test_desv"]["macro_f1"], m["test_desv"]["macro_f1"]),
                            "mayor_que_desv": bool(delta > max(b["test_desv"]["macro_f1"], m["test_desv"]["macro_f1"]))}
        pb, pm = _por_semilla(rutas.final / "baseline", "pred_test.csv"), _por_semilla(rutas.final / mejor, "pred_test.csv")
        s0 = sorted(set(pb) & set(pm))[0]
        a = pb[s0].set_index("id"); c = pm[s0].set_index("id").loc[a.index]
        ok_b, ok_m = (a["y_real"] == a["y_pred"]), (c["y_real"] == c["y_pred"])
        b01, c10 = int((ok_b & ~ok_m).sum()), int((~ok_b & ok_m).sum())
        R["mcnemar"] = {"semilla": s0, "base_acierta_modelo_falla": b01, "base_falla_modelo_acierta": c10,
                        "p_valor": float(binomtest(min(b01, c10), b01 + c10, 0.5).pvalue) if b01 + c10 > 0 else 1.0}
        R["bootstrap"] = bootstrap_ic(rutas, mejor)
        u = config.cargar_umbrales(rutas)
        fnv = m.get("fruta_no_vista")
        filas = [("macro_f1_prueba", m["test_media"]["macro_f1"]), ("exactitud_prueba", m["test_media"]["exactitud"]),
                 ("macro_f1_fruta_no_vista", None if not fnv else fnv["macro_f1_media"]),
                 ("recall_danada_prueba", m["test_media"]["recall_danada"]),
                 ("reduccion_error_sobre_linea_base", R["comparacion"]["reduccion_error"])]
        R["umbrales"] = [{"metrica": k, "valor": v, "umbral": u[k]["umbral"], "cumple": None if v is None else bool(v >= u[k]["umbral"])} for k, v in filas]
        R["umbrales"][-1]["cumple"] = None if R["umbrales"][-1]["valor"] is None else bool(R["umbrales"][-1]["cumple"] and R["comparacion"]["mayor_que_desv"])
    return R


def bootstrap_ic(rutas: Rutas, mejor: str, B: int = 1000, semilla: int = 0) -> dict:
    """IC 95 % por bootstrap sobre las imagenes de prueba (primera semilla comun): macro-F1 de cada modelo y su diferencia."""
    pb, pm = _por_semilla(rutas.final / "baseline", "pred_test.csv"), _por_semilla(rutas.final / mejor, "pred_test.csv")
    s0 = sorted(set(pb) & set(pm))[0]
    a = pb[s0].set_index("id")
    c = pm[s0].set_index("id").loc[a.index]
    y, pa, pc = (a.y_real == "danada").to_numpy(), (a.y_pred == "danada").to_numpy(), (c.y_pred == "danada").to_numpy()
    rng = np.random.default_rng(semilla)
    db, dm = [], []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        db.append(f1_score(y[i], pa[i], average="macro", zero_division=0))
        dm.append(f1_score(y[i], pc[i], average="macro", zero_division=0))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"semilla": s0, "B": B, "base": q(db), "mejor": q(dm), "delta": q(np.array(dm) - np.array(db))}


def calibracion(pred: pd.DataFrame, umbral_revision: float = 0.80, bins=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)) -> dict:
    """Calibracion (ECE en 5 intervalos de confianza) y efecto de la regla de revision humana (confianza < umbral)."""
    conf = np.where(pred.y_pred == "danada", pred.p_danada, 1 - pred.p_danada)
    ok = (pred.y_real == pred.y_pred).to_numpy()
    filas, ece = [], 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (conf >= lo) & (conf < hi)
        if sel.any():
            filas.append({"lo": lo, "n": int(sel.sum()), "confianza": float(conf[sel].mean()), "exactitud": float(ok[sel].mean())})
            ece += abs(ok[sel].mean() - conf[sel].mean()) * sel.sum() / len(ok)
    rev = conf < umbral_revision
    return {"ece": float(ece), "intervalos": filas, "umbral_revision": umbral_revision,
            "fraccion_revision": float(rev.mean()), "exactitud_automatica": float(ok[~rev].mean()) if (~rev).any() else None,
            "errores_capturados": float((~ok & rev).sum() / (~ok).sum()) if (~ok).any() else None}


def figuras(R: dict, rutas: Rutas):
    d = rutas.final / "figuras"
    d.mkdir(parents=True, exist_ok=True)
    mods = list(R["modelos"])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.bar(range(len(mods)), [R["modelos"][k]["test_media"]["macro_f1"] for k in mods],
           yerr=[R["modelos"][k]["test_desv"]["macro_f1"] for k in mods], capsize=3, color=["#999999", "#4c78a8", "#f58518"][:len(mods)])
    ax.axhline(0.90, ls="--", lw=0.8, color="k")
    ax.set_xticks(range(len(mods)), ["Base", "ResNet-50", "EffNet-B0"][:len(mods)] if len(mods) == 3 else mods, fontsize=7)
    lo = min(R["modelos"][k]["test_media"]["macro_f1"] for k in mods)
    ax.set_ylim(max(0, lo - 0.08), 1.0)
    ax.set_ylabel("Macro-F1 (prueba)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(d / "macro_f1_modelos.png", dpi=200)
    plt.close(fig)
    if "mejor_modelo" in R:
        m = R["modelos"][R["mejor_modelo"]]
        guardar_matriz_confusion(m["matriz_confusion_primera_semilla"], config.CLASES, d / "matriz_confusion_mejor.png",
                                 f"{NOMBRES[R['mejor_modelo']]} (prueba)")
        pb, pm = _por_semilla(rutas.final / "baseline", "pred_test.csv"), _por_semilla(rutas.final / R["mejor_modelo"], "pred_test.csv")
        s0 = sorted(set(pb) & set(pm))[0]
        R["calibracion"] = {"baseline": calibracion(pb[s0]), "mejor": calibracion(pm[s0]), "semilla": s0}
        fig, axs = plt.subplots(1, 2, figsize=(3.45, 2.0))
        for k, dfp, col in (("Base", pb[s0], "#999999"), ("Aj. fino", pm[s0], "#4c78a8")):
            pr, rc, _ = precision_recall_curve((dfp.y_real == "danada").to_numpy(), dfp.p_danada.to_numpy())
            axs[0].plot(rc, pr, color=col, lw=1, label=k)
        axs[0].set_xlabel("Recall (dañada)", fontsize=7)
        axs[0].set_ylabel("Precisión (dañada)", fontsize=7)
        axs[0].legend(fontsize=6)
        ints = R["calibracion"]["mejor"]["intervalos"]
        axs[1].plot([0.5, 1], [0.5, 1], "k--", lw=0.7)
        axs[1].plot([i["confianza"] for i in ints], [i["exactitud"] for i in ints], "o-", color="#4c78a8", ms=3, lw=1)
        axs[1].set_xlabel("Confianza", fontsize=7)
        axs[1].set_ylabel("Exactitud", fontsize=7)
        for a in axs:
            a.tick_params(labelsize=6)
        fig.tight_layout()
        fig.savefig(d / "calibracion_pr.png", dpi=200)
        plt.close(fig)
        fn = [(k, R["modelos"][k].get("fruta_no_vista")) for k in ("baseline", R["mejor_modelo"])]
        if all(v for _, v in fn):
            fig, ax = plt.subplots(figsize=(3.4, 2.4))
            w = 0.38
            for i, (k, v) in enumerate(fn):
                ax.bar(np.arange(len(config.FRUTAS)) + i * w, [v["por_fruta"].get(f, {"macro_f1": 0})["macro_f1"] for f in config.FRUTAS], w,
                       label="Base" if k == "baseline" else "Ajuste fino")
            ax.axhline(0.70, ls="--", lw=0.8, color="k")
            ax.set_xticks(np.arange(len(config.FRUTAS)) + w / 2, config.FRUTAS, fontsize=7, rotation=20)
            ax.set_ylabel("Macro-F1 en fruta no vista", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(d / "fruta_no_vista.png", dpi=200)
            plt.close(fig)


def errores(R: dict, rutas: Rutas, docs: Path, n_por_tipo: int = 8):
    if "mejor_modelo" not in R:
        return None
    mejor = R["mejor_modelo"]
    p = _por_semilla(rutas.final / mejor, "pred_test.csv")
    d = p[sorted(p)[0]]
    e = d[d.y_real != d.y_pred].copy()
    e["confianza"] = np.where(e.y_pred == "danada", e.p_danada, 1 - e.p_danada)
    e["tipo_error"] = np.where(e.y_real == "fresca", "falso positivo (fresca -> danada)", "falso negativo (danada -> fresca)")
    e = e.sort_values("confianza", ascending=False)
    e.to_csv(rutas.final / f"errores_test_{mejor}.csv", index=False)
    sel = pd.concat([g.head(n_por_tipo) for _, g in e.groupby("tipo_error")])
    plantilla = docs / "analisis_errores.csv"
    if not plantilla.exists():
        docs.mkdir(parents=True, exist_ok=True)
        t = sel[["id", "fruta", "y_real", "y_pred", "confianza", "tipo_error"]].copy()
        t["confianza"] = t["confianza"].round(3)
        t["categoria_causa"] = ""      # etiqueta | ambiguo | encuadre | iluminacion | otra
        t["causa_hipotesis"] = ""
        t.to_csv(plantilla, index=False)
    montaje(sel, rutas, rutas.final / "figuras" / "errores_montaje.png")
    return {"n_errores_test": int(len(e)), "por_tipo": e["tipo_error"].value_counts().to_dict(),
            "confianza_media_por_tipo": {k: float(v) for k, v in e.groupby("tipo_error")["confianza"].mean().items()},
            "n_confianza_alta": int((e["confianza"] >= 0.9).sum()),
            "por_fruta": e["fruta"].value_counts().to_dict(), "plantilla": str(plantilla)}


def montaje(sel: pd.DataFrame, rutas: Rutas, salida: Path, cols: int = 4):
    from PIL import Image
    filas = [(r, rutas.raw / r["id"]) for _, r in sel.iterrows() if (rutas.raw / r["id"]).exists()]
    if not filas:
        return
    n = len(filas)
    rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.9))
    axs = np.atleast_2d(axs)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, (r, ruta) in zip(axs.ravel(), filas):
        with Image.open(ruta) as im:
            im.draft("RGB", (400, 400))
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"{r['fruta']}: real {r['y_real']}\npred {r['y_pred']} ({r['confianza']:.2f})", fontsize=7)
    fig.tight_layout()
    salida.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(salida, dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", type=Path, default=config.ROOT)
    a = ap.parse_args(argv)
    rutas = Rutas(a.raiz)
    R = resumir(rutas)
    if not R["modelos"]:
        raise SystemExit("No hay corridas finales en experiments/final/. Ejecute baseline_final, tune y run_final.")
    figuras(R, rutas)
    R["errores"] = errores(R, rutas, a.raiz / "docs")
    (rutas.final / "resumen.json").write_text(json.dumps(R, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    for k, v in R["modelos"].items():
        t, s = v["test_media"], v["test_desv"]
        print(f"{NOMBRES[k]:<36} macro-F1 {t['macro_f1']:.3f} +/- {s['macro_f1']:.3f} | exactitud {t['exactitud']:.3f} | recall danada {t['recall_danada']:.3f}")
    for u in R.get("umbrales", []):
        print(f"  {u['metrica']:<36} {'n/d' if u['valor'] is None else format(u['valor'], '.3f'):>6} (umbral {u['umbral']})  {u['cumple']}")
    print(f"-> {rutas.final / 'resumen.json'}")


if __name__ == "__main__":
    main()
