"""Genera el informe tecnico (articulo a doble columna) en LaTeX y lo compila con pdflatex.

Lee: experiments/final/resumen.json (python -m src.evaluar_final), experiments/baseline/ (Hito 2), data/manifest.csv,
data/splits.csv y docs/analisis_errores.csv. Si faltan los resultados finales, deja marcas [pendiente] (modo borrador).
Requiere pdflatex (TeX Live / MiKTeX).   Uso:  python docs/informe/build_informe.py [--raiz .]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PEND = r"\textbf{[pendiente]}"
FRUTAS = ["manzana", "banano", "mango", "naranja", "uva"]
NOM = {"baseline": "Línea base", "resnet50": "ResNet-50 (aj. fino)", "efficientnet_b0": "EfficientNet-B0 (aj. fino)"}


def esc(s) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s


def n(x, d=3):
    if x is None or (isinstance(x, float) and x != x):
        return PEND
    return f"{x:,.{d}f}".replace(",", "\\,").replace(".", ",") if d == 0 else f"{x:.{d}f}".replace(".", ",")


def pm(m, s, d=3):
    return PEND if m is None else f"{n(m, d)}\\,$\\pm$\\,{n(s, d)}"


def pct(x, d=0):
    return PEND if x is None else f"{x * 100:.{d}f}".replace(".", ",") + r"\,\%"


def miles(x):
    return f"{int(x):,}".replace(",", r"\,")


def jl(ruta: Path):
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else None


def diagrama(salida: Path):
    fig, ax = plt.subplots(figsize=(3.45, 1.9))
    ax.axis("off")
    cajas = [(0.02, "Foto\n(RGB)"), (0.23, "Redimensionar\n256 y recorte\n224"), (0.5, "CNN preentrenada\nResNet-50 /\nEfficientNet-B0"), (0.79, "Cabeza\n2 clases")]
    for x, t in cajas:
        ax.add_patch(plt.Rectangle((x, 0.45), 0.18 if x < 0.5 else 0.25, 0.42, fc="#eef3fa", ec="k", lw=0.8))
        ax.text(x + (0.09 if x < 0.5 else 0.125), 0.66, t, ha="center", va="center", fontsize=6)
    for x0, x1 in ((0.2, 0.23), (0.41, 0.5), (0.75, 0.79)):
        ax.annotate("", xy=(x1, 0.66), xytext=(x0, 0.66), arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.text(0.5, 0.25, "Línea base: CNN congelada + cabeza lineal (SGD)\nAjuste fino: AdamW, coseno, aumentos solo en entrenamiento\nSalida: fresca / dañada con probabilidad (umbral 0,5)", ha="center", va="center", fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.savefig(salida, dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig(rel: Path, base: Path, ancho: str, pie: str, etiqueta: str, doble=False) -> str:
    env = "figure*" if doble else "figure"
    if (base / rel).exists():
        cuerpo = rf"\includegraphics[width={ancho}]{{{rel.as_posix()}}}"
    else:
        cuerpo = rf"\fbox{{\parbox{{{ancho}}}{{\centering {PEND}: figura no generada}}}}"
    return rf"\begin{{{env}}}[t]\centering{cuerpo}\caption{{{pie}}}\label{{{etiqueta}}}\end{{{env}}}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raiz", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--salida", type=Path, default=None)
    a = ap.parse_args()
    raiz = a.raiz
    out = a.salida or raiz / "docs" / "informe"
    out.mkdir(parents=True, exist_ok=True)
    (out / "figs").mkdir(exist_ok=True)

    R = jl(raiz / "experiments/final/resumen.json") or {}
    H2 = jl(raiz / "experiments/baseline/resultados_val.json")
    EJ = jl(raiz / "experiments/final/resumen_ejecucion.json") or {}
    TUN = jl(raiz / "experiments/final/tuning/mejor_config.json") or {}
    final = bool(R.get("modelos")) and "mejor_modelo" in R
    for f in (raiz / "experiments/final/figuras").glob("*.png"):
        if f.name != "errores_montaje.png":       # contiene imagenes del dataset (CC BY-NC-ND): no se redistribuye
            shutil.copy(f, out / "figs" / f.name)
    diagrama(out / "figs" / "arquitectura.png")

    # ---- datos ----
    m = pd.read_csv(raiz / "data/manifest.csv")
    sp = pd.read_csv(raiz / "data/splits.csv")
    part = sp["particion"].value_counts()
    cnt = m.pivot_table(index="clase", columns="fruta", values="id", aggfunc="count")
    tot = m["clase"].value_counts()
    g = m.groupby("grupo")
    n_grupos, n_mix = int((g.size() > 1).sum()), int((g["clase"].nunique() > 1).sum())
    ancho_min, ancho_max, n_res = int(m["ancho"].min()), int(m["ancho"].max()), int((m["ancho"].astype(str) + "x" + m["alto"].astype(str)).nunique())
    tabla_datos = "\n".join([r"\begin{tabular}{@{}lcccccr@{}}\toprule", " & " + " & ".join(FRUTAS) + r" & Total \\ \midrule"] +
        [f"{nom} & " + " & ".join(miles(cnt.loc[c, f]) for f in FRUTAS) + f" & {miles(tot[c])} \\\\" for c, nom in (("fresca", "fresca"), ("danada", "dañada"))] +
        [r"\bottomrule\end{tabular}"])

    M = R.get("modelos", {})
    base, mejor_k = M.get("baseline"), R.get("mejor_modelo")
    mejor = M.get(mejor_k) if mejor_k else None
    cmp_ = R.get("comparacion", {})

    # ---- tablas de resultados ----
    filas = []
    for k in ("baseline", "resnet50", "efficientnet_b0"):
        v = M.get(k)
        filas.append(f"{NOM[k]} & " + (" & ".join([pm(v["test_media"]["macro_f1"], v["test_desv"]["macro_f1"]), pm(v["test_media"]["exactitud"], v["test_desv"]["exactitud"]),
                                                    pm(v["test_media"]["recall_danada"], v["test_desv"]["recall_danada"])]) if v else " & ".join([PEND] * 3)) + r" \\")
    tabla_res = r"\begin{tabular}{@{}lccc@{}}\toprule Modelo & Macro-F1 & Exactitud & Recall dañada \\ \midrule" + "\n" + "\n".join(filas) + "\n" + r"\bottomrule\end{tabular}"
    umb_filas = []
    nombres_u = {"macro_f1_prueba": "Macro-F1 en prueba", "exactitud_prueba": "Exactitud en prueba", "macro_f1_fruta_no_vista": "Macro-F1, fruta no vista",
                 "recall_danada_prueba": "Recall de «dañada»", "reduccion_error_sobre_linea_base": "Reducción del error vs. base"}
    for u in R.get("umbrales", []):
        val = pct(u["valor"], 1) if u["metrica"].startswith("reduccion") else n(u["valor"])
        um = pct(u["umbral"]) if u["metrica"].startswith("reduccion") else n(u["umbral"], 2)
        umb_filas.append(f"{nombres_u[u['metrica']]} & {val} & $\\geq$ {um} & {'Sí' if u['cumple'] else 'No' if u['cumple'] is False else PEND} \\\\")
    tabla_umb = (r"\begin{tabular}{@{}lccc@{}}\toprule Métrica & Valor & Umbral & ¿Cumple? \\ \midrule" + "\n" + "\n".join(umb_filas) + "\n" + r"\bottomrule\end{tabular}") if umb_filas else PEND
    pf_filas = []
    for fr in FRUTAS:
        b = (base or {}).get("por_fruta", {}).get(fr)
        mm = (mejor or {}).get("por_fruta", {}).get(fr)
        fn_b = ((base or {}).get("fruta_no_vista") or {}).get("por_fruta", {}).get(fr)
        fn_m = ((mejor or {}).get("fruta_no_vista") or {}).get("por_fruta", {}).get(fr)
        pf_filas.append(f"{fr} & {n(b['macro_f1']) if b else PEND} & {n(mm['macro_f1']) if mm else PEND} & {n(fn_b['macro_f1']) if fn_b else PEND} & {n(fn_m['macro_f1']) if fn_m else PEND} \\\\")
    tabla_fruta = r"\begin{tabular}{@{}lcccc@{}}\toprule & \multicolumn{2}{c}{Prueba} & \multicolumn{2}{c}{Fruta no vista} \\ \cmidrule(lr){2-3}\cmidrule(lr){4-5} Fruta & Base & Aj. fino & Base & Aj. fino \\ \midrule" + "\n" + "\n".join(pf_filas) + "\n" + r"\bottomrule\end{tabular}"

    # ---- errores ----
    ea = raiz / "docs/analisis_errores.csv"
    err_filas, n_hip = [], 0
    if ea.exists():
        E = pd.read_csv(ea).fillna("")
        for _, r in E.iterrows():
            hip = str(r["causa_hipotesis"]).strip()
            n_hip += bool(hip)
            err_filas.append(f"{esc(r['fruta'])} & {'FP' if str(r['tipo_error']).startswith('falso positivo') else 'FN'} & {n(float(r['confianza']), 2)} & {esc(hip) if hip else PEND} \\\\")
    tabla_err = (r"\begin{tabular}{@{}llcp{4.6cm}@{}}\toprule Fruta & Tipo & Conf. & Hipótesis de causa \\ \midrule" + "\n" + "\n".join(err_filas) + "\n" + r"\bottomrule\end{tabular}") if err_filas else PEND
    e_info = R.get("errores") or {}
    nerr_txt = (f"{e_info['n_errores_test']} errores en prueba para {NOM[mejor_k]} (semilla {mejor['semillas'][0]}); "
                f"{e_info['por_tipo'].get('falso positivo (fresca -> danada)', 0)} falsos positivos (fresca clasificada como dañada) y "
                f"{e_info['por_tipo'].get('falso negativo (danada -> fresca)', 0)} falsos negativos") if e_info else PEND

    # ---- textos dependientes de los resultados ----
    h2 = H2["linea_base"] if H2 else None
    if final:
        red, mc = cmp_["reduccion_error"], R.get("mcnemar", {})
        cumple_red = any(u["metrica"].startswith("reduccion") and u["cumple"] for u in R["umbrales"])
        fnv = (mejor.get("fruta_no_vista") or {})
        pf = fnv.get("por_fruta", {})
        peor = min(pf, key=lambda k: pf[k]["macro_f1"]) if pf else None
        mejor_f = max(pf, key=lambda k: pf[k]["macro_f1"]) if pf else None
        dif_f = [mejor["por_fruta"][f]["macro_f1"] for f in mejor["por_fruta"]]
        resumen = (f"Se estudia la clasificación del estado de una fruta (fresca o dañada) a partir de una foto, con {miles(len(m))} imágenes originales de cinco frutas "
                   f"del conjunto FruitVision. Se compara una línea base (ResNet-50 preentrenada y congelada con cabeza lineal) contra el ajuste fino de ResNet-50 y EfficientNet-B0, "
                   f"con partición sin fuga entre duplicados, tres semillas y una única evaluación en prueba. El mejor modelo ({NOM[mejor_k]}) obtuvo un macro-F1 de "
                   f"{pm(mejor['test_media']['macro_f1'], mejor['test_desv']['macro_f1'])} frente a {pm(base['test_media']['macro_f1'], base['test_desv']['macro_f1'])} de la línea base "
                   f"(reducción relativa del error de {pct(red, 0)}). Al evaluar en una fruta que el modelo no vio durante el entrenamiento, el macro-F1 medio fue "
                   f"{n(fnv.get('macro_f1_media'))}, con grandes diferencias entre frutas. Se analizan los errores, los riesgos de sesgo y las limitaciones de un conjunto de imágenes de laboratorio.")
        concl = (rf"\textbf{{Pregunta 1 (ajuste fino vs. extractor congelado).}} El ajuste fino de {NOM[mejor_k]} alcanzó un macro-F1 de {pm(mejor['test_media']['macro_f1'], mejor['test_desv']['macro_f1'])} "
                 rf"frente a {pm(base['test_media']['macro_f1'], base['test_desv']['macro_f1'])} de la línea base: una reducción relativa del error de {pct(red, 0)}, "
                 rf"{'que cumple' if cumple_red else 'que no cumple'} el criterio declarado (30\,\% y mayor que la desviación entre semillas, que fue {n(cmp_['desv_max'])}). "
                 rf"La prueba de McNemar (semilla {mc.get('semilla')}) da $p={n(mc.get('p_valor'), 3)}$ con {mc.get('base_acierta_modelo_falla')} casos en que solo la línea base acierta y {mc.get('base_falla_modelo_acierta')} en que solo acierta el modelo ajustado. "
                 rf"\textbf{{Pregunta 2 (fruta no vista).}} El macro-F1 medio al excluir una fruta del entrenamiento fue {n(fnv.get('macro_f1_media'))} "
                 rf"(umbral 0,70); la fruta más difícil fue {esc(peor)} ({n(pf[peor]['macro_f1'])}) y la más fácil {esc(mejor_f)} ({n(pf[mejor_f]['macro_f1'])}). "
                 rf"Por tanto, las conclusiones se limitan a estas cinco frutas fotografiadas en condiciones de laboratorio: el rendimiento no se extrapola a otras frutas ni a una línea de empaque real.")
        disp = (f"La diferencia entre la mejor y la peor fruta en prueba es de {n(max(dif_f) - min(dif_f))} puntos de macro-F1 y, ante una fruta no vista, el macro-F1 baja a {n(pf[peor]['macro_f1'])} en {esc(peor)}.")
        ent = EJ.get("entorno", {})
        entorno = (f"Python {esc(ent.get('python', '?'))}, PyTorch {esc(ent.get('torch', '?'))}, torchvision {esc(ent.get('torchvision', '?'))}, "
                   f"{'GPU ' + esc(ent.get('gpu')) if ent.get('gpu') else 'CPU'}")
        hp = []
        for k in ("resnet50", "efficientnet_b0"):
            if k in TUN:
                hp.append(f"{NOM[k]}: tasa de aprendizaje {TUN[k]['lr']:g}, capas {'todas' if TUN[k]['descongelar'] == 'todo' else 'solo el último bloque'}, {TUN[k]['epocas']} épocas máximas")
        hp_txt = "; ".join(hp) + "."
        alfa = (base or {}).get("config") or {}
    else:
        resumen = (f"Se estudia la clasificación del estado de una fruta (fresca o dañada) a partir de una foto, con {miles(len(m))} imágenes originales de cinco frutas del conjunto FruitVision. "
                   f"Se compara una línea base (ResNet-50 preentrenada y congelada con cabeza lineal) contra el ajuste fino de ResNet-50 y EfficientNet-B0. "
                   f"En validación, la línea base obtuvo un macro-F1 de {pm(h2['macro_f1_media'], h2['macro_f1_desv']) if h2 else PEND}. {PEND}: resultados finales en prueba y en fruta no vista.")
        concl = PEND + " Las conclusiones se redactan cuando existan los resultados finales (experiments/final/resumen.json)."
        disp = PEND
        entorno = PEND
        hp_txt = PEND

    gen = (H2 or {}).get("generalizacion_fruta_no_vista", {})
    h2_txt = (f"En validación (Hito 2), la línea base obtuvo un macro-F1 de {pm(h2['macro_f1_media'], h2['macro_f1_desv'])} y, con una fruta no vista, {n(gen.get('macro_f1_media_5_frutas'))} en promedio" if h2 else "")


    # ---- tablas y textos adicionales ----
    pc = sp.merge(m[["id", "clase"]], on="id").pivot_table(index="particion", columns="clase", values="id", aggfunc="count")
    tabla_part = "\n".join([r"\begin{tabular}{@{}lccc@{}}\toprule Partición & Fresca & Dañada & Total \\ \midrule"] +
        [f"{nom} & {miles(pc.loc[k, 'fresca'])} & {miles(pc.loc[k, 'danada'])} & {miles(pc.loc[k].sum())} \\\\" for k, nom in (("train", "Entrenamiento"), ("val", "Validación"), ("test", "Prueba"))] +
        [r"\bottomrule\end{tabular}"])
    por_fruta_tot = m["fruta"].value_counts()
    txt_balance = (f"Las clases están casi equilibradas ({pct(tot['fresca'] / len(m), 1)} frescas) y todas las frutas aportan entre {miles(por_fruta_tot.min())} y {miles(por_fruta_tot.max())} imágenes (Tabla~\\ref{{tab:part}}).")

    gr = raiz / "experiments/final/tuning/grid_val.csv"
    if gr.exists():
        G = pd.read_csv(gr)
        mx = G.groupby("arq")["macro_f1_val"].transform("max")
        fl = []
        for (_, r), best in zip(G.iterrows(), (G["macro_f1_val"] == mx)):
            nb = (lambda t: rf"\textbf{{{t}}}" if best else t)
            fl.append(f"{nb({'resnet50': 'ResNet-50', 'efficientnet_b0': 'EffNet-B0'}[r['arq']])} & {nb(f'{r.lr:g}'.replace('.', ','))} & {nb('todas' if r['descongelar'] == 'todo' else 'último bloque')} & {nb(n(r['macro_f1_val']))} & {nb(str(int(r['mejor_epoca'])))} \\\\")
        tabla_tuning = r"\begin{tabular}{@{}llccc@{}}\toprule Modelo & Tasa & Capas & F1 val. & Época \\ \midrule" + "\n" + "\n".join(fl) + "\n" + r"\bottomrule\end{tabular}"
    else:
        tabla_tuning = PEND

    def curvas(salida: Path):
        fig_, ax_ = plt.subplots(figsize=(3.45, 2.2))
        hay = False
        for k, col in (("resnet50", "#4c78a8"), ("efficientnet_b0", "#f58518")):
            for f in sorted((raiz / "experiments/final" / k).glob("s*/resumen.json")):
                h = json.loads(f.read_text(encoding="utf-8")).get("historial", [])
                if h:
                    hay = True
                    ax_.plot([x["epoca"] for x in h], [x["macro_f1_val"] for x in h], color=col, alpha=0.8, lw=1,
                             label=NOM[k] if f.parent.name == "s42" else None)
        if hay:
            ax_.set_xlabel("Época", fontsize=8)
            ax_.set_ylabel("Macro-F1 (validación)", fontsize=8)
            ax_.tick_params(labelsize=7)
            ax_.legend(fontsize=6)
            fig_.tight_layout()
            fig_.savefig(salida, dpi=200)
        plt.close(fig_)
    curvas(out / "figs" / "curvas.png")

    ref_nom = lambda k: NOM[k]
    if final:
        tipos = (R.get("errores") or {}).get("por_tipo", {})
        fp, fn_ = tipos.get("falso positivo (fresca -> danada)", 0), tipos.get("falso negativo (danada -> fresca)", 0)
        bs = R.get("bootstrap")
        n_cumple = sum(1 for u in R["umbrales"] if u["cumple"])
        no_cumple = [nombres_u[u["metrica"]] for u in R["umbrales"] if u["cumple"] is False]
        ic_txt = ""
        if bs:
            lo, hi = bs["delta"]
            veredicto = ("indica una mejora que no se explica solo por el muestreo de la prueba" if lo > 0 else
                         "indica que el ajuste fino empeora respecto a la línea base" if hi < 0 else
                         "incluye el cero: con esta prueba no se puede afirmar que haya una diferencia")
            ic_txt = (f"El intervalo de confianza del 95\\,\\% (bootstrap con {bs['B']} remuestreos de las imágenes de prueba, semilla {bs['semilla']}) para la diferencia de macro-F1 entre "
                      f"{NOM[mejor_k]} y la línea base es [{n(lo)}; {n(hi)}]; {veredicto}. ")
        seg_m, seg_b = mejor.get("segundos_s42"), None
        otro = [k for k in ("resnet50", "efficientnet_b0") if k in M and k != mejor_k]
        t_otro = M[otro[0]].get("segundos_s42") if otro else None
        tiempos = (f"Una corrida de {NOM[mejor_k]} (semilla 42) tardó {n(seg_m, 0)}\\,s" + (f" y una de {NOM[otro[0]]}, {n(t_otro, 0)}\\,s" if t_otro else "") +
                   " en el entorno indicado; ResNet-50 tiene unos 25,6\\,M de parámetros y EfficientNet-B0 unos 5,3\\,M. ") if seg_m else ""
        disc = (f"{ic_txt}{tiempos}De los cinco criterios declarados se cumplen {n_cumple}" + (f"; no se cumplen: {', '.join(no_cumple)}. " if no_cumple else ". ") +
                "Los resultados corresponden a una única partición y tres semillas: las cifras por fruta (unas 200 imágenes de prueba cada una) son ruidosas.")
        r_fnv = n(((mejor.get("fruta_no_vista") or {}).get("macro_f1_media")))
        riesgos_vals = {"FNV": r_fnv, "TEST": n(mejor["test_media"]["macro_f1"]), "REC": n(mejor["test_media"]["recall_danada"]), "FN": str(fn_), "FP": str(fp)}
    else:
        disc, riesgos_vals = PEND, {k: PEND for k in ("FNV", "TEST", "REC", "FN", "FP")}
    rv = riesgos_vals
    tabla_riesgos = (r"\begin{tabular}{@{}p{2.9cm}p{2.7cm}p{5.6cm}p{5.6cm}@{}}\toprule Riesgo & Afectados & Evidencia & Mitigación \\ \midrule" + "\n" +
        rf"Sesgo de dominio (otras frutas, fondos o cámaras) & Productores con frutas o condiciones distintas & Macro-F1 en fruta no vista: {rv['FNV']} frente a {rv['TEST']} en prueba; grandes diferencias entre frutas & Evaluar por fruta; recolectar datos locales y repetir la evaluación; no extrapolar \\" + "\n" +
        rf"Falsos negativos (dañada aprobada) & Consumidores & Recall de ``dañada'' {rv['REC']}; {rv['FN']} FN en prueba (mejor modelo, semilla 42) & Priorizar el recall; revisión humana si la confianza es menor a 0,80; muestreo manual \\" + "\n" +
        rf"Falsos positivos (fruta buena descartada) & Productores y comerciantes & {rv['FP']} FP en prueba (mejor modelo, semilla 42) & Segunda revisión de descartes; umbral de decisión ajustable \\" + "\n" +
        r"Privacidad & Personas fotografiadas sin querer & Datos sin personas por diseño; las fotos de celular pueden traer metadatos & No se redistribuyen imágenes; el prototipo no guarda las fotos que recibe \\" + "\n" +
        r"Uso indebido & Consumidores y titulares de la licencia & No detecta formalina; licencia CC BY-NC-ND (sin uso comercial) & Aviso visible en el prototipo; uso académico; documentar el uso previsto \\" + "\n" +
        rf"Resultados inflados por fuga & Quien confíe en las cifras & {n_mix} de {n_grupos} grupos de duplicados mezclan clases; no hay identificador del fruto físico & Reportar la limitación; partición por grupos; verificar con datos externos \\" + "\n" + r"\bottomrule\end{tabular}")
    fc = []
    for k in ("baseline", "resnet50", "efficientnet_b0"):
        v = M.get(k)
        for c, nom in (("fresca", "fresca"), ("danada", "dañada")):
            pcl = v["por_clase"][c] if v and "por_clase" in v else None
            fc.append(f"{NOM[k] if c == 'fresca' else ''} & {nom} & " + (f"{n(pcl['precision'])} & {n(pcl['recall'])} & {n(pcl['f1'])}" if pcl else " & ".join([PEND] * 3)) + r" \\")
    tabla_clase = r"\begin{tabular}{@{}llccc@{}}\toprule Modelo & Clase & Precisión & Recall & F1 \\ \midrule" + "\n" + "\n".join(fc) + "\n" + r"\bottomrule\end{tabular}"
    ei = R.get("errores") or {}
    cm_t = ei.get("confianza_media_por_tipo", {})
    pe = ei.get("por_fruta", {})
    txt_err = (f"La confianza media de los errores es {n(cm_t.get('falso positivo (fresca -> danada)'), 2)} en los FP y {n(cm_t.get('falso negativo (danada -> fresca)'), 2)} en los FN, y {ei.get('n_confianza_alta')} de {ei.get('n_errores_test')} errores tienen confianza $\\geq$ 0,90; por fruta, " +
               ", ".join(f"{esc(k)} ({v})" for k, v in pe.items()) + ".") if ei else PEND
    if ea.exists() and "categoria_causa" in pd.read_csv(ea).columns:
        cats = pd.read_csv(ea).fillna("")["categoria_causa"].astype(str).str.strip()
        cats = cats[cats != ""].value_counts()
        txt_cat = ("Categorías asignadas: " + ", ".join(f"{esc(k)} ({v})" for k, v in cats.items()) + ".") if len(cats) else PEND
    else:
        txt_cat = PEND
    cal = R.get("calibracion")
    if cal and final:
        cm_, cb_ = cal["mejor"], cal["baseline"]
        txt_cal = (f"Una confianza alta no garantiza un acierto, por lo que el prototipo marca para revisión humana toda predicción con confianza menor a {n(cm_['umbral_revision'], 2)}. "
                   f"Sobre la prueba (semilla {cal['semilla']}), el error de calibración esperado (ECE, 5 intervalos) de {NOM[mejor_k]} es {n(cm_['ece'])} (línea base: {n(cb_['ece'])}). "
                   f"La regla envía a revisión el {pct(cm_['fraccion_revision'], 1)} de las imágenes; el resto se decide automáticamente con una exactitud de {n(cm_['exactitud_automatica'])}, y la regla captura el {pct(cm_['errores_capturados'], 0)} de los errores (Fig.~\ref{{fig:cal}}). "
                   f"La curva precisión-recall de la clase ``dañada'' muestra el compromiso al mover el umbral de decisión; se presenta como descripción y no para elegir el umbral con la prueba.")
    else:
        txt_cal = PEND
    tabla_peas = (r"\begin{tabular}{@{}p{1.5cm}p{5.6cm}@{}}\toprule \textit{Medida} & Macro-F1 y recall de ``dañada'' (no dejar pasar fruta dañada) \\ \textit{Entorno} & Imagen de una fruta individual; parcialmente observable, estocástico, episódico, estático, un agente \\ \textit{Actuadores} & Etiqueta de estado y decisión continuar o desviar (simulada) \\ \textit{Sensores} & Imagen RGB \\ \bottomrule\end{tabular}")

    refs = r"""
\bibitem{he} K. He, X. Zhang, S. Ren y J. Sun, ``Deep residual learning for image recognition,'' en \textit{Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)}, 2016.
\bibitem{tan} M. Tan y Q. V. Le, ``EfficientNet: Rethinking model scaling for convolutional neural networks,'' en \textit{Proc. Int. Conf. Machine Learning (ICML)}, 2019.
\bibitem{yos} J. Yosinski, J. Clune, Y. Bengio y H. Lipson, ``How transferable are features in deep neural networks?,'' en \textit{Advances in Neural Information Processing Systems (NeurIPS)}, 2014.
\bibitem{pan} S. J. Pan y Q. Yang, ``A survey on transfer learning,'' \textit{IEEE Trans. Knowledge and Data Engineering}, 2010.
\bibitem{ism} N. Ismail y O. A. Malik, ``Real-time visual inspection system for grading fruits using computer vision and deep learning techniques,'' \textit{Information Processing in Agriculture}, 2022.
\bibitem{fv} M. H. I. Bijoy, S. Z. Tasnim, S. A. Awsaf y M. Z. Hasan, ``FruitVision: A benchmark dataset for fresh, rotten, and formalin-mixed fruit detection,'' \textit{Data in Brief}, 2025; Mendeley Data, V2, doi:10.17632/xkbjx8959c.2.
\bibitem{mit} M. Mitchell \textit{et al.}, ``Model cards for model reporting,'' en \textit{Proc. ACM Conf. Fairness, Accountability, and Transparency (FAT*)}, 2019.
\bibitem{loc} I. Loshchilov y F. Hutter, ``Decoupled weight decay regularization,'' en \textit{Int. Conf. Learning Representations (ICLR)}, 2019.
\bibitem{pt} A. Paszke \textit{et al.}, ``PyTorch: An imperative style, high-performance deep learning library,'' en \textit{Advances in Neural Information Processing Systems (NeurIPS)}, 2019.
\bibitem{rus} O. Russakovsky \textit{et al.}, ``ImageNet large scale visual recognition challenge,'' \textit{Int. J. Computer Vision}, 2015.
\bibitem{kau} S. Kaufman, S. Rosset, C. Perlich y O. Stitelman, ``Leakage in data mining: Formulation, detection, and avoidance,'' \textit{ACM Trans. Knowledge Discovery from Data}, 2012.
\bibitem{sok} M. Sokolova y G. Lapalme, ``A systematic analysis of performance measures for classification tasks,'' \textit{Information Processing \& Management}, 2009.
\bibitem{die} T. G. Dietterich, ``Approximate statistical tests for comparing supervised classification learning algorithms,'' \textit{Neural Computation}, 1998.
\bibitem{efr} B. Efron y R. J. Tibshirani, \textit{An Introduction to the Bootstrap}. Chapman \& Hall, 1993.
\bibitem{geb} T. Gebru \textit{et al.}, ``Datasheets for datasets,'' \textit{Communications of the ACM}, 2021.
\bibitem{aima} S. Russell y P. Norvig, \textit{Artificial Intelligence: A Modern Approach}, Global Edition. Pearson.
"""
    reflexion = (
        r"\textbf{Caso 1: licencia.} La IA anotó la licencia de FruitVision como MIT, porque así figura en el repositorio de uno de los autores. Al verificar la página oficial del dataset se vio que era CC BY-NC-ND 4.0, más restrictiva; se corrigió la documentación y se aplicó la licencia más estricta. "
        r"\textbf{Caso 2: resolución.} La IA documentó imágenes de 512$\times$512 según el README de los autores; el manifiesto mostró " + f"{n_res}" + r" resoluciones distintas (hasta " + f"{ancho_max}" + r" px de ancho), y se corrigió. "
        r"\textbf{Caso 3: duplicados.} La IA propuso un control de duplicados con dHash; al aplicarlo, " + f"{n_mix} de {n_grupos}" + r" grupos mezclaban clases (falsos positivos del hash con fondos lisos), lo que se reporta como limitación. "
        r"\textbf{Caso 4: cuadrícula.} La IA propuso una cuadrícula de regularización cuyo mejor valor quedó en el extremo; se detectó al revisar los resultados del Hito 2 y se amplió en la línea base final. "
        r"En todos los casos la detección vino de contrastar las afirmaciones de la IA con la fuente primaria (página del dataset, imágenes reales, resultados).")

    tpl = (Path(__file__).parent / "plantilla.tex").read_text(encoding="utf-8")
    valores = {
        "BORRADOR": "" if final else r"\par\smallskip{\small\textbf{BORRADOR: faltan los resultados finales (marcas [pendiente])}}",
        "RESUMEN": resumen,
        "N_TOTAL": miles(len(m)), "N_FRESCA": miles(tot["fresca"]), "N_DANADA": miles(tot["danada"]),
        "TABLA_DATOS": tabla_datos, "N_TRAIN": miles(part["train"]), "N_VAL": miles(part["val"]), "N_TEST": miles(part["test"]),
        "N_GRUPOS": str(n_grupos), "N_MIX": str(n_mix), "N_RES": str(n_res), "ANCHO_MIN": str(ancho_min), "ANCHO_MAX": str(ancho_max),
        "H2_TXT": h2_txt, "ENTORNO": entorno, "HP_TXT": hp_txt,
        "TABLA_RES": tabla_res, "TABLA_UMB": tabla_umb, "TABLA_FRUTA": tabla_fruta, "TABLA_ERR": tabla_err, "NERR_TXT": nerr_txt,
        "N_HIP": str(n_hip), "N_FILAS_ERR": str(len(err_filas)),
        "FIG_ARQ": fig(Path("figs/arquitectura.png"), out, r"\columnwidth", "Flujo del método.", "fig:arq"),
        "FIG_BARRAS": fig(Path("figs/macro_f1_modelos.png"), out, r"\columnwidth", "Macro-F1 en prueba (media y desviación entre semillas); línea discontinua: umbral 0,90.", "fig:barras"),
        "FIG_FNV": fig(Path("figs/fruta_no_vista.png"), out, r"\columnwidth", "Macro-F1 al excluir una fruta del entrenamiento; línea discontinua: umbral 0,70.", "fig:fnv"),
        "FIG_CM": fig(Path("figs/matriz_confusion_mejor.png"), out, "0.8\\columnwidth", "Matriz de confusión del mejor modelo en prueba.", "fig:cm"),
        "TXT_CAL": txt_cal, "FIG_CAL": fig(Path("figs/calibracion_pr.png"), out, r"\columnwidth", "Izquierda: curva precisión-recall de ``dañada''. Derecha: confianza frente a exactitud (mejor modelo, prueba).", "fig:cal"),
        "TABLA_CLASE": tabla_clase, "TXT_ERR": txt_err, "TXT_CAT": txt_cat, "TABLA_PART": tabla_part, "TXT_BALANCE": txt_balance, "TABLA_TUNING": tabla_tuning, "TABLA_RIESGOS": tabla_riesgos, "TABLA_PEAS": tabla_peas, "DISC": disc,
        "FIG_CURVAS": fig(Path("figs/curvas.png"), out, r"\columnwidth", "Macro-F1 en validación por época (una línea por semilla).", "fig:curvas"),
        "DISP": disp, "CONCL": concl, "REFLEXION": reflexion, "REFS": refs,
    }
    tex = tpl
    for k, v in valores.items():
        tex = tex.replace(f"<<{k}>>", v)
    (out / "informe.tex").write_text(tex, encoding="utf-8")
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "informe.tex"], cwd=out, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(r.stdout[-2500:])
            raise SystemExit("pdflatex falló; revise docs/informe/informe.log")
    paginas = subprocess.run("pdfinfo informe.pdf | awk '/Pages/{print $2}'", shell=True, cwd=out, capture_output=True, text=True).stdout.strip()
    print(f"-> {out / 'informe.pdf'} ({paginas} páginas; modo {'FINAL' if final else 'BORRADOR'})")


if __name__ == "__main__":
    main()
