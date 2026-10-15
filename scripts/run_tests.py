"""
Ejecuta las pruebas de la Fase 3 contra las medidas de desempeño M1-M6.

    python scripts/run_tests.py            # pruebas completas (aprox. 10 min en un solo núcleo)
    python scripts/run_tests.py --rapido   # humo: 1 repetición y 40 unidades por corrida

Salidas en logs/pruebas/:
    log_clasificacion.csv, log_fallas.csv        registro por unidad (percepción, decisión, acción, verdad)
    resultados_por_corrida.csv                   métricas por escenario y repetición
    reportes_proveedor.csv                       reportes por lote y verificación de consistencia
    resultados_resumen.json                      media, desviación estándar y cumplimiento por medida
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from freshcheck.evaluation import (ejecutar_escenario, m1_exactitud, m2_reduccion_defectuosas,  # noqa: E402
                                   m3_descarte_indebido, m4_latencia, tasa_defecto_empaque)
from freshcheck.model import ClasificadorEstado                                                  # noqa: E402
from freshcheck.reports import consistencia, reporte_desde_agente, reporte_desde_log             # noqa: E402
from freshcheck.scenarios import FALLAS, LIMITE, NOMINALES                                       # noqa: E402

ESPERADO = {
    "sin_dato": ("REVISION_MANUAL", "SENSOR_SIN_DATO"), "vacia": ("REVISION_MANUAL", "VACIA"),
    "formato": ("REVISION_MANUAL", "FORMATO_INVALIDO"), "nan": ("REVISION_MANUAL", "NAN"),
    "tipo": ("REVISION_MANUAL", "TIPO_INVALIDO"), "negra": ("REVISION_MANUAL", "OSCURA"),
    "saturada": ("REVISION_MANUAL", "SATURADA"), "banda_vacia": ("NINGUNA", "SIN_FRUTA"),
    "solapadas": ("REVISION_MANUAL", "SOLAPAMIENTO"),
}


def evaluar_fallas(df: pd.DataFrame, esc_nombre: str, n: int, excepciones: int) -> dict:
    """Devuelve instancias evaluadas y manejadas correctamente para un escenario de falla."""
    inst = ok = 0
    detalle = []
    if len(df) != n or excepciones:
        return dict(instancias=1, manejadas=0, detalle=[f"filas={len(df)} esperadas={n} excepciones={excepciones}"])
    if esc_nombre.startswith("F10"):
        inst = 1
        r = df.set_index("unit_id")
        cond = all(r.loc[i, "accion"] == "REVISION_MANUAL" and r.loc[i, "motivo_decision"] == "SENSOR_SIN_DATO"
                   for i in range(10, 15))
        cond = cond and r.loc[15, "accion"] == "REVISION_MANUAL" and r.loc[15, "motivo_decision"] == "CAMARA_EN_FALLA"
        cond = cond and r.loc[16, "estado_camara"] == "OK" and r.loc[16, "accion"] in ("CONTINUAR", "DESVIAR")
        cond = cond and (r.loc[12:14, "estado_camara"] == "EN_FALLA").all()
        return dict(instancias=inst, manejadas=int(cond), detalle=[] if cond else ["secuencia de falla/recuperación incorrecta"])
    inyectadas = df[df["falla_inyectada"] != ""]
    for _, f in inyectadas.iterrows():
        inst += 1
        if esc_nombre.startswith("F11"):
            bien = f["accion"] in ("REVISION_MANUAL", "NINGUNA")
        else:
            acc, mot = ESPERADO[f["falla_inyectada"]]
            bien = f["accion"] == acc and f["motivo_decision"] == mot
        ok += int(bien)
        if not bien:
            detalle.append(f"unit {f['unit_id']} ({f['falla_inyectada']}): {f['accion']}/{f['motivo_decision']}")
    return dict(instancias=inst, manejadas=ok, detalle=detalle)


def resumen_medida(valores: pd.Series) -> dict:
    return dict(media=float(valores.mean()), sd=float(valores.std(ddof=1)) if len(valores) > 1 else 0.0,
                minimo=float(valores.min()), maximo=float(valores.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--modelo", default=str(RAIZ / "models" / "freshcheck_model.joblib"))
    ap.add_argument("--salida", default=str(RAIZ / "logs" / "pruebas"))
    a = ap.parse_args()

    umbrales = json.loads((RAIZ / "config" / "umbrales.json").read_text(encoding="utf-8"))
    n_u = 40 if a.rapido else umbrales["unidades_por_corrida"]
    reps = 1 if a.rapido else umbrales["repeticiones"]
    base = umbrales["semilla_base_pruebas"]
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    clf = ClasificadorEstado.cargar(a.modelo)

    t0 = time.time()
    logs, filas_run, reps_prov = [], [], []
    grupos = [("nominal", NOMINALES), ("limite", LIMITE)]
    idx_esc = 0
    for estrato, escenarios in grupos:
        for esc in escenarios:
            for rep in range(reps):
                seed = base + 100 * idx_esc + rep
                df, agente = ejecutar_escenario(clf, esc, n_u, seed, rep)
                df["estrato"] = estrato
                logs.append(df)
                r2, d0, d1 = m2_reduccion_defectuosas(df)
                p95, tard = m4_latencia(df)
                sin, con = tasa_defecto_empaque(df)
                rep_ag, rep_lg = reporte_desde_agente(agente), reporte_desde_log(df)
                cons = consistencia(rep_ag, rep_lg)
                rep_lg = rep_lg.assign(run_id=df["run_id"].iloc[0], consistencia_con_agente=cons)
                reps_prov.append(rep_lg)
                filas_run.append(dict(
                    estrato=estrato, escenario=esc.nombre, rep=rep, seed=seed, unidades=len(df),
                    M1_exactitud=m1_exactitud(df), M2_reduccion=r2, D0=d0, D1=d1,
                    tasa_defecto_sin_agente=sin, tasa_defecto_con_agente=con,
                    M3_descarte_indebido=m3_descarte_indebido(df), M4_latencia_p95_ms=p95,
                    M4_tardias=tard, M6_consistencia=cons))
            print(f"[{time.time() - t0:6.0f} s] {esc.nombre} listo", flush=True)
            idx_esc += 1

    log_cls = pd.concat(logs, ignore_index=True)
    res_run = pd.DataFrame(filas_run)
    log_cls.to_csv(salida / "log_clasificacion.csv", index=False)
    res_run.to_csv(salida / "resultados_por_corrida.csv", index=False)
    pd.concat(reps_prov, ignore_index=True).to_csv(salida / "reportes_proveedor.csv", index=False)

    # ---- M5: fallas ----
    logs_f, filas_f = [], []
    for k, esc in enumerate(FALLAS):
        n_f = 60 if not esc.nombre.startswith("F11") else 100
        if a.rapido:
            n_f = max(n_f // 2, 20) if not esc.nombre.startswith("F10") else 30
        for rep in range(reps):
            seed = base + 5000 + 100 * k + rep
            excepciones = 0
            try:
                df, _ = ejecutar_escenario(clf, esc, n_f, seed, rep)
            except Exception:
                excepciones = 1
                traceback.print_exc()
                df = pd.DataFrame()
            ev = evaluar_fallas(df, esc.nombre, n_f, excepciones) if len(df) else dict(instancias=1, manejadas=0, detalle=["excepción"])
            if len(df):
                logs_f.append(df)
            filas_f.append(dict(escenario=esc.nombre, rep=rep, seed=seed, unidades=n_f,
                                instancias=ev["instancias"], manejadas=ev["manejadas"],
                                M5_tasa=ev["manejadas"] / max(ev["instancias"], 1),
                                detalle=" | ".join(ev["detalle"][:3])))
        print(f"[{time.time() - t0:6.0f} s] {esc.nombre} listo", flush=True)
    pd.concat(logs_f, ignore_index=True).to_csv(salida / "log_fallas.csv", index=False)
    res_f = pd.DataFrame(filas_f)
    res_f.to_csv(salida / "resultados_fallas.csv", index=False)

    # ---- Resumen y cumplimiento ----
    def cumple(valor, u):
        return bool(valor >= u["umbral"]) if u["sentido"] == ">=" else bool(valor <= u["umbral"])

    nom = res_run[res_run.estrato == "nominal"]
    lim = res_run[res_run.estrato == "limite"]
    resumen = {"configuracion": dict(unidades_por_corrida=n_u, repeticiones=reps, semilla_base=base,
                                      segundos=round(time.time() - t0, 1)), "medidas": {}, "por_escenario": {}}
    M = resumen["medidas"]
    # Estadística: media global = media de las medias por escenario; SD = desviación de las medias por corrida
    def agrega(df, col):
        por_esc = df.groupby("escenario")[col].mean()
        return dict(media=float(por_esc.mean()), sd_entre_corridas=float(df[col].std(ddof=1)),
                    peor_escenario=str(por_esc.idxmin() if col in ("M1_exactitud", "M2_reduccion") else por_esc.idxmax()),
                    valor_peor_escenario=float(por_esc.min() if col in ("M1_exactitud", "M2_reduccion") else por_esc.max()),
                    n_escenarios=int(df["escenario"].nunique()), n_corridas=int(len(df)))

    for nombre, df, col in [("M1_exactitud_nominal", nom, "M1_exactitud"), ("M1_exactitud_limite", lim, "M1_exactitud"),
                            ("M3_descarte_indebido_nominal", nom, "M3_descarte_indebido"),
                            ("M3_descarte_indebido_limite", lim, "M3_descarte_indebido")]:
        M[nombre] = agrega(df, col)
        M[nombre]["cumple"] = cumple(M[nombre]["media"], umbrales[nombre])
    M["M2_reduccion_defectuosas"] = agrega(res_run, "M2_reduccion")
    M["M2_reduccion_defectuosas"]["cumple"] = cumple(M["M2_reduccion_defectuosas"]["media"], umbrales["M2_reduccion_defectuosas"])
    M["M2_reduccion_defectuosas"]["tasa_defecto_sin_agente_media"] = float(res_run["tasa_defecto_sin_agente"].mean())
    M["M2_reduccion_defectuosas"]["tasa_defecto_con_agente_media"] = float(res_run["tasa_defecto_con_agente"].mean())
    p95_glob = float(np.percentile(log_cls["latencia_ms"], 95))
    M["M4_latencia_p95_ms"] = dict(media=p95_glob, maximo_observado=float(log_cls["latencia_ms"].max()),
                                   sd_entre_corridas=float(res_run["M4_latencia_p95_ms"].std(ddof=1)),
                                   cumple=cumple(p95_glob, umbrales["M4_latencia_p95_ms"]))
    M["M4_decisiones_tardias"] = dict(total=int(res_run["M4_tardias"].sum()),
                                      cumple=cumple(int(res_run["M4_tardias"].sum()), umbrales["M4_decisiones_tardias"]))
    tasa_f = float(res_f["manejadas"].sum() / res_f["instancias"].sum())
    M["M5_robustez_fallas"] = dict(media=tasa_f, instancias=int(res_f["instancias"].sum()),
                                   manejadas=int(res_f["manejadas"].sum()),
                                   n_escenarios=int(res_f["escenario"].nunique()),
                                   cumple=cumple(tasa_f, umbrales["M5_robustez_fallas"]))
    M["M6_consistencia_reporte"] = dict(media=float(res_run["M6_consistencia"].mean()),
                                        sd_entre_corridas=float(res_run["M6_consistencia"].std(ddof=1)),
                                        n_corridas=int(len(res_run)),
                                        cumple=cumple(float(res_run["M6_consistencia"].mean()), umbrales["M6_consistencia_reporte"]))
    for esc, g in res_run.groupby("escenario", sort=False):
        resumen["por_escenario"][esc] = {c: resumen_medida(g[c]) for c in
                                         ["M1_exactitud", "M2_reduccion", "M3_descarte_indebido", "M4_latencia_p95_ms"]}
    for esc, g in res_f.groupby("escenario", sort=False):
        resumen["por_escenario"][esc] = dict(M5_tasa=resumen_medida(g["M5_tasa"]),
                                             instancias=int(g["instancias"].sum()), manejadas=int(g["manejadas"].sum()))
    (salida / "resultados_resumen.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== RESUMEN ===")
    for k, v in M.items():
        print(f"{k:34s} media={v.get('media', v.get('total', 0)):.4f}  cumple={v['cumple']}")
    print(f"Tiempo total: {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
