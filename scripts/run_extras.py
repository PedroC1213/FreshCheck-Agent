"""
Experimentos complementarios (no son medidas de desempeño del diseño):

  E1  Ablación: agente con modelo del mundo (basado en modelo) vs. sin él (reflejo simple).
  E2  Variedad no vista (pera) antes y después del reentrenamiento con retroalimentación del personal de calidad.

    python scripts/run_extras.py
"""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from freshcheck.evaluation import ejecutar_escenario, m1_exactitud            # noqa: E402
from freshcheck.learning import RetroalimentacionCalidad, reentrenar          # noqa: E402
from freshcheck.model import ClasificadorEstado                               # noqa: E402
from freshcheck.scenarios import FALLAS, LIMITE, NOMINALES                    # noqa: E402
from freshcheck.simulator import Escenario, generar_dataset                   # noqa: E402

REPS, N = 5, 100


def main():
    t0 = time.time()
    clf = ClasificadorEstado.cargar(RAIZ / "models" / "freshcheck_model.joblib")
    datos = np.load(RAIZ / "models" / "datos_entrenamiento.npz")
    salida = {}

    # ---------------- E1: ablación ----------------
    e1 = {}
    objetivo = [NOMINALES[0], NOMINALES[1], LIMITE[0], LIMITE[1]]
    for esc in objetivo:
        fila = {}
        for etiqueta, usar in (("con_modelo_del_mundo", True), ("sin_modelo_del_mundo", False)):
            accs = [m1_exactitud(ejecutar_escenario(clf, esc, N, 9800 + r, r, usar)[0]) for r in range(REPS)]
            fila[etiqueta] = dict(media=float(np.mean(accs)), sd=float(np.std(accs, ddof=1)))
        e1[esc.nombre] = fila
    # ráfaga de fallas: ¿cuántas unidades se mandan a inspección manual durante la recuperación?
    f10 = next(f for f in FALLAS if f.nombre.startswith("F10"))
    rec = {}
    for etiqueta, usar in (("con_modelo_del_mundo", True), ("sin_modelo_del_mundo", False)):
        df, _ = ejecutar_escenario(clf, f10, 30, 9850, 0, usar)
        rec[etiqueta] = dict(
            revision_manual_unidades_10_a_16=int((df.loc[10:16, "accion"] == "REVISION_MANUAL").sum()),
            estado_camara_unidad_13=str(df.loc[13, "estado_camara"]))
    salida["E1_ablacion"] = dict(exactitud=e1, rafaga_de_fallas=rec)
    print(f"[{time.time() - t0:5.0f} s] E1 listo", flush=True)

    # ---------------- E2: variedad no vista + reentrenamiento ----------------
    esc_pera = Escenario("E2_pera", "Variedad no vista en el entrenamiento (pera)", variedades=("pera",))
    antes = [m1_exactitud(ejecutar_escenario(clf, esc_pera, N, 9900 + r, r)[0]) for r in range(REPS)]
    imgs, ets = generar_dataset(40, 7000, variedades=("pera",))          # 120 muestras etiquetadas por "calidad"
    retro = RetroalimentacionCalidad()
    for im, et in zip(imgs, ets):
        retro.agregar(im, et)
    clf2 = reentrenar(clf, datos["X"], datos["y"], retro)
    despues = [m1_exactitud(ejecutar_escenario(clf2, esc_pera, N, 9900 + r, r)[0]) for r in range(REPS)]
    base_nom = [m1_exactitud(ejecutar_escenario(clf2, NOMINALES[0], N, 9800 + r, r)[0]) for r in range(REPS)]
    salida["E2_reentrenamiento"] = dict(
        muestras_de_retroalimentacion=len(retro),
        exactitud_pera_antes=dict(media=float(np.mean(antes)), sd=float(np.std(antes, ddof=1))),
        exactitud_pera_despues=dict(media=float(np.mean(despues)), sd=float(np.std(despues, ddof=1))),
        exactitud_nominal_tras_reentrenar=dict(media=float(np.mean(base_nom)), sd=float(np.std(base_nom, ddof=1))))
    print(f"[{time.time() - t0:5.0f} s] E2 listo", flush=True)

    (RAIZ / "logs" / "pruebas" / "experimentos_extra.json").write_text(
        json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(salida, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
