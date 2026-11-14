"""Prueba de humo del Hito 3 con datos SINTETICOS y modelos SIN pesos preentrenados (no descarga nada).

Recorre: cache -> tune -> baseline_final -> run_final (ajuste fino, prueba, fruta no vista) -> evaluar_final ->
guardar/cargar modelo -> inferencia. Requiere torch; si no esta instalado se omite.
Uso:  python -m tests.test_final
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch  # noqa: F401
except ImportError:
    print("OMITIDO: torch no esta instalado")
    raise SystemExit(0)

from src import baseline_final, cache, config, evaluar_final, finetune, make_manifest, predecir, run_final, split, tune
from src.config import Rutas
from tests.test_pipeline import construir_datos


def main():
    rng = np.random.default_rng(1)
    raiz = Path(tempfile.mkdtemp())
    rutas = Rutas(raiz)
    (raiz / "config").mkdir()
    shutil.copy(config.ROOT / "config" / "umbrales_v2.json", raiz / "config" / "umbrales_v2.json")
    construir_datos(raiz, rng, por_estrato=16)
    make_manifest.main(["--raiz", str(raiz)])
    split.main(["--raiz", str(raiz)])
    config.SEMILLAS = (42, 43)           # solo para acelerar la prueba

    # cache
    cache.construir(rutas, lado=72)
    datos = finetune.cargar_datos(rutas, 72)
    assert datos[1].shape[1:] == (72, 72, 3) and len(datos[1]) == len(datos[0])

    # el entrenamiento aprende: ResNet-50 sin pesos, todas las capas, clases separables por color
    cfg = finetune.Cfg(arq="resnet50", epocas=6, lote=16, tam=64, lado=72, sin_pesos=True, descongelar="todo", lr=1e-3, evaluar_test=True)
    r = finetune.entrenar(cfg, datos, "cpu", verbose=False)
    assert r["historial"][-1]["loss_train"] < r["historial"][0]["loss_train"] and r["test"]["exactitud"] > 0.9, r["test"]["exactitud"]

    # ajuste fino directo, ambos modelos, ambos modos de descongelado
    for arq, desc in (("resnet50", "ultimo_bloque"), ("efficientnet_b0", "todo")):
        cfg = finetune.Cfg(arq=arq, epocas=2, lote=16, tam=64, lado=72, sin_pesos=True, descongelar=desc, evaluar_test=True)
        r = finetune.entrenar(cfg, datos, "cpu", verbose=False)
        assert len(r["historial"]) == 2 and r["pred_test"] is not None and len(r["pred_test"]) == (datos[0].particion == "test").sum()
        assert set(r["pred_test"].columns) == {"id", "fruta", "y_real", "y_pred", "p_danada"}
        assert 0 <= r["test"]["exactitud"] <= 1
        finetune.guardar_modelo(r["net"], rutas.modelos / f"{arq}_prueba.pt", cfg)
    # el modo estandar NO debe predecir prueba sin --evaluar-test
    r0 = finetune.entrenar(finetune.Cfg(arq="efficientnet_b0", epocas=1, lote=16, tam=64, lado=72, sin_pesos=True), datos, "cpu", verbose=False)
    assert r0["pred_test"] is None and "test" not in r0
    # fruta no vista
    rf = finetune.entrenar(finetune.Cfg(arq="efficientnet_b0", epocas=1, lote=16, tam=64, lado=72, sin_pesos=True,
                                        modo="fruta_no_vista", fruta="uva"), datos, "cpu", verbose=False)
    assert set(rf["pred_val"]["fruta"]) == {"uva"} and rf["pred_test"] is None

    # tune (2 lr x 2 descongelar) solo con efficientnet para ir rapido
    tune.main(["--raiz", str(raiz), "--arqs", "efficientnet_b0", "--epocas", "1", "--lado", "72", "--tam", "64", "--sin-pesos"])
    cfgs = json.loads((rutas.final / "tuning" / "mejor_config.json").read_text())
    assert set(cfgs) == {"efficientnet_b0"} and len(pd.read_csv(rutas.final / "tuning" / "grid_val.csv")) == 4
    cfgs["resnet50"] = {"lr": 1e-4, "descongelar": "ultimo_bloque", "epocas": 1, "mejor_epoca": 1, "macro_f1_val": 0.0}
    (rutas.final / "tuning" / "mejor_config.json").write_text(json.dumps(cfgs))

    # linea base final con features sinteticas (media RGB)
    m = pd.read_csv(rutas.manifiesto)
    X = [np.asarray(Image.open(rutas.raw / i).convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(0) for i in m["id"]]
    ruta_f = rutas.features
    ruta_f.parent.mkdir(parents=True)
    np.savez_compressed(ruta_f, ids=np.array(m["id"].tolist()), X=np.array(X, dtype=np.float32), meta="{}")
    baseline_final.correr(rutas, semillas=config.SEMILLAS)
    assert (rutas.final / "baseline" / "s42" / "pred_test.csv").exists() and (rutas.modelos / "baseline_head.joblib").exists()
    assert len(list((rutas.final / "baseline" / "s42").glob("fnv_*.csv"))) == 5

    # corridas finales
    run_final.main(["--raiz", str(raiz), "--sin-pesos", "--lado", "72", "--tam", "64"])
    ej = json.loads((rutas.final / "resumen_ejecucion.json").read_text())
    assert ej["mejor_arquitectura"] in config.ARQUITECTURAS
    assert (rutas.modelos / "resnet50.pt").exists() and (rutas.modelos / "meta.json").exists()
    assert len(list((rutas.final / ej["mejor_arquitectura"]).glob("fnv_*_s*.csv"))) == 5 * len(config.SEMILLAS)

    # agregacion
    evaluar_final.main(["--raiz", str(raiz)])
    R = json.loads((rutas.final / "resumen.json").read_text())
    assert set(R["modelos"]) == {"baseline", "resnet50", "efficientnet_b0"}, set(R["modelos"])
    assert {u["metrica"] for u in R["umbrales"]} == {"macro_f1_prueba", "exactitud_prueba", "macro_f1_fruta_no_vista",
                                                    "recall_danada_prueba", "reduccion_error_sobre_linea_base"}
    assert R["modelos"][R["mejor_modelo"]]["fruta_no_vista"] is not None and "mcnemar" in R
    assert (rutas.final / "figuras" / "macro_f1_modelos.png").exists() and (rutas.final / "figuras" / "matriz_confusion_mejor.png").exists()

    # inferencia con el modelo guardado (sin pesos de ImageNet) y con la cabeza de la linea base no se prueba aqui
    mod = predecir._cargar_ajustado(rutas.modelos / "efficientnet_b0_prueba.pt")
    out = mod.predecir(Image.new("RGB", (120, 90), COLOR_DANADA))
    assert out["clase"] in config.CLASES and 0 <= out["p_danada"] <= 1 and out["advertencias"], out
    print("\nOK: Hito 3 sintetico correcto ->", {k: round(R['modelos'][k]['test_media']['macro_f1'], 3) for k in R['modelos']})
    if os.environ.get("FRESHCHECK_KEEP"):
        print("Datos sinteticos conservados en", raiz)
    else:
        shutil.rmtree(raiz)


COLOR_DANADA = (90, 50, 20)

if __name__ == "__main__":
    main()
