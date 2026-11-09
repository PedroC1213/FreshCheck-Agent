"""Prueba de humo con datos SINTETICOS (no requiere torch ni descargas).

Verifica: manifiesto, duplicados (y tope de grupos), particion sin fuga, linea base,
generalizacion a fruta no vista y archivos de salida.
Uso:  python -m tests.test_pipeline
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src import config, make_manifest, split, train_baseline
from src.config import Rutas

COLOR = {"fresca": (30, 170, 40), "danada": (90, 50, 20)}
TINTE = {"manzana": (40, 0, 0), "banano": (0, 40, 0), "mango": (0, 0, 40), "naranja": (30, 30, 0), "uva": (0, 30, 30)}


def _img(color, rng, ruta):
    a = np.clip(np.array(color) + rng.normal(0, 25, (48, 48, 3)), 0, 255).astype(np.uint8)
    Image.fromarray(a).save(ruta)


def construir_datos(raiz: Path, rng, por_estrato=24):
    for clase in config.CLASES:
        d = raiz / "data" / "raw" / "fruitvision" / clase
        d.mkdir(parents=True, exist_ok=True)
        for fruta in config.FRUTAS:
            for i in range(por_estrato):
                _img(np.array(COLOR[clase]) + np.array(TINTE[fruta]), rng, d / f"{fruta}__{i:03d}.png")
    # duplicado exacto (misma clase y fruta)
    f = raiz / "data" / "raw" / "fruitvision" / "fresca"
    shutil.copy(f / "manzana__000.png", f / "manzana__dup.png")


def main():
    rng = np.random.default_rng(0)
    raiz = Path(tempfile.mkdtemp())
    rutas = Rutas(raiz)
    (raiz / "config").mkdir()
    shutil.copy(config.ROOT / "config" / "umbrales_v2.json", raiz / "config" / "umbrales_v2.json")
    construir_datos(raiz, rng)

    # 1) manifiesto y duplicados
    make_manifest.main(["--raiz", str(raiz)])
    m = pd.read_csv(rutas.manifiesto)
    assert len(m) == 2 * 5 * 24 + 1, len(m)
    assert set(m["fruta"]) == set(config.FRUTAS), set(m["fruta"])
    assert (m.groupby("grupo").size() > 1).sum() >= 1, "no se detecto el duplicado"

    # 2) tope de grupos: 30 hashes identicos = falso positivo -> se deshace; 2 identicos aparte se conservan
    h = np.array([5] * 30 + [1 << 40, 1 << 40, 7 << 20], dtype=np.uint64)
    et, deshechos = make_manifest.agrupar_duplicados(h, 2, max_grupo=20)
    assert deshechos == 1 and len(set(et[:30])) == 30, "el grupo grande no se deshizo"
    assert et[30] == et[31] and et[31] != et[32], "el par duplicado debio conservarse"

    # 3) particion y fuga
    split.main(["--raiz", str(raiz)])
    s = pd.read_csv(rutas.particiones).merge(m, on="id")
    assert (s.groupby("grupo")["particion"].nunique() == 1).all(), "fuga: un grupo en dos particiones"
    frac = s["particion"].value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.06 and abs(frac["val"] - 0.15) < 0.06, frac.to_dict()
    tabla = s.pivot_table(index=["clase", "fruta"], columns="particion", values="id", aggfunc="count", fill_value=0)
    assert (tabla > 0).all().all(), "algun estrato (clase, fruta) quedo sin imagenes en una particion"
    a = s.loc[s.id.str.endswith("fresca/manzana__000.png"), "particion"].iloc[0]
    b = s.loc[s.id.str.endswith("fresca/manzana__dup.png"), "particion"].iloc[0]
    assert a == b, "el duplicado quedo en otra particion"

    # 4) features sinteticas (media RGB) -> linea base
    X = [np.asarray(Image.open(raiz / "data" / "raw" / i).convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(0)
         for i in s["id"]]
    ruta_f = raiz / "experiments" / "features" / "sint.npz"
    ruta_f.parent.mkdir(parents=True)
    np.savez_compressed(ruta_f, ids=np.array(s["id"].tolist()), X=np.array(X, dtype=np.float32), meta="{}")
    r = train_baseline.correr(rutas, ruta_f)
    assert r["piso_clase_mayoritaria"]["macro_f1"] < r["linea_base"]["macro_f1_media"]
    assert r["linea_base"]["macro_f1_media"] > 0.9, r["linea_base"]
    g = r["generalizacion_fruta_no_vista"]
    assert set(g["por_fruta"]) == set(config.FRUTAS) and g["macro_f1_media_5_frutas"] is not None
    for f in ("resultados_val.json", "grid_val.csv", "errores_val.csv", "matriz_confusion_val.png",
              "generalizacion_fruta_no_vista.csv", "entorno.json"):
        assert (rutas.baseline / f).exists(), f
    assert "no se uso" in r["conjunto"]
    nombres = {e["metrica"] for e in r["estado_umbrales_indicativo"]}
    assert nombres == {"macro_f1_prueba", "exactitud_prueba", "macro_f1_fruta_no_vista", "recall_danada_prueba"}, nombres
    json.loads((rutas.baseline / "resultados_val.json").read_text(encoding="utf-8"))
    print("\nOK: pipeline sintetico correcto ->", round(r["linea_base"]["macro_f1_media"], 3))
    shutil.rmtree(raiz)


if __name__ == "__main__":
    main()
