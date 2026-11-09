"""Prueba de humo con datos SINTETICOS (no requiere torch ni descargas).

Verifica: manifiesto (licencia y duplicados), particion sin fuga, linea base y metricas.
Uso:  python -m tests.test_pipeline
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from src import config, make_manifest, split, train_baseline
from src.config import Rutas

COLOR = {"fresca": (30, 170, 40), "maduracion": (220, 200, 30), "danada": (90, 50, 20)}


def _img(color, rng, ruta):
    a = np.clip(np.array(color) + rng.normal(0, 25, (48, 48, 3)), 0, 255).astype(np.uint8)
    Image.fromarray(a).save(ruta)


def construir_datos(raiz: Path, rng):
    filas_lic = []
    for fuente, clases, n in (("fruitvision", ("fresca", "danada"), 60), ("internet", config.CLASES, 40)):
        for c in clases:
            d = raiz / "data" / "raw" / fuente / c
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                nombre = f"banano__{i:03d}.png"
                _img(COLOR[c], rng, d / nombre)
                if fuente == "internet":
                    filas_lic.append({"archivo": nombre, "url": "https://ejemplo.org", "autor": "x", "licencia": "CC0"})
    # duplicado exacto de una imagen de FruitVision dentro de internet (misma clase)
    dup = raiz / "data" / "raw" / "internet" / "fresca" / "dup__000.png"
    shutil.copy(raiz / "data" / "raw" / "fruitvision" / "fresca" / "banano__000.png", dup)
    filas_lic.append({"archivo": dup.name, "url": "https://ejemplo.org", "autor": "x", "licencia": "CC0"})
    pd.DataFrame(filas_lic).drop_duplicates("archivo").to_csv(raiz / "data" / "sources_internet.csv", index=False)


def main():
    rng = np.random.default_rng(0)
    raiz = Path(tempfile.mkdtemp())
    rutas = Rutas(raiz)
    (raiz / "config").mkdir()
    shutil.copy(config.ROOT / "config" / "umbrales_v2.json", raiz / "config" / "umbrales_v2.json")
    construir_datos(raiz, rng)

    # 1) manifiesto
    make_manifest.main(["--raiz", str(raiz)])
    m = pd.read_csv(rutas.manifiesto)
    assert len(m) == 60 * 2 + 40 * 3 + 1, len(m)
    g = m.groupby("grupo").size()
    assert (g > 1).sum() >= 1, "no se detecto el duplicado"

    # 2) sin licencia debe fallar
    (raiz / "data" / "sources_internet.csv").write_text("archivo,url,autor,licencia\n")
    try:
        make_manifest.construir(rutas)
        raise AssertionError("debio fallar sin licencias")
    except SystemExit:
        pass

    # 3) particion y fuga
    construir_datos(raiz, rng)
    make_manifest.main(["--raiz", str(raiz)])
    # 3b) CSV guardado por Excel en espanol: punto y coma, BOM y acentos en cp1252
    csv_ = raiz / "data" / "sources_internet.csv"
    lic = pd.read_csv(csv_, dtype=str)
    lic.loc[0, "autor"] = "Fot\u00f3grafo \u00d1"
    csv_.write_bytes(lic.to_csv(index=False, sep=";").encode("cp1252"))
    make_manifest.main(["--raiz", str(raiz)])
    csv_.write_bytes(b"\xef\xbb\xbf" + lic.to_csv(index=False, sep=";").encode("utf-8"))
    make_manifest.main(["--raiz", str(raiz)])
    construir_datos(raiz, rng)
    make_manifest.main(["--raiz", str(raiz)])
    split.main(["--raiz", str(raiz)])
    s = pd.read_csv(rutas.particiones).merge(pd.read_csv(rutas.manifiesto), on="id")
    assert (s.groupby("grupo")["particion"].nunique() == 1).all(), "fuga: un grupo en dos particiones"
    frac = s["particion"].value_counts(normalize=True)
    assert abs(frac["train"] - 0.70) < 0.06 and abs(frac["val"] - 0.15) < 0.06, frac.to_dict()
    a = s.loc[s.id.str.endswith("fruitvision/fresca/banano__000.png"), "particion"].iloc[0]
    b = s.loc[s.id.str.endswith("internet/fresca/dup__000.png"), "particion"].iloc[0]
    assert a == b, "el duplicado quedo en otra particion"

    # 4) features sinteticas (media RGB) -> linea base
    X = []
    for i in s["id"]:
        X.append(np.asarray(Image.open(raiz / "data" / "raw" / i).convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(0))
    ruta_f = raiz / "experiments" / "features" / "sint.npz"
    ruta_f.parent.mkdir(parents=True)
    np.savez_compressed(ruta_f, ids=np.array(s["id"].tolist()), X=np.array(X, dtype=np.float32), meta="{}")
    r = train_baseline.correr(rutas, ruta_f)
    assert r["piso_clase_mayoritaria"]["macro_f1"] < r["linea_base"]["macro_f1_media"]
    assert r["linea_base"]["macro_f1_media"] > 0.9, r["linea_base"]
    for f in ("resultados_val.json", "grid_val.csv", "errores_val.csv", "matriz_confusion_val.png", "entorno.json"):
        assert (rutas.baseline / f).exists(), f
    assert "test" not in r["conjunto"].split("(")[0]
    print("\nOK: pipeline sintetico correcto ->", r["linea_base"]["macro_f1_media"])
    shutil.rmtree(raiz)


if __name__ == "__main__":
    main()
