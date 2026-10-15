"""
Entrena el clasificador de FreshCheck-Agent.

Uso:
    python scripts/train.py                      # imágenes sintéticas del simulador
    python scripts/train.py --real-data RUTA     # carpetas RUTA/fresca, RUTA/maduracion, RUTA/danada

Con --real-data se usan imágenes reales (por ejemplo FruitVision o fotografías propias). La
segmentación asume una banda/fondo de color uniforme visible en el borde de la imagen.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from freshcheck.config import CLASES, IMG_SIZE                       # noqa: E402
from freshcheck.model import calibrar_banda, construir_matriz, entrenar  # noqa: E402
from freshcheck.perception import observar_banda                     # noqa: E402
from freshcheck.simulator import generar_dataset                     # noqa: E402

SEED_ENTRENAMIENTO = 1000      # semillas disjuntas: entrenamiento 1000, validación 2000, pruebas 9000+
SEED_VALIDACION = 2000


def cargar_carpetas(ruta: Path):
    imgs, ets = [], []
    for clase in CLASES:
        carpeta = ruta / clase
        if not carpeta.is_dir():
            raise SystemExit(f"Falta la carpeta {carpeta}")
        for f in sorted(carpeta.iterdir()):
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            im = cv2.imread(str(f))
            if im is None:
                continue
            imgs.append(cv2.resize(im, (IMG_SIZE, IMG_SIZE)))
            ets.append(clase)
    return imgs, ets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-por-clase", type=int, default=1500)
    ap.add_argument("--n-val-por-clase", type=int, default=300)
    ap.add_argument("--real-data", type=str, default=None)
    ap.add_argument("--salida", type=str, default=str(RAIZ / "models"))
    a = ap.parse_args()

    t0 = time.time()
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)

    if a.real_data:
        imgs, ets = cargar_carpetas(Path(a.real_data))
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(imgs))
        corte = int(0.8 * len(idx))
        tr = [(imgs[i], ets[i]) for i in idx[:corte]]
        va = [(imgs[i], ets[i]) for i in idx[corte:]]
        belt_l = float(np.median([observar_banda(im)[1] for im, _ in tr]))
        fuente = f"datos_reales:{a.real_data}"
        tr_i, tr_e = [x for x, _ in tr], [y for _, y in tr]
        va_i, va_e = [x for x, _ in va], [y for _, y in va]
    else:
        belt_l = calibrar_banda()
        tr_i, tr_e = generar_dataset(a.n_por_clase, SEED_ENTRENAMIENTO)
        va_i, va_e = generar_dataset(a.n_val_por_clase, SEED_VALIDACION)
        fuente = "simulador"

    print(f"Luminancia nominal de la banda (calibración): {belt_l:.2f}")
    X, y = construir_matriz(tr_i, tr_e, belt_l)
    Xv, yv = construir_matriz(va_i, va_e, belt_l)
    modelo = entrenar(X, y, belt_l, metadatos={"fuente": fuente})
    acc_val = float((modelo.modelo.predict(Xv) == yv).mean())
    print(f"Entrenamiento: {len(y)} muestras | validación: {len(yv)} muestras | exactitud val = {acc_val:.4f}")

    modelo.guardar(salida / "freshcheck_model.joblib")
    np.savez_compressed(salida / "datos_entrenamiento.npz", X=X, y=y)
    with open(salida / "entrenamiento.json", "w", encoding="utf-8") as f:
        json.dump(dict(fuente=fuente, n_entrenamiento=int(len(y)), n_validacion=int(len(yv)),
                       exactitud_validacion=acc_val, belt_l_nominal=belt_l,
                       seed_entrenamiento=SEED_ENTRENAMIENTO, seed_validacion=SEED_VALIDACION,
                       segundos=round(time.time() - t0, 1)), f, indent=2, ensure_ascii=False)
    print(f"Modelo guardado en {salida / 'freshcheck_model.joblib'} ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
