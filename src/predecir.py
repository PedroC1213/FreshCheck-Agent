"""Inferencia del prototipo: clasifica una foto de fruta como fresca o danada.

Usa, en este orden: el modelo ajustado elegido (models_v2/meta.json -> models_v2/<arq>.pt) o la cabeza de la
linea base (models_v2/baseline_head.joblib). Uso:  python -m src.predecir foto.jpg [--modelo auto|baseline|resnet50|efficientnet_b0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import config
from .config import Rutas

UMBRAL_CONFIANZA = 0.80
AVISO = ("Prototipo académico entrenado solo con FruitVision (manzana, banano, mango, naranja y uva; fondos simples). "
         "No detecta formalina ni otros daños, y no debe usarse para decisiones de seguridad alimentaria.")


def _prep(img: Image.Image, lado: int = 256, tam: int = 224):
    import torch
    img = img.convert("RGB")
    w, h = img.size
    e = lado / min(w, h)
    img = img.resize((max(lado, round(w * e)), max(lado, round(h * e))), Image.BILINEAR)
    w, h = img.size
    l, t = (w - lado) // 2, (h - lado) // 2
    img = img.crop((l, t, l + lado, t + lado))
    o = (lado - tam) // 2
    a = np.asarray(img.crop((o, o, o + tam, o + tam)), dtype=np.float32) / 255
    a = (a - np.array((0.485, 0.456, 0.406), dtype=np.float32)) / np.array((0.229, 0.224, 0.225), dtype=np.float32)
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


class Modelo:
    def __init__(self, nombre: str, fn):
        self.nombre, self._fn = nombre, fn

    def predecir(self, img: Image.Image) -> dict:
        p = float(self._fn(img))
        clase = "danada" if p >= 0.5 else "fresca"
        conf = p if clase == "danada" else 1 - p
        adv = []
        if conf < UMBRAL_CONFIANZA:
            adv.append("Confianza baja: se recomienda revisión humana.")
        if min(img.size) < 224:
            adv.append("La imagen es pequeña; el resultado puede ser poco fiable.")
        return {"clase": clase, "p_danada": p, "confianza": conf, "modelo": self.nombre, "advertencias": adv}


def _cargar_ajustado(ruta: Path) -> Modelo:
    import torch
    from .finetune import construir_modelo
    ck = torch.load(ruta, map_location="cpu")
    net, _, _ = construir_modelo(ck["arq"], "todo", pesos=False)
    net.load_state_dict({k: (v.float() if v.is_floating_point() else v) for k, v in ck["estado"].items()})
    net.eval()

    def fn(img):
        with torch.inference_mode():
            return torch.softmax(net(_prep(img, tam=ck["tam"])), dim=1)[0, 1].item()
    return Modelo(f"{ck['arq']} (ajuste fino)", fn)


def _cargar_baseline(ruta: Path) -> Modelo:
    import joblib
    import torch
    from torchvision import models
    d = joblib.load(ruta)
    pesos = models.ResNet50_Weights.IMAGENET1K_V2
    net = models.resnet50(weights=pesos)
    net.fc = torch.nn.Identity()
    net.eval()
    tfm = pesos.transforms()
    i = list(d["clases"]).index("danada")

    def fn(img):
        with torch.inference_mode():
            f = net(tfm(img.convert("RGB")).unsqueeze(0)).numpy()
        return d["clf"].predict_proba(d["scaler"].transform(f))[0, i]
    return Modelo("ResNet-50 congelado + cabeza lineal (línea base)", fn)


def modelos_disponibles(rutas: Rutas | None = None) -> dict[str, Path]:
    rutas = rutas or Rutas()
    out = {}
    for arq in config.ARQUITECTURAS:
        if (rutas.modelos / f"{arq}.pt").exists():
            out[arq] = rutas.modelos / f"{arq}.pt"
    if (rutas.modelos / "baseline_head.joblib").exists():
        out["baseline"] = rutas.modelos / "baseline_head.joblib"
    return out


def cargar_modelo(preferido: str = "auto", rutas: Rutas | None = None) -> Modelo:
    rutas = rutas or Rutas()
    disp = modelos_disponibles(rutas)
    if not disp:
        raise FileNotFoundError("No hay modelos en models_v2/. Ejecute python -m src.baseline_final (y run_final).")
    if preferido == "auto":
        meta = rutas.modelos / "meta.json"
        mejor = json.loads(meta.read_text(encoding="utf-8")).get("mejor") if meta.exists() else None
        preferido = mejor if mejor in disp else ("baseline" if "baseline" in disp else next(iter(disp)))
    if preferido not in disp:
        raise FileNotFoundError(f"Modelo '{preferido}' no disponible. Disponibles: {list(disp)}")
    return _cargar_baseline(disp[preferido]) if preferido == "baseline" else _cargar_ajustado(disp[preferido])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("imagen", type=Path)
    ap.add_argument("--modelo", default="auto")
    ap.add_argument("--raiz", type=Path, default=config.ROOT)
    a = ap.parse_args(argv)
    r = cargar_modelo(a.modelo, Rutas(a.raiz)).predecir(Image.open(a.imagen))
    print(f"{a.imagen.name}: {r['clase'].upper()}  (P(dañada) = {r['p_danada']:.3f}, confianza {r['confianza']:.3f}, modelo: {r['modelo']})")
    for adv in r["advertencias"]:
        print("  Aviso:", adv)
    print(AVISO)


if __name__ == "__main__":
    main()
