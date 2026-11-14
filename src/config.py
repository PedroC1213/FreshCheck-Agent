"""Constantes y rutas del proyecto final (FreshCheck-Agent v2, solo FruitVision)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CLASES = ("fresca", "danada")        # FruitVision no tiene "en maduracion" (ver docs/cambios_respecto_hito1.md)
FRUTAS = ("manzana", "banano", "mango", "naranja", "uva")
PARTICIONES = ("train", "val", "test")
FRACCIONES = (0.70, 0.15, 0.15)      # declarado en el Hito 1
SEED = 42                            # semilla de la particion (Hito 1)
SEMILLAS = (42, 43, 44)              # 3 semillas por modelo (Hito 1)
ALFAS = (1e-5, 1e-4, 1e-3, 1e-2)     # cuadricula de regularizacion de la cabeza lineal (Hito 2)
ALFAS_AMPLIADA = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)   # Hito 3: el optimo del Hito 2 quedo en el borde
ARQUITECTURAS = ("resnet50", "efficientnet_b0")
EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
UMBRAL_HAMMING = 2                   # distancia maxima entre dHash para tratar dos imagenes como duplicadas
MAX_GRUPO = 20                       # un grupo mayor se considera falso positivo del hash y se deshace


@dataclass(frozen=True)
class Rutas:
    raiz: Path = ROOT

    @property
    def raw(self) -> Path:
        return self.raiz / "data" / "raw"

    @property
    def fruitvision(self) -> Path:
        return self.raw / "fruitvision"

    @property
    def manifiesto(self) -> Path:
        return self.raiz / "data" / "manifest.csv"

    @property
    def particiones(self) -> Path:
        return self.raiz / "data" / "splits.csv"

    @property
    def features(self) -> Path:
        return self.raiz / "experiments" / "features" / "resnet50_imagenet.npz"

    @property
    def baseline(self) -> Path:
        return self.raiz / "experiments" / "baseline"

    def cache_imgs(self, lado: int = 256) -> Path:
        return self.raiz / "experiments" / "cache" / f"imgs_{lado}.npy"

    @property
    def final(self) -> Path:
        return self.raiz / "experiments" / "final"

    @property
    def modelos(self) -> Path:
        return self.raiz / "models_v2"

    @property
    def umbrales(self) -> Path:
        return self.raiz / "config" / "umbrales_v2.json"


def cargar_umbrales(rutas: Rutas | None = None) -> dict:
    rutas = rutas or Rutas()
    with open(rutas.umbrales, encoding="utf-8") as f:
        return json.load(f)["umbrales"]
