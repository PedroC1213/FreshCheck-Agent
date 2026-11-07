"""Constantes y rutas del proyecto final (FreshCheck-Agent v2)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CLASES = ("fresca", "maduracion", "danada")
FUENTES = ("fruitvision", "internet")
PARTICIONES = ("train", "val", "test")
FRACCIONES = (0.70, 0.15, 0.15)      # declarado en el Hito 1
SEED = 42                            # semilla de la partición (Hito 1)
SEMILLAS = (42, 43, 44)              # 3 semillas por modelo (Hito 1)
ALFAS = (1e-5, 1e-4, 1e-3, 1e-2)     # cuadrícula de regularización de la cabeza lineal
EXTENSIONES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
UMBRAL_HAMMING = 2                   # distancia máxima entre dHash para tratar dos imágenes como duplicadas


@dataclass(frozen=True)
class Rutas:
    raiz: Path = ROOT

    @property
    def raw(self) -> Path:
        return self.raiz / "data" / "raw"

    @property
    def manifiesto(self) -> Path:
        return self.raiz / "data" / "manifest.csv"

    @property
    def particiones(self) -> Path:
        return self.raiz / "data" / "splits.csv"

    @property
    def fuentes_internet(self) -> Path:
        return self.raiz / "data" / "sources_internet.csv"

    @property
    def features(self) -> Path:
        return self.raiz / "experiments" / "features" / "resnet50_imagenet.npz"

    @property
    def baseline(self) -> Path:
        return self.raiz / "experiments" / "baseline"

    @property
    def umbrales(self) -> Path:
        return self.raiz / "config" / "umbrales_v2.json"


def cargar_umbrales(rutas: Rutas | None = None) -> dict:
    rutas = rutas or Rutas()
    with open(rutas.umbrales, encoding="utf-8") as f:
        return json.load(f)["umbrales"]
