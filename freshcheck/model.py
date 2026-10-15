"""
Modelo de clasificación del estado de la fruta (fresca / maduración / dañada).

Implementación: características de visión clásica (color, anomalías, textura) + gradient boosting
(``HistGradientBoostingClassifier``). Se eligió este enfoque, y no una CNN, por las razones
documentadas en la sección 9.6 del informe (CPU de un solo núcleo, trazabilidad y rapidez de
entrenamiento). El contrato ``predecir_proba`` permite sustituirlo por una CNN sin tocar el agente.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import CLASES
from .perception import (BELT_L_NOMINAL_POR_DEFECTO, NOMBRES_CARACTERISTICAS, observar_banda,
                         procesar_imagen)
from .simulator import Condiciones, renderizar_unidad


def calibrar_banda(n: int = 20, seed: int = 7) -> float:
    """Calibración: luminancia mediana (Lab-L) de la banda vacía con iluminación nominal."""
    rng = np.random.default_rng(seed)
    cond = Condiciones()
    valores = []
    for _ in range(n):
        img = renderizar_unidad(rng, None, None, cond, 1.0)
        valores.append(observar_banda(img)[1])
    return float(np.median(valores))


def construir_matriz(imagenes: Sequence[np.ndarray], etiquetas: Sequence[str],
                     belt_l_nominal: float) -> Tuple[np.ndarray, np.ndarray]:
    X: List[np.ndarray] = []
    y: List[int] = []
    for img, et in zip(imagenes, etiquetas):
        obs = procesar_imagen(img, belt_l_nominal)
        if obs.caracteristicas is None or obs.n_frutas != 1:
            continue
        X.append(obs.caracteristicas)
        y.append(CLASES.index(et))
    return np.vstack(X), np.array(y)


@dataclass
class ClasificadorEstado:
    """Artefacto entrenado: modelo + parámetro de calibración de la cámara."""
    modelo: HistGradientBoostingClassifier
    belt_l_nominal: float = BELT_L_NOMINAL_POR_DEFECTO
    metadatos: Optional[dict] = None

    def predecir_proba(self, x: np.ndarray) -> np.ndarray:
        """Probabilidades (fresca, maduracion, danada) para un vector de características."""
        p = self.modelo.predict_proba(x.reshape(1, -1))[0]
        out = np.zeros(len(CLASES))
        for idx, clase_id in enumerate(self.modelo.classes_):
            out[int(clase_id)] = p[idx]
        return out

    def guardar(self, ruta: str | Path) -> None:
        joblib.dump(self, ruta)

    @staticmethod
    def cargar(ruta: str | Path) -> "ClasificadorEstado":
        return joblib.load(ruta)


def entrenar(X: np.ndarray, y: np.ndarray, belt_l_nominal: float, semilla: int = 0,
             metadatos: Optional[dict] = None) -> ClasificadorEstado:
    modelo = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1.0,
        random_state=semilla,
    )
    modelo.fit(X, y)
    return ClasificadorEstado(modelo=modelo, belt_l_nominal=belt_l_nominal,
                              metadatos=metadatos or {"caracteristicas": list(NOMBRES_CARACTERISTICAS)})
