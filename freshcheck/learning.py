"""
Aprendizaje y adaptación (componente 7 de IBM).

El personal de calidad corrige clasificaciones erróneas o etiqueta unidades nuevas; esas muestras
se acumulan en ``RetroalimentacionCalidad`` y se usan para reentrenar el modelo de forma periódica
y offline (el agente no se modifica durante la operación en línea).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from .config import CLASES
from .model import ClasificadorEstado, construir_matriz, entrenar


@dataclass
class RetroalimentacionCalidad:
    imagenes: List[np.ndarray] = field(default_factory=list)
    etiquetas: List[str] = field(default_factory=list)

    def agregar(self, imagen: np.ndarray, etiqueta_correcta: str) -> None:
        if etiqueta_correcta not in CLASES:
            raise ValueError(f"Etiqueta inválida: {etiqueta_correcta}")
        self.imagenes.append(imagen)
        self.etiquetas.append(etiqueta_correcta)

    def __len__(self) -> int:
        return len(self.imagenes)


def reentrenar(base: ClasificadorEstado, X_base: np.ndarray, y_base: np.ndarray,
               retro: RetroalimentacionCalidad, peso_retro: int = 3, semilla: int = 0) -> ClasificadorEstado:
    """Reentrena con los datos originales + las muestras corregidas (repetidas ``peso_retro`` veces)."""
    if len(retro) == 0:
        return base
    Xn, yn = construir_matriz(retro.imagenes, retro.etiquetas, base.belt_l_nominal)
    X = np.vstack([X_base] + [Xn] * peso_retro)
    y = np.concatenate([y_base] + [yn] * peso_retro)
    return entrenar(X, y, base.belt_l_nominal, semilla=semilla,
                    metadatos={**(base.metadatos or {}), "reentrenado_con": int(len(retro))})
