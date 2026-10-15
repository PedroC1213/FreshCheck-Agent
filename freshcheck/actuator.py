"""
Actuador: interfaz con el sistema de control de la banda (desviador hacia la línea de descarte).

``InterfazDesviador`` es el punto de integración con el hardware real (por ejemplo, una señal
a un PLC). ``DesviadorSimulado`` reproduce sus limitaciones físicas: plazo máximo de reacción y
tiempo de recarga entre accionamientos.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from .config import COOLDOWN_DESVIADOR_S


@dataclass
class ResultadoActuador:
    ejecutado: bool
    motivo: str          # OK | DECISION_TARDIA | DESVIADOR_OCUPADO


class InterfazDesviador(ABC):
    @abstractmethod
    def accionar(self, unit_id: int, t_actuacion_s: float, latencia_ms: float,
                 deadline_ms: float) -> ResultadoActuador:
        """Intenta desviar la unidad ``unit_id`` cuando llegue al desviador."""


class DesviadorSimulado(InterfazDesviador):
    def __init__(self, cooldown_s: float = COOLDOWN_DESVIADOR_S):
        self.cooldown_s = cooldown_s
        self._ultimo_s: Optional[float] = None
        self.historial = []          # (unit_id, ejecutado, motivo)

    def accionar(self, unit_id: int, t_actuacion_s: float, latencia_ms: float,
                 deadline_ms: float) -> ResultadoActuador:
        if latencia_ms > deadline_ms:
            res = ResultadoActuador(False, "DECISION_TARDIA")
        elif self._ultimo_s is not None and (t_actuacion_s - self._ultimo_s) < self.cooldown_s:
            res = ResultadoActuador(False, "DESVIADOR_OCUPADO")
        else:
            self._ultimo_s = t_actuacion_s
            res = ResultadoActuador(True, "OK")
        self.historial.append((unit_id, res.ejecutado, res.motivo))
        return res
