"""Escenarios de prueba de la Fase 3 (definidos antes de ejecutar las pruebas)."""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, Tuple

from .simulator import Condiciones, Escenario, Lote

BASE = Condiciones()

# ---- Clasificación (M1-M4, M6): 8 nominales + 5 límite ----
NOMINALES: Tuple[Escenario, ...] = (
    Escenario("N1_nominal", "Condiciones nominales de operación"),
    Escenario("N2_luz_baja", "Iluminación 20 % por debajo de la nominal",
              cond=replace(BASE, ganancia_media=0.80)),
    Escenario("N3_luz_alta", "Iluminación 25 % por encima de la nominal",
              cond=replace(BASE, ganancia_media=1.25)),
    Escenario("N4_ruido", "Ruido de sensor alto y desenfoque leve",
              cond=replace(BASE, ruido_sigma=8.0, desenfoque_px=3)),
    Escenario("N5_posicion", "Fruta muy desplazada, escala y rotación extremas",
              cond=replace(BASE, jitter_px=26.0, escala=(0.80, 1.20))),
    Escenario("N6_banda_lenta", "Banda lenta (0,3 m/s)", velocidad_ms=0.30),
    Escenario("N7_una_variedad", "Lote de una sola variedad (naranja)", variedades=("naranja",)),
    Escenario("N8_lote_critico", "Lote con 40 % de fruta dañada",
              lotes=(Lote("L-901", "Proveedor C", (0.30, 0.30, 0.40)),)),
)

LIMITE: Tuple[Escenario, ...] = (
    Escenario("L1_luz_muy_baja", "Iluminación 55 % de la nominal (fuera del rango de entrenamiento)",
              cond=replace(BASE, ganancia_media=0.55)),
    Escenario("L2_sobreexposicion", "Iluminación 155 % de la nominal (fuera del rango de entrenamiento)",
              cond=replace(BASE, ganancia_media=1.55)),
    Escenario("L3_desenfoque", "Desenfoque de movimiento fuerte y ruido alto",
              cond=replace(BASE, desenfoque_px=9, ruido_sigma=9.0)),
    Escenario("L4_frontera", "Casos fronterizos: maduración incipiente y daños pequeños",
              cond=replace(BASE, frontera=True)),
    Escenario("L5_alta_densidad", "Banda rápida (1,5 m/s) y unidades muy próximas",
              velocidad_ms=1.5, separacion_m=(0.06, 0.15)),
)

# ---- Fallas de percepción (M5): 10 casos dirigidos + 1 mezcla aleatoria ----
def _falla(nombre: str, tipo: str, descr: str) -> Escenario:
    return Escenario(nombre, descr, plan_fallas=((10, tipo), (30, tipo), (50, tipo)))


FALLAS: Tuple[Escenario, ...] = (
    _falla("F01_sin_dato", "sin_dato", "El sensor no entrega imagen (None)"),
    _falla("F02_imagen_vacia", "vacia", "Imagen de tamaño cero"),
    _falla("F03_formato", "formato", "Imagen en escala de grises (formato inesperado)"),
    _falla("F04_nan", "nan", "Imagen con valores NaN"),
    _falla("F05_tipo", "tipo", "El percepto trae una cadena de texto en lugar de imagen"),
    _falla("F06_lente_tapado", "negra", "Imagen completamente oscura (lente obstruido)"),
    _falla("F07_saturada", "saturada", "Imagen saturada en blanco (destello)"),
    _falla("F08_banda_vacia", "banda_vacia", "Disparo con la banda vacía (sin fruta)"),
    _falla("F09_solapadas", "solapadas", "Dos frutas en contacto en la misma imagen"),
    Escenario("F10_rafaga", "Ráfaga de 5 imágenes sin dato seguidas (cámara en falla) y recuperación",
              plan_fallas=tuple((i, "sin_dato") for i in range(10, 15))),
    Escenario("F11_mezcla", "Mezcla aleatoria de fallas (12 % de las unidades)", prob_falla=0.12),
)


def todos_clasificacion() -> Tuple[Escenario, ...]:
    return NOMINALES + LIMITE


def por_nombre() -> Dict[str, Escenario]:
    return {e.nombre: e for e in NOMINALES + LIMITE + FALLAS}
