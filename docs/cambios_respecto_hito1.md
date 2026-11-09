# Cambios respecto al Hito 1 (justificación escrita)

Fecha: 2026-11-08. Registrados **antes de entrenar cualquier modelo**. El Hito 1 se redactó con imágenes de internet y tres
clases; esta nota documenta por qué cambia el alcance y qué se hace en su lugar. Debe informarse al docente.

| Aspecto | Hito 1 | Ahora | Motivo |
|---|---|---|---|
| Datos | FruitVision + ≥ 200 imágenes de internet por clase, con licencia y etiquetado manual | Solo FruitVision (imágenes originales) | Reunir y etiquetar ≈ 600 imágenes con licencia abierta no es viable en el plazo; FruitVision es suficiente y su licencia (CC BY-NC-ND 4.0) permite el uso académico |
| Clases | fresca, en maduración, dañada | fresca, dañada | «En maduración» solo habría salido de internet; FruitVision no la tiene. Queda como trabajo futuro |
| Pregunta | ¿cuánto se degrada al pasar de FruitVision a imágenes de internet? | ¿cuánto se degrada ante una fruta no vista? | Sin internet no hay cambio de dominio; se mide la generalización entrenando con 4 frutas y evaluando en la quinta |
| Partición | 70/15/15 por clase y fuente | 70/15/15 por clase y fruta (semilla 42, por grupos de duplicados) | Hay una sola fuente; la fruta pasa a ser el subgrupo de análisis |
| Línea base y comparación | ResNet-50 congelado; ajuste fino y EfficientNet-B0 | Sin cambios | — |

## Umbrales (declarados antes de experimentar)

| Hito 1 | Ahora | Motivo |
|---|---|---|
| Macro-F1 en prueba, todas las fuentes ≥ 0,90 | Macro-F1 en prueba ≥ 0,90 | Queda una sola fuente |
| Exactitud en prueba de FruitVision ≥ 95 % | Exactitud en prueba ≥ 95 % | Sin cambio de valor |
| Macro-F1 en prueba de internet ≥ 0,70 | Macro-F1 en fruta no vista ≥ 0,70 (media de las 5 frutas) | Reemplaza la degradación por cambio de dominio |
| Recall de «dañada» ≥ 0,95 | Igual | — |
| Mejora ≥ 5 puntos de macro-F1 sobre la línea base | Reducción relativa del error (1 − macro-F1) ≥ 30 % y mayor que la desviación entre semillas | Con dos clases y fondos simples la línea base puede superar 0,95, y 5 puntos serían inalcanzables |

Versión máquina-legible y registro de cambios: `config/umbrales_v2.json`.

## Limitaciones nuevas que se reportarán

- Sin identificador del fruto físico: fotos distintas de la misma fruta pueden quedar en particiones diferentes e inflar los resultados.
- Imágenes de laboratorio con fondos simples: los resultados no se extrapolan a una línea de empaque real.
- Con dos clases el desempeño puede acercarse al techo y el análisis de errores (≥ 10 casos) dependerá del tamaño de la prueba.
