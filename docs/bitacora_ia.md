# Bitácora humano-IA — proyecto final (FreshCheck-Agent v2)

Registro de cada uso relevante de IA generativa. La bitácora de la actividad anterior está en `docs/Bitacora_uso_IA.pdf`.

> **Borrador inicial.** Las filas describen lo que ocurrió en las conversaciones con la herramienta. Las celdas con
> `[PEDRO: …]` deben completarse con lo que verificaste tú mismo antes de entregar. Borra esta nota al terminar.

| Fecha | Integrante | Herramienta | Para qué se usó | Qué se aceptó, modificó o descartó | Cómo se verificó |
|---|---|---|---|---|---|
| 2026-11-07 | Pedro Castro | Claude (Sonnet 5.5, claude.ai) | Entender las pautas del proyecto final y elegir problema y ruta | Aceptó continuar FreshCheck-Agent en la ruta F (visión por computador). Descartó empezar con un problema nuevo | Claude revisó el repositorio anterior (simulador con imágenes sintéticas y gradient boosting) para comprobar que el alcance crece de forma evidente. [PEDRO: contrastar con las secciones «Rutas admitidas» y «Continuidad» de las pautas] |
| 2026-11-07 | Pedro Castro | Claude (Sonnet 5.5, claude.ai) | Redactar la propuesta del Hito 1: PEAS, datos, línea base, métricas y umbrales | Aceptó la estructura, la línea base (ResNet-50 congelado + cabeza lineal) y las métricas. Modificó: usar solo imágenes de internet en lugar de fotos propias, con umbral menor para ese subconjunto (macro-F1 0,80 → 0,70) y 200 imágenes por clase en lugar de 100. Descartó: el enfoque mixto con 30–50 fotos propias de Cartagena como prueba | Claude consultó la página del dataset (categorías fresh/rotten/formalin, 10 154 originales, 81 232 con aumentos) y detectó que FruitVision no tiene la clase «en maduración», que el diseño anterior sí usaba. [PEDRO: confirmar la licencia en la página de Mendeley] |
| 2026-11-07 | Pedro Castro | Claude (Sonnet 5.5, claude.ai) | Generar el código del repositorio v2 (`src/`, `config/umbrales_v2.json`, `tests/`, README) y el borrador del Hito 2 | [PEDRO: anotar qué aceptaste, qué cambiaste y qué descartaste después de leer el código] | Claude ejecutó `python -m tests.test_pipeline` con datos sintéticos (manifiesto, licencias, duplicados, partición sin fuga, línea base). **No** se ejecutó la extracción con ResNet-50 ni datos reales. [PEDRO: ejecutar `python -m src.run_baseline` y registrar el resultado y la versión de Python/torch usada] |

## Reflexión: un caso en que la IA propuso algo inadecuado (borrador para el informe)

Al proponer fotos propias, la IA fijó un mínimo de 100 imágenes por clase. Al revisar la partición 70/15/15 se vio que
dejaría unas 15 imágenes por clase en prueba, muy pocas para que el macro-F1 fuera estable; se subió el mínimo a 200.
[PEDRO: añade aquí los casos reales que detectes tú al revisar el código y los resultados; el informe necesita al menos uno
con la forma «qué propuso la IA, cómo lo detecté y qué hice».]
