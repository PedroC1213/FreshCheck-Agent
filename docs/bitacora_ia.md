# Bitácora humano-IA — proyecto final (FreshCheck-Agent v2)

Registro de cada uso relevante de IA generativa. La bitácora de la actividad anterior está en `docs/Bitacora_uso_IA.pdf`.

> **Borrador inicial.** Las filas describen lo que ocurrió en las conversaciones con la herramienta. Las celdas con
> `[PEDRO: …]` deben completarse con lo que verificaste tú mismo antes de entregar. Borra esta nota al terminar.

| Fecha | Integrante | Herramienta | Para qué se usó | Qué se aceptó, modificó o descartó | Cómo se verificó |
|---|---|---|---|---|---|
| 2026-11-07 | Pedro Castro | Claude (Sonnet 5.5, claude.ai) | Entender las pautas del proyecto final y elegir problema y ruta | Aceptó continuar FreshCheck-Agent en la ruta F (visión por computador). Descartó empezar con un problema nuevo | Claude revisó el repositorio anterior (simulador con imágenes sintéticas y gradient boosting) para comprobar que el alcance crece de forma evidente. [PEDRO: contrastar con las secciones «Rutas admitidas» y «Continuidad» de las pautas] |
| 2026-11-07 | Pedro Castro | Claude (Sonnet 5.5, claude.ai) | Redactar la propuesta del Hito 1: PEAS, datos, línea base, métricas y umbrales | Aceptó la estructura, la línea base (ResNet-50 congelado + cabeza lineal) y las métricas. Modificó: imágenes de internet en lugar de fotos propias, con umbral menor para ese subconjunto. Descartó: el enfoque mixto con fotos propias de Cartagena como prueba | Claude consultó la descripción del dataset (categorías fresh/rotten/formalin, 10 154 originales, 81 232 con aumentos) y detectó que FruitVision no tiene la clase «en maduración». [PEDRO: contrastar con la página de Mendeley] |
| 2026-11-07 | Pedro Castro | Claude (Sonnet 5.5, claude.ai) | Generar el código del repositorio v2 (`src/`, `config/umbrales_v2.json`, `tests/`, README) | [PEDRO: anotar qué aceptaste, qué cambiaste y qué descartaste después de leer el código] | Claude ejecutó `python -m tests.test_pipeline` con datos sintéticos. **No** se ejecutó la extracción con ResNet-50 ni datos reales. [PEDRO: ejecutar `python -m src.run_baseline` y registrar el resultado y la versión de Python/torch usada] |
| 2026-11-08 | Pedro Castro | Claude (Sonnet 5.5, claude.ai) | Revisar el estado del repositorio en GitHub y corregir errores de subida (archivos sin extensión, README roto, archivos faltantes) | Aceptó las correcciones y las subió con `git`. Descartó subir archivos arrastrándolos a la web | Claude clonó el repositorio, comparó cada archivo con el original (idénticos salvo el README) y simuló las correcciones sobre un clon con la prueba sintética. [PEDRO: confirmar que `python -m tests.test_pipeline` pasa en tu equipo] |
| 2026-11-08 | Pedro Castro | Claude (Sonnet 5.5, claude.ai) | Verificar la licencia y la estructura de FruitVision y rediseñar el alcance para usar **solo FruitVision** (sin imágenes de internet) | Pidió quitar las imágenes de internet. Claude advirtió que eso elimina la clase «en maduración» y propuso: dos clases (fresca y dañada), medir la generalización a una fruta no vista y ajustar los umbrales. [PEDRO: anotar si aceptaste todo o qué modificaste] | Claude leyó la página de Mendeley y el README del repositorio de los autores (tres celulares, fondos simples, 512×512, licencias). Los cambios y su justificación están en `docs/cambios_respecto_hito1.md`. [PEDRO: informar al docente y anotar su respuesta] |

## Reflexión: un caso en que la IA propuso algo inadecuado (borrador para el informe)

En el Hito 1 y en `data/README.md` la IA anotó la licencia de FruitVision como MIT, porque así figura en el repositorio de GitHub
de uno de los autores, y dijo que se confirmaría en Mendeley. Al revisar la página oficial del dataset en Mendeley, la licencia
era **CC BY-NC-ND 4.0** (uso no comercial y sin obras derivadas), más restrictiva. Se corrigió la documentación, se mantuvo la
regla de no redistribuir las imágenes y se aplicó la licencia más restrictiva. La lección es verificar siempre en la fuente
oficial del dataset, no en un repositorio secundario.
[PEDRO: añade los casos reales que detectes tú al revisar el código y los resultados; el informe necesita al menos uno con la
forma «qué propuso la IA, cómo lo detecté y qué hice».]
