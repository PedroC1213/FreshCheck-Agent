[README.md](https://github.com/user-attachments/files/34523488/README.md)
# FreshCheck-Agent

Clasificación del estado de frutas (**fresca**, **en maduración**, **dañada**) con visión por computador.
Este repositorio tiene dos etapas del curso de Inteligencia Artificial (Universidad Tecnológica de Bolívar):

- **v2 — Proyecto final (ruta F, visión por computador):** CNN con transferencia de aprendizaje sobre imágenes reales. Trabajo individual.
- **v1 — Actividad anterior:** agente sobre un simulador con imágenes sintéticas (sección al final).

## v2 — Proyecto final

**Pregunta:** ¿cuánto mejora el ajuste fino de una CNN preentrenada la clasificación del estado de una fruta frente a un extractor congelado, y cuánto se degrada al pasar de un dataset público a imágenes de internet?

| Elemento | Detalle |
|---|---|
| Línea base (Hito 2) | ResNet-50 (ImageNet) congelado + cabeza lineal; piso: clase mayoritaria |
| Comparación (Hito 3) | ResNet-50 con ajuste fino y EfficientNet-B0 con ajuste fino |
| Métricas y umbrales | `config/umbrales_v2.json` (declarados en el Hito 1, antes de experimentar) |
| Evaluación en Hito 2 | Solo **validación**; la partición de prueba no se toca hasta la evaluación final |
| Semillas | partición 42; modelo 42, 43, 44 |

### Datos

Detalle, licencias y criterio de etiquetado en [`data/README.md`](data/README.md). Estructura esperada:

```
data/raw/
  fruitvision/{fresca,danada}/        <fruta>__<id>.jpg   (se genera con src.import_fruitvision)
  internet/{fresca,maduracion,danada}/ <fruta>__<id>.jpg  (se llena a mano; ver data/sources_internet.csv)
```

Las imágenes no se suben al repositorio (`data/raw/` está en `.gitignore`); sí se suben `data/manifest.csv`,
`data/splits.csv` y `data/sources_internet.csv` (URL, autor y licencia de cada imagen).

### Instalación y ejecución (4 comandos)

```bash
python -m venv .venv && source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.import_fruitvision --origen RUTA/FruitVision --excluir-regex "aug"   # una vez; use --simular primero
python -m src.run_baseline                                   # manifiesto -> partición -> features -> línea base
```

Antes del último comando deben estar las imágenes de internet en `data/raw/internet/` y su fila en
`data/sources_internet.csv`. La primera ejecución descarga los pesos de ResNet-50 (torchvision).

Salidas en `experiments/baseline/`: `resultados_val.json`, `grid_val.csv`, `errores_val.csv`,
`matriz_confusion_val.png`, `entorno.json` (versiones y hash de la partición).

Prueba de humo con datos sintéticos (sin descargas ni torch): `python -m tests.test_pipeline`.

### Estructura v2

```
src/            config, make_manifest, split, features, train_baseline, run_baseline, import_fruitvision, metricas
tests/          prueba de humo del pipeline (datos sintéticos)
config/umbrales_v2.json   métricas y umbrales del Hito 1
data/           README, sources_internet.csv, manifest.csv, splits.csv (raw/ no se versiona)
experiments/    resultados que respaldan cada tabla del informe (features/ no se versiona)
notebooks/      exploración
docs/bitacora_ia.md       bitácora humano-IA del proyecto final
```

### Control de fuga de datos

La partición 70/15/15 se hace sobre imágenes originales, estratificada por clase y fuente, y por **grupos de
duplicados** (dHash, distancia de Hamming ≤ 2): ninguna imagen casi idéntica queda en dos particiones.
Los aumentos de datos, cuando se usen (Hito 3), se aplican solo sobre entrenamiento.

---

## FreshCheck-Agent v1 (actividad anterior)

Agente inteligente de inspección de calidad de frutas en una línea de empaque piloto
(curso de Inteligencia Artificial, Universidad Tecnológica de Bolívar).

El agente recibe la imagen de cada fruta que pasa bajo la cámara, infiere su estado
(**fresca**, **en maduración** o **dañada**) y ordena a un desviador enviar las unidades dañadas
a la línea de descarte. Es un **agente reflejo basado en modelo** (Russell & Norvig): mantiene un
estado interno (banda de referencia, salud de la cámara, velocidad estimada, contadores por lote)
y usa un modelo entrenado para inferir lo que no se observa directamente.

> **Alcance.** El entorno es un **simulador** (permitido por la guía de la actividad): genera
> imágenes sintéticas de frutas con iluminación, ruido, posición y variedad aleatorios, una banda
> con dinámica (velocidad con deriva, plazo de decisión) y fallas de percepción. Los resultados
> **no** equivalen al desempeño sobre fruta real. Para usar imágenes reales (FruitVision o
> fotografías propias) vea la sección *Datos reales*.

### Estructura

```
freshcheck/
  simulator.py    Entorno: línea de empaque simulada (imágenes, banda, lotes, fallas)
  perception.py   Validación, normalización de iluminación, segmentación y características
  model.py        Clasificador (gradient boosting) y calibración de la banda
  agent.py        FreshCheckAgent: percibir() -> decidir() -> actuar() + ModeloDelMundo
  actuator.py     Interfaz del desviador (InterfazDesviador) y DesviadorSimulado
  reports.py      Reportes por lote y proveedor
  learning.py     Retroalimentación del personal de calidad y reentrenamiento
  evaluation.py   Ejecución de escenarios y cálculo de las medidas M1-M6
  scenarios.py    Escenarios de prueba (nominales, límite y de falla)
scripts/          train.py, run_tests.py, run_extras.py, decision_matrix.py, make_diagrams.py
config/umbrales.json   Umbrales de aceptación (fijados antes de ejecutar las pruebas)
logs/pruebas/     Logs CSV/JSON de las pruebas oficiales
app.py            Interfaz Streamlit (demostración)
docs/             Informe PDF, bitácora de uso de IA, guion del video, diagramas y figuras
analysis/         Matriz de decisión y sensibilidad (Fase 1)
```

### Instalación

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Probado con Python 3.12 (numpy 2.4, scikit-learn 1.8, OpenCV 4.13, pandas 3.0, Streamlit 1.66).

### Ejecución

```bash
python scripts/train.py            # entrena el modelo (~90 s) -> models/freshcheck_model.joblib
python scripts/run_tests.py        # pruebas de la Fase 3 (~6 min) -> logs/pruebas/
python scripts/run_extras.py       # ablación y reentrenamiento (experimentos complementarios)
python scripts/decision_matrix.py  # matriz de decisión y sensibilidad (Fase 1)
streamlit run app.py               # demostración interactiva
```

`python scripts/run_tests.py --rapido` ejecuta una prueba de humo de ~20 s.

### Reproducibilidad

Todas las semillas son fijas y disjuntas: entrenamiento `1000`, validación `2000`, pruebas
`9000+`. Con las mismas versiones de librerías, `run_tests.py` reproduce los resultados del
informe (las latencias dependen de la máquina).

### Datos reales

```
datos/
  fresca/       *.jpg
  maduracion/   *.jpg
  danada/       *.jpg
```

```bash
python scripts/train.py --real-data datos
```

La segmentación asume que el borde de la imagen muestra un fondo (banda) de color uniforme. Con
fondos complejos habría que sustituir `segmentar_fruta` o usar una CNN (ver *Limitaciones* del
informe). El modelo incluido en `models/` fue entrenado solo con imágenes sintéticas.

### Resultados (resumen)

Ver `docs/Informe_FreshCheck_Agent.pdf`, sección 9.7. Con las semillas y versiones indicadas: exactitud nominal 98,8 %, límite 96,1 %,
reducción de defectuosas 98,7 %, latencia p95 14,8 ms y consistencia del reporte 100 %; la robustez ante fallas (M5) quedó en 98,99 %
(umbral 100 %) por frutas en contacto no detectadas. Los resultados son sobre un simulador.

### Despliegue (Streamlit Community Cloud)

1. Suba este repositorio a GitHub (incluya `models/freshcheck_model.joblib`).
2. En share.streamlit.io elija el repositorio, rama principal y archivo `app.py`.
3. Casos de prueba sugeridos: escenario `N1_nominal`; escenario `L1_luz_muy_baja`; en la pestaña
   *Caso de falla*, `F01_sin_dato` y `F10_rafaga`.
