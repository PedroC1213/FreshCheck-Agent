# FreshCheck-Agent

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

## Estructura

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

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Probado con Python 3.12 (numpy 2.4, scikit-learn 1.8, OpenCV 4.13, pandas 3.0, Streamlit 1.66).

## Ejecución

```bash
python scripts/train.py            # entrena el modelo (~90 s) -> models/freshcheck_model.joblib
python scripts/run_tests.py        # pruebas de la Fase 3 (~6 min) -> logs/pruebas/
python scripts/run_extras.py       # ablación y reentrenamiento (experimentos complementarios)
python scripts/decision_matrix.py  # matriz de decisión y sensibilidad (Fase 1)
streamlit run app.py               # demostración interactiva
```

`python scripts/run_tests.py --rapido` ejecuta una prueba de humo de ~20 s.

## Reproducibilidad

Todas las semillas son fijas y disjuntas: entrenamiento `1000`, validación `2000`, pruebas
`9000+`. Con las mismas versiones de librerías, `run_tests.py` reproduce los resultados del
informe (las latencias dependen de la máquina).

## Datos reales

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

## Resultados (resumen)

Ver `docs/Informe_FreshCheck_Agent.pdf`, sección 9.7. Con las semillas y versiones indicadas: exactitud nominal 98,8 %, límite 96,1 %,
reducción de defectuosas 98,7 %, latencia p95 14,8 ms y consistencia del reporte 100 %; la robustez ante fallas (M5) quedó en 98,99 %
(umbral 100 %) por frutas en contacto no detectadas. Los resultados son sobre un simulador.

## Despliegue (Streamlit Community Cloud)

1. Suba este repositorio a GitHub (incluya `models/freshcheck_model.joblib`).
2. En share.streamlit.io elija el repositorio, rama principal y archivo `app.py`.
3. Casos de prueba sugeridos: escenario `N1_nominal`; escenario `L1_luz_muy_baja`; en la pestaña
   *Caso de falla*, `F01_sin_dato` y `F10_rafaga`.
