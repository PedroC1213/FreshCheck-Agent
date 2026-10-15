# Guion del video de evidencia (máximo 20 minutos)

La guía pide cámara y voz de todos los integrantes; al ser una entrega individual, basta con la cámara y la voz de quien presenta
(conviene mencionarlo al inicio). Publique el video como enlace (YouTube no listado, OneDrive o Drive con acceso habilitado) y
copie el índice con minutos en la descripción.

## Índice con minutos (pegar al inicio de la descripción)

| Minuto | Contenido |
|---|---|
| 00:00 | Introducción: problema, agente elegido y alcance (entorno simulado) |
| 01:30 | Fase 1: matriz de decisión, resultado 3,95 vs 3,85 y sensibilidad |
| 04:00 | Arquitectura y ciclo percepción → decisión → acción (`agent.py`, `perception.py`) |
| 07:00 | Ejecución desde cero: clonar, `pip install -r requirements.txt`, `python scripts/train.py` |
| 09:30 | Demo del escenario típico en Streamlit (`N1_nominal`) y reporte por proveedor |
| 11:30 | Demo de falla esperada (`F01_sin_dato` y `F10_rafaga`) |
| 13:00 | Ejecución de las pruebas (`python scripts/run_tests.py`) mientras se explica el análisis |
| 17:00 | Resultados: tabla de la sección 9.7, M5 incumplida y su causa |
| 19:00 | Limitaciones, trabajo futuro y cierre |

## Qué debe verse en pantalla

1. **El agente ejecutándose desde cero**: terminal limpia, entorno virtual recién creado, instalación y entrenamiento
   (~90 s; puede acelerar esa parte en la edición).
2. **Un escenario típico y uno de falla esperada**, en la app (`streamlit run app.py`).
3. **La ejecución de las pruebas y la obtención de los valores de la tabla**: `python scripts/run_tests.py` tarda unos 5–6
   minutos; inícielo en el minuto 13 y use ese tiempo para explicar el análisis. Al terminar, muestre
   `logs/pruebas/resultados_resumen.json` y compare con la tabla del informe.
4. Los archivos de log abiertos (`log_clasificacion.csv`, `log_fallas.csv`) para mostrar percepción, decisión y acción con marca de tiempo.

## Frases clave para no olvidar

- El entorno es un simulador permitido por la guía; los porcentajes no equivalen al desempeño con fruta real.
- La medida M5 no se cumplió (98,99 %) y por qué: frutas en contacto no detectadas.
- El agente en ejecución no usa ningún LLM.
- El estado interno (`ModeloDelMundo`) no mejoró la exactitud (ablación E1); su valor está en la gestión de fallas.
