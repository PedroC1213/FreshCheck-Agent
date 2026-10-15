"""Genera docs/Bitacora_uso_IA.pdf."""
import sys
from pathlib import Path

from reportlab.lib.units import cm
from reportlab.platypus import Spacer

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from informe_lib import B, P, a_horizontal, crear_documento, tabla   # noqa: E402

el = []
add = el.append

add(P("Bitácora de uso de inteligencia artificial", "h1"))
add(P("FreshCheck-Agent — actividad de implementación de un agente inteligente (curso de Inteligencia Artificial, UTB). "
      "Estudiante: Pedro José Castro Blanquicett (T00067714). Entrega individual."))
add(P("<b>Herramienta:</b> Claude (Anthropic), modelo Claude Sonnet 5.5, en la interfaz web claude.ai, dentro de un proyecto del curso. "
      "<b>Alcance:</b> apoyo en el diseño, la implementación, las pruebas y la redacción. El agente en ejecución no utiliza ningún modelo de lenguaje.", "nota"))

add(P("1. Registro de usos", "h2"))
filas = [
    ["Fecha", "Actividad", "Qué se pidió a la IA", "Qué produjo la IA", "Verificación y correcciones"],
    ["28/08/2026", "Diseño inicial (Tarea 2)",
     "Completar la plantilla de diseño de agente; investigar fuentes; adaptar el documento al contexto académico (sin cámara ni banda reales) y ampliar las justificaciones.",
     "Tres versiones del documento (completa, concisa y adaptada), con tabla de ambientes y componentes IBM.",
     "Se contrastó con las lecturas del curso. En la revisión del 14/10 se detectó que la cifra «87 % a más de 98 %» atribuida a Ismail y Malik no correspondía a ese estudio; se corrigió."],
    ["14/10/2026", "Fase 1: matriz de decisión",
     "Comparar FreshCheck-Agent con RetainIQ-Agent con los criterios y pesos sugeridos por la guía, con evidencia por puntaje y análisis de sensibilidad.",
     "Puntajes con evidencia, cálculo reproducible (scripts/decision_matrix.py) y justificación.",
     "La IA advirtió que el margen es estrecho (3,95 frente a 3,85) y que la decisión cambia si Complejidad de IA pesa más de 19 %. Los puntajes son juicios y deben ser revisados por el estudiante."],
    ["14/10/2026", "Fase 2: desarrollo",
     "Implementar el agente: simulador de la línea, percepción, modelo, agente, actuador, reportes, aprendizaje e interfaz.",
     "Código del paquete freshcheck, scripts y app.py.",
     "Se ejecutó el entrenamiento (exactitud de validación 98,0 %), se inspeccionaron imágenes sintéticas, se probó la app con una prueba automática y se ajustó la regla de detección de frutas en contacto usando datos de validación, no de prueba."],
    ["14/10/2026", "Fase 3: pruebas",
     "Definir escenarios y umbrales antes de ejecutar; ejecutar las pruebas; reportar lo cumplido y lo incumplido.",
     "Umbrales en config/umbrales.json (hash SHA-256 en el informe), ejecución oficial con semillas 9000+, logs CSV y JSON, experimentos E1 y E2.",
     "Los umbrales no se modificaron tras ejecutar. La medida M5 no se cumplió (98,99 %) y se reportó con su análisis, sin ajustar el umbral."],
    ["14/10/2026", "Documento y diagramas",
     "Generar el informe PDF con tabla de cambios, matriz, sección 9 y pruebas; diagramas de arquitectura y de secuencia.",
     "Informe de 23 páginas con cifras tomadas automáticamente de los logs; diagramas en PNG y fuentes PlantUML.",
     "Las cifras del informe se generan desde los archivos de resultados, de modo que coinciden con los logs. Se revisó visualmente el PDF."],
]
add(tabla(filas, [1.9, 3.0, 6.3, 5.3, 8.8]))

add(P("2. Errores o límites de lo generado por la IA", "h2"))
for t in [
    "Una cifra de la literatura estaba mal atribuida en la versión anterior del diseño (ver 28/08/2026). Las cifras actuales de Ismail y Malik (99,2 % y 98,6 %) y de la visión clásica (89–92 %) provienen de fuentes secundarias consultadas por la IA y deben contrastarse con los artículos originales.",
    "Los resultados de exactitud provienen de imágenes sintéticas; no equivalen al desempeño sobre fruta real.",
    "Los puntajes de la matriz de decisión son juicios razonados, no mediciones.",
    "La referencia del conjunto de datos FruitVision se conservó tal como la entregó la guía; no se verificaron autores ni año.",
]:
    add(B(t))

add(P("3. Revisión del estudiante (completar antes de entregar)", "h2"))
add(tabla([
    ["Punto", "Estado"],
    ["Revisé y ajusté, si correspondía, los puntajes de la matriz de decisión", "[ ]"],
    ["Ejecuté el entrenamiento y las pruebas en mi equipo y obtuve resultados equivalentes", "[ ]"],
    ["Entiendo el ciclo percepción → decisión → acción del código y puedo explicarlo", "[ ]"],
    ["Contrasté las cifras de la literatura con los artículos originales", "[ ]"],
    ["Agregué los enlaces del repositorio y de la plataforma o el video", "[ ]"],
    ["Incorporé la retroalimentación del docente en la tabla de cambios", "[ ]"],
], [13.5, 3.0]))

doc = crear_documento(RAIZ / "docs" / "Bitacora_uso_IA.pdf", "Bitácora de uso de IA — FreshCheck-Agent", titulo="Bitácora de uso de IA", inicial="landscape")
# Documento corto: se usa la plantilla horizontal para la tabla de registro.
from reportlab.platypus import NextPageTemplate  # noqa: E402
doc.build(el)
print("Bitácora generada")
