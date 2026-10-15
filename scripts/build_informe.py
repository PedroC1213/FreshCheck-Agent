"""
Genera docs/Informe_FreshCheck_Agent.pdf a partir de los resultados reales (logs/pruebas, analysis).

    python scripts/build_informe.py
"""
import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import pandas as pd
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Spacer

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from informe_lib import (B, P, a_horizontal, a_vertical, crear_documento, figura, num, pct,  # noqa: E402
                         tabla)

LOGS = RAIZ / "logs" / "pruebas"
FIG = RAIZ / "docs" / "figuras"
res = json.loads((LOGS / "resultados_resumen.json").read_text(encoding="utf-8"))
extra = json.loads((LOGS / "experimentos_extra.json").read_text(encoding="utf-8"))
mat = json.loads((RAIZ / "analysis" / "matriz_decision.json").read_text(encoding="utf-8"))
umbr = json.loads((RAIZ / "config" / "umbrales.json").read_text(encoding="utf-8"))
sens = pd.read_csv(RAIZ / "analysis" / "sensibilidad.csv")
mtz = pd.read_csv(RAIZ / "analysis" / "matriz_decision.csv")
run = pd.read_csv(LOGS / "resultados_por_corrida.csv")
fal = pd.read_csv(LOGS / "resultados_fallas.csv")
log_c = pd.read_csv(LOGS / "log_clasificacion.csv", keep_default_na=False)
log_f = pd.read_csv(LOGS / "log_fallas.csv", keep_default_na=False)
entren = json.loads((RAIZ / "models" / "entrenamiento.json").read_text(encoding="utf-8"))
hash_umbrales = hashlib.sha256((RAIZ / "config" / "umbrales.json").read_bytes()).hexdigest()

M = res["medidas"]
PE = res["por_escenario"]


def miles(x):
    return f"{x:,}".replace(",", ".")


def ver(paquete):
    try:
        return metadata.version(paquete)
    except Exception:
        return "-"


# ------------------------------------------------------------------ cifras derivadas del log
vlog = log_c[(log_c["clase_real"] != "") & (log_c["falla_inyectada"] == "")]
n_validas = len(vlog)
solap_falsos = int((vlog["motivo_decision"] == "SOLAPAMIENTO").sum())
d0_total = int((vlog["clase_real"] == "danada").sum())
d1_total = int(((vlog["clase_real"] == "danada") & (vlog["destino"] == "empaque_final")).sum())
l5 = log_c[(log_c["escenario"] == "L5_alta_densidad") & (log_c["accion"] == "DESVIAR")]
l5_ocupado = int((l5["motivo_actuador"] == "DESVIADOR_OCUPADO").sum())
l5_total = len(l5)
conf = pd.crosstab(vlog["clase_real"], vlog["clase_predicha"])
err_fm = int(conf.loc["fresca", "maduracion"]) + int(conf.loc["maduracion", "fresca"])
err_dan_perdida = int(conf.loc["danada", "fresca"]) + int(conf.loc["danada", "maduracion"])
f09 = fal[fal["escenario"] == "F09_solapadas"]
f09_inst, f09_ok = int(f09["instancias"].sum()), int(f09["manejadas"].sum())
fail_det = fal[fal["manejadas"] < fal["instancias"]]

elementos = []
add = elementos.append

# =================================================================== PORTADA
add(Spacer(1, 2.0 * cm))
add(P("UNIVERSIDAD TECNOLÓGICA DE BOLÍVAR", "portada_b"))
add(P("Ingeniería de Sistemas y Computación", "portada"))
add(Spacer(1, 2.2 * cm))
add(P("Implementación de un Agente Inteligente", "portada_t"))
add(Spacer(1, 0.3 * cm))
add(P("FreshCheck-Agent: agente de inspección de calidad de alimentos", "portada_s"))
add(Spacer(1, 0.4 * cm))
add(P("Documento de diseño ajustado, selección justificada, desarrollo y pruebas", "portada_s"))
add(Spacer(1, 2.4 * cm))
add(P("Presentado por:", "portada_b"))
add(P("Pedro José Castro Blanquicett — T00067714", "portada"))
add(P("Entrega individual", "portada"))
add(Spacer(1, 1.2 * cm))
add(P("Curso: Inteligencia Artificial", "portada"))
add(P("Docente: Edwin Puertas, PhD.", "portada"))
add(Spacer(1, 1.6 * cm))
add(P("Cartagena de Indias, 14 de octubre de 2026", "portada"))
add(PageBreak())

# =================================================================== RESUMEN
add(P("Resumen de la entrega", "h1"))
add(P(f"Este documento corrige y amplía el diseño inicial de FreshCheck-Agent y agrega la selección justificada del "
      f"agente (Fase 1), la descripción de su implementación (sección 9) y las pruebas contra las medidas de desempeño "
      f"(Fase 3). El trabajo se realizó de forma individual; por ello se compara FreshCheck-Agent con RetainIQ-Agent, el "
      f"otro agente que el estudiante diseñó durante el semestre."))
add(P(f"<b>Agente implementado:</b> FreshCheck-Agent, agente reflejo basado en modelo, probado sobre un simulador de la línea "
      f"de empaque (imágenes sintéticas, banda con dinámica y fallas de percepción). <b>Resultados:</b> de las seis medidas "
      f"de desempeño (nueve criterios de aceptación), ocho se cumplieron y uno no: la robustez ante fallas de percepción "
      f"(M5) alcanzó {pct(M['M5_robustez_fallas']['media'], 2)} frente a un umbral de 100 %, debido a que "
      f"{f09_inst - f09_ok} de {f09_inst} casos de frutas en contacto no fueron detectados (sección 9.7)."))
add(P("<b>Advertencia de alcance.</b> Los porcentajes de exactitud corresponden al simulador, cuya dificultad se fijó antes de "
      "las pruebas; no son una estimación del desempeño sobre fruta real. La sección 9.8 detalla cómo reentrenar con "
      "imágenes reales (FruitVision o fotografías propias).", "nota"))
add(tabla([
    ["Entregable", "Ubicación"],
    ["Documento de diseño ajustado (este PDF)", "docs/Informe_FreshCheck_Agent.pdf"],
    ["Repositorio con README e instrucciones", "[PEGAR ENLACE DEL REPOSITORIO]"],
    ["Plataforma desplegada o video (máx. 20 min)", "[PEGAR ENLACE DE LA PLATAFORMA O DEL VIDEO]"],
    ["Logs de las pruebas", "logs/pruebas/*.csv y resultados_resumen.json"],
    ["Bitácora de uso de IA", "docs/Bitacora_uso_IA.pdf"],
], [7.5, 9.0]))
add(Spacer(1, 0.2 * cm))

# =================================================================== A. TABLA DE CAMBIOS
elementos.extend(a_horizontal())
add(P("A. Tabla de cambios respecto al diseño inicial", "h1"))
add(P("Los cambios provienen de la revisión del propio diseño al implementarlo. La última fila queda reservada para las "
      "correcciones que el docente haya señalado en su retroalimentación.", "cuerpo_izq"))
cambios = [
    ["Sección", "Qué decía", "Qué dice ahora", "Por qué cambió"],
    ["Portada", "«Docente: Edwin Puerta»; sin programa académico ni indicación de autoría individual.",
     "Docente: Edwin Puertas, PhD.; programa académico; entrega individual.", "Corrección de un error de transcripción y de la identificación de autoría."],
    ["2. Tipo de agente", "El «modelo interno» era únicamente el modelo de visión entrenado; no se definía estado interno.",
     "Se define un estado interno explícito (ModeloDelMundo): banda de referencia, estado de la cámara, velocidad estimada, contadores por lote.",
     "Un agente basado en modelo debe mantener estado; el diseño no lo especificaba y la implementación lo exige."],
    ["3. Ambiente: dinamismo", "«Dinámico (entorno objetivo) / Estático (implementación académica)».",
     "«Dinámico», sin doble clasificación: el simulador reproduce banda con deriva de velocidad, plazo de decisión y recarga del desviador.",
     "La guía exige que el simulador respete las propiedades declaradas; una clasificación doble era ambigua."],
    ["3. Ambiente: determinismo", "Estocástico, justificado de forma general.",
     "Estocástico, con las fuentes de aleatoriedad concretas del simulador (iluminación, ruido, posición, rotación, desenfoque, variedad, velocidad).",
     "Trazabilidad entre diseño e implementación."],
    ["5. Entradas", "El «metadato de origen y etiqueta de referencia» figuraba como entrada del agente.",
     "Se elimina de las entradas; la etiqueta real queda oculta y se usa solo para evaluar.",
     "Era incoherente con un ambiente parcialmente observable: en operación el agente no conoce la etiqueta."],
    ["5. Entradas", "Sin señal de disparo ni velocidad de banda (omitidas en la versión adaptada).",
     "Se incluyen disparo, velocidad de banda (encoder con ruido), lote/proveedor e imagen.",
     "La velocidad define el plazo de decisión y el lote permite el reporte por proveedor."],
    ["6. Salidas", "Clasificación, señal de desvío y registro.",
     "Se agregan la acción REVISION_MANUAL (percepción inválida o frutas en contacto) y la alerta de cámara en falla.",
     "Las condiciones excepcionales de la sección 4 necesitaban una acción segura; sustenta la medida M5."],
    ["7. Medidas de desempeño", "Viñetas SMART sin fórmulas; «reducción de 30 %» sin línea base; plazo ligado a «tres meses».",
     "Seis medidas (M1 a M6) con fórmula, umbral, estratos y línea base definida (D0 = unidades dañadas sin agente).",
     "La Fase 3 exige fórmula y umbral fijados antes de las pruebas."],
    ["7. Alcanzable", "«Exactitudes de 87 % a más de 98 %» atribuidas a Ismail y Malik (2022).",
     "Ismail y Malik reportan 99,2 % (manzana) y 98,6 % (banano) con EfficientNet; la visión clásica reporta cerca de 89–92 %.",
     "La cifra de 87 % no corresponde a ese estudio; además, la comparación respalda el umbral y la elección del modelo."],
    ["2 y 8. Modelo de visión", "«Aprendizaje profundo / modelo de visión por computador» sin precisar el método.",
     "Características de visión clásica y gradient boosting; la CNN queda como trabajo futuro (sección 9.6).",
     "Hardware de un núcleo de CPU, trazabilidad y tiempo de desarrollo disponible."],
    ["1, 5 y 7. Fuente de datos", "Entrenamiento y evaluación con FruitVision, fotografías propias e imágenes de internet.",
     "Entrenamiento y pruebas sobre imágenes sintéticas del simulador; FruitVision y fotos reales quedan como trabajo futuro (script --real-data).",
     "La guía permite un simulador; los datos reales no se integraron y se evita presentar un resultado sintético como real."],
    ["8. Componentes IBM", "Descripción conceptual de los siete componentes.",
     "Cada componente se asigna a un módulo del código y se indica si quedó implementado, parcial o sin implementar (sección 9.1).",
     "Requisito de la actividad."],
    ["Referencias", "Russell y Norvig «(s.f.)»; sin referencias técnicas de la implementación.",
     "Russell y Norvig (2021, 4.ª ed.); se agregan scikit-learn, OpenCV y las fuentes de la matriz de decisión.",
     "Cita incompleta y fuentes nuevas."],
    ["Conclusiones", "Sección 9.", "Sección 10, con los resultados reales de las pruebas.", "Numeración exigida por la actividad (sección 9 = Desarrollo)."],
    ["[Retroalimentación del docente]", "[Completar]", "[Completar]", "[Completar con las correcciones señaladas por el docente]"],
]
add(tabla(cambios, [3.3, 6.4, 7.6, 6.4]))

# =================================================================== B. FASE 1
elementos.extend(a_vertical())
add(P("B. Fase 1 — Selección justificada del agente", "h1"))
add(P("El estudiante diseñó dos agentes durante el semestre: <b>Agente A — FreshCheck-Agent</b> (clasificación de frutas en una línea de "
      "empaque) y <b>Agente B — RetainIQ-Agent</b> (predicción de rotación de empleados). La elección se hizo con una matriz de "
      "decisión ponderada, con pesos definidos antes de puntuar."))
add(P("B.1 Criterios y pesos", "h2"))
add(P("Se adoptan los seis criterios y los pesos sugeridos por la guía (suman 100 %), porque se ajustan a la rúbrica de la actividad:"))
add(tabla([
    ["Criterio", "Peso", "Justificación del peso"],
    ["Viabilidad técnica", "25 %", "Sin un agente construible no hay nada que demostrar; la implementación vale 25 de 100 puntos de la rúbrica."],
    ["Disponibilidad de datos o entorno", "20 %", "Sin datos o un entorno simulable no hay percepciones ni pruebas posibles."],
    ["Medibilidad del desempeño", "20 %", "La Fase 3 exige probar cada medida con fórmula y umbral; las pruebas valen 20 puntos."],
    ["Complejidad de IA", "15 %", "Interesa que el agente exija razonamiento real, pero la rúbrica evalúa la coherencia con el tipo declarado, no el tipo más alto."],
    ["Costo y recursos", "10 %", "Restricción práctica; ambos agentes se pueden construir con herramientas gratuitas."],
    ["Valor del problema", "10 %", "Pertinencia del problema para un usuario identificable; no condiciona la viabilidad de la entrega."],
], [4.6, 1.5, 10.4], negrita_primera=True))

elementos.extend(a_horizontal())
add(P("B.2 Matriz de decisión", "h1"))
filas = [["Criterio", "Peso (w)", "A: puntaje (1–5)", "A: evidencia (FreshCheck-Agent)", "B: puntaje (1–5)", "B: evidencia (RetainIQ-Agent)"]]
import importlib.util
spec = importlib.util.spec_from_file_location("dm", RAIZ / "scripts" / "decision_matrix.py")
dm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dm)
for n_, w_, pa_, ea_, pb_, eb_ in dm.CRITERIOS:
    filas.append([n_, pct(w_, 0), str(pa_), ea_, str(pb_), eb_])
filas.append(["<b>Puntaje ponderado S</b>", "<b>100 %</b>", f"<b>{num(mat['S_A'])}</b>", "S(A) = Σ wᵢ·pᵢ", f"<b>{num(mat['S_B'])}</b>", "S(B) = Σ wᵢ·pᵢ"])
add(tabla(filas, [3.6, 1.4, 1.6, 7.4, 1.6, 7.9]))
add(P("B.3 Cálculo", "h2"))
calc_a = " + ".join(f"{num(r.peso, 2)}×{int(r.A_puntaje)}" for r in mtz.itertuples())
calc_b = " + ".join(f"{num(r.peso, 2)}×{int(r.B_puntaje)}" for r in mtz.itertuples())
add(P(f"S(A) = {calc_a} = <b>{num(mat['S_A'])}</b><br/>S(B) = {calc_b} = <b>{num(mat['S_B'])}</b>   (puntaje máximo posible: 5)"))

elementos.extend(a_vertical())
add(P("B.4 Análisis de sensibilidad", "h2"))
infl = mat["criterio_mas_influyente"]
add(P(f"El criterio más influyente es <b>{infl}</b> (mayor contribución absoluta a la diferencia, |w·Δp| = 0,30). Se varió el peso de cada "
      f"criterio en ±10 puntos y se reescalaron proporcionalmente los demás pesos para que sumen 100 %."))
filas = [["Criterio", "Cambio de peso", "Peso nuevo", "S(A)", "S(B)", "Elegido"]]
for r in sens.itertuples():
    filas.append([r.criterio, r.cambio_peso.replace("%", " puntos").replace("+", "+").replace("-", "−"), pct(r.peso_nuevo, 0),
                  num(r.S_A), num(r.S_B), r.elegido])
add(tabla(filas, [5.6, 2.6, 2.0, 1.8, 1.8, 2.0]))
add(Spacer(1, 0.2 * cm))
add(P(f"En el criterio más influyente, el punto de equilibrio está en un peso de {num(100 * mat['peso_equilibrio'], 1)} %: si Complejidad de IA "
      f"pesara más que eso, RetainIQ-Agent superaría a FreshCheck-Agent."))

add(P("B.5 Justificación de la elección", "h2"))
add(P(f"<b>Agente elegido: FreshCheck-Agent (A).</b> Puntajes finales: S(A) = {num(mat['S_A'])} y S(B) = {num(mat['S_B'])} sobre 5. "
      f"El margen es estrecho (0,10) y se reconoce como tal: la decisión no es robusta ante cualquier cambio de pesos."))
add(P("<b>Sensibilidad.</b> La decisión se mantiene al variar ±10 puntos Viabilidad, Costo y Valor, y al subir Disponibilidad o Medibilidad. "
      "Cambia a favor de B en tres casos: Complejidad con +10 puntos (S(A) = 3,84 frente a S(B) = 3,99) y Disponibilidad o Medibilidad con −10 puntos "
      "(3,94 frente a 3,96). Cambia, por tanto, en el criterio más influyente."))
add(P("<b>Por qué se mantiene la elección.</b> (1) La ventaja de B depende de que la complejidad del tipo de agente pese cerca del 19 %, casi tanto como la "
      "medibilidad o la disponibilidad (20 %), lo que contradice el énfasis de la rúbrica en demostrar con pruebas el cumplimiento de las medidas. "
      "(2) La medida de B (precisión ≥ 70 % a tres meses con evaluación trimestral) no se puede verificar con el dataset disponible, que no tiene "
      "dimensión temporal: exigiría reformularla y simular las trayectorias de los empleados, de modo que sus pruebas serían en buena parte circulares. "
      "(3) La medida de A es verificable con etiquetas conocidas y una línea base definible."))
add(P("<b>Qué se pierde al descartar RetainIQ-Agent.</b> Un agente de mayor complejidad (aprendizaje con elemento de desempeño basado en utilidad y "
      "memoria de trayectorias), un dataset tabular real disponible de inmediato (IBM HR Analytics), un menor esfuerzo de percepción (sin visión por "
      "computador ni simulador de imágenes) y la oportunidad de trabajar la dimensión ética del tratamiento de datos de personal."))

# =================================================================== 1-8 DISEÑO
add(PageBreak())
add(P("1. Introducción y planteamiento del problema", "h1"))
add(P("La inspección manual de frutas en líneas de empaque es lenta, subjetiva y propensa a error humano, lo que permite que unidades dañadas lleguen "
      "al empaque final y generen desperdicio, devoluciones y pérdidas económicas. Un agente inteligente basado en visión por computador ejecuta de forma "
      "continua y consistente el ciclo de percepción, clasificación y acción sobre cada fruta, tal como lo muestran sistemas de inspección visual en "
      "tiempo real reportados en la literatura (Ismail & Malik, 2022). FreshCheck-Agent clasifica frutas según su estado (fresca, en maduración o dañada) "
      "en una línea de empaque piloto, con una exactitud mínima de 95 % y una reducción de al menos 30 % en las unidades defectuosas que llegan al empaque final."))
add(P("<b>Alcance académico.</b> El diseño apunta a una línea piloto real, pero este trabajo de curso no dispone de una instalación industrial. Por ello el "
      "entorno se implementó como un simulador que genera imágenes sintéticas y reproduce las propiedades del ambiente declaradas en la sección 3. "
      "FruitVision y las fotografías propias o de internet quedan como la ruta para validar el agente con fruta real (sección 9.8).", "nota"))

add(P("2. Tipo de agente inteligente", "h1"))
add(P("FreshCheck-Agent es un <b>agente reflejo basado en modelo</b> (Russell & Norvig, 2021). La decisión se justifica por contraste con las demás categorías:"))
add(B("<b>No es un agente reflejo simple</b>, porque este actúa solo sobre el percepto inmediato mediante reglas fijas. Distinguir una maduración normal de un "
      "daño real exige un modelo entrenado, y operar con fallas de cámara exige recordar lo ocurrido en cuadros anteriores."))
add(B("<b>Sí es basado en modelo</b>, porque mantiene un estado interno que se actualiza con cada percepción (<i>ModeloDelMundo</i>): color de la banda "
      "suavizado en el tiempo, estado de la cámara (OK, degradada, en falla) con histéresis de recuperación, velocidad estimada de la banda y contadores por "
      "lote. Ese estado condiciona la decisión: con la cámara en falla, el agente deriva a inspección manual aunque la imagen siguiente parezca válida. "
      "Además, el modelo entrenado infiere el estado real de la fruta, que no es observable directamente en el percepto."))
add(B("<b>No es basado en metas</b>, porque no busca ni planifica secuencias de acciones: cada unidad requiere una decisión de un solo paso."))
add(B("<b>No es basado en utilidad</b>, porque la acción se deriva directamente de la categoría inferida; no se comparan acciones alternativas con una "
      "función de utilidad."))
add(P("El reentrenamiento periódico con muestras corregidas por el personal de calidad añade un elemento de agente de aprendizaje, pero desacoplado de la "
      "operación en línea, que sigue el esquema reflejo basado en modelo. La sección 9.7 incluye una ablación que muestra qué aporta el estado interno y qué no."))

add(P("3. Tipos de ambiente", "h1"))
amb = [
    ["Dimensión", "Clasificación", "Justificación (y alternativa descartada)"],
    ["Observabilidad", "Parcialmente observable",
     "La imagen solo muestra la apariencia externa; el estado real (firmeza, descomposición interna) debe inferirse, y la etiqueta verdadera no viaja en el percepto. Se descarta «totalmente observable»."],
    ["Determinismo", "Estocástico",
     "El simulador introduce aleatoriedad en iluminación (deriva y parpadeo), ruido del sensor, posición, rotación y escala de la fruta, desenfoque, variedad y velocidad de la banda; una misma fruta no produce siempre la misma clasificación. Se descarta «determinista»."],
    ["Episodicidad", "Episódico",
     "Cada imagen se clasifica de forma independiente; la decisión sobre una unidad no depende de las anteriores ni afecta a las siguientes. El estado interno solo gestiona la cámara y los contadores, no condiciona la clasificación de la fruta. Se descarta «secuencial»."],
    ["Dinamismo", "Dinámico",
     "La banda avanza de forma continua, su velocidad tiene deriva aleatoria y el agente dispone de un plazo (distancia ÷ velocidad × 0,8) para decidir antes de que la unidad llegue al desviador; si decide tarde, la unidad pasa de largo. El simulador implementa estas propiedades en tiempo simulado. Se descartan «estático» y «semidinámico»."],
    ["Discretización", "Continuo (acción discreta)",
     "Los valores de píxel y la velocidad varían de forma continua; las salidas son un conjunto finito de tres clases y cuatro acciones. Se descarta «discreto» para la entrada."],
    ["Número de agentes", "Agente único",
     "Un solo agente percibe, clasifica y decide; el desviador y el personal de calidad no son agentes racionales independientes dentro del ciclo de decisión. Se descarta «multiagente»."],
]
add(tabla(amb, [3.0, 3.2, 10.3], negrita_primera=True))

add(P("4. Descripción del entorno", "h1"))
add(P("<b>Actores:</b> unidades de fruta de distintos lotes y proveedores, la cámara sobre la banda, el sistema de control de la banda con su desviador, el "
      "personal de calidad y los proveedores. <b>Restricciones:</b> plazo de decisión acotado por la velocidad de la banda, tiempo de recarga del desviador, "
      "iluminación variable y requisitos de higiene propios de la industria alimentaria. <b>Condiciones normales:</b> fruta aislada, centrada y bien iluminada. "
      "<b>Condiciones excepcionales:</b> iluminación fuera de rango, ruido o desenfoque, frutas en contacto, banda vacía, cámara sin dato, imagen corrupta, "
      "alta densidad de unidades y variedades no vistas en el entrenamiento."))

add(P("5. Entradas y sensores", "h1"))
add(tabla([
    ["Entrada", "Sensor (entorno real → simulado)", "Tipo de señal"],
    ["Imagen RGB de la fruta", "Cámara industrial sobre la banda → imagen sintética generada por el simulador", "Matriz 160×160×3, uint8"],
    ["Señal de disparo", "Sensor fotoeléctrico de presencia → bandera del simulador", "Binaria"],
    ["Velocidad de la banda", "Encoder del sistema de control → lectura con 2 % de ruido", "Numérica continua (m/s)"],
    ["Lote y proveedor", "Lectura de código de barras o QR → campos del lote simulado", "Texto"],
], [4.0, 8.5, 4.0], negrita_primera=True))
add(P("La etiqueta real de cada fruta <b>no</b> es una entrada del agente: el simulador la conserva aparte y solo se usa para calcular las medidas.", "nota"))

add(P("6. Salidas y actuadores", "h1"))
add(tabla([
    ["Salida", "Actuador o canal", "Efecto"],
    ["Clasificación (fresca, maduración, dañada) con probabilidades", "Modelo de visión", "Base de la decisión; no modifica el entorno."],
    ["Acción DESVIAR", "Desviador neumático (InterfazDesviador; en el curso, DesviadorSimulado)", "Redirige la unidad dañada al descarte, si llega a tiempo y el desviador está libre."],
    ["Acción CONTINUAR", "Ninguno", "La unidad sigue hacia el empaque final."],
    ["Acción REVISION_MANUAL", "Derivación a inspección humana", "Se aplica ante percepción inválida, cámara en falla o frutas en contacto."],
    ["Registro y reporte por proveedor", "Log CSV y reportes", "Historial por lote para los reportes de calidad."],
], [5.4, 5.6, 5.5], negrita_primera=True))

add(P("7. Medidas de desempeño", "h1"))
add(P("Las medidas se formulan en formato SMART. Los umbrales se fijaron antes de ejecutar las pruebas (archivo config/umbrales.json, "
      f"SHA-256 {hash_umbrales[:16]}…)."))
add(tabla([
    ["Medida", "Fórmula", "Umbral", "Plazo"],
    ["M1. Exactitud de clasificación", "correctas ÷ unidades con fruta única, sin falla inyectada (REVISION_MANUAL y NINGUNA cuentan como error)",
     "≥ 95 % nominal; ≥ 85 % en condiciones límite", "Fase de pruebas del curso"],
    ["M2. Reducción de unidades defectuosas", "(D0 − D1) ÷ D0; D0 = dañadas que llegarían al empaque sin agente; D1 = dañadas que llegan con agente",
     "≥ 30 %", "Fase de pruebas"],
    ["M3. Descarte indebido", "no dañadas desviadas al descarte ÷ no dañadas", "≤ 5 % nominal; ≤ 10 % límite", "Fase de pruebas"],
    ["M4. Latencia de decisión", "percentil 95 de la latencia percepción + decisión; número de decisiones fuera de plazo", "p95 ≤ 200 ms; 0 tardías", "Fase de pruebas"],
    ["M5. Robustez ante fallas", "inyecciones manejadas con la acción y el código esperados, sin excepción y con registro ÷ inyecciones", "100 %", "Fase de pruebas"],
    ["M6. Consistencia del reporte por proveedor", "celdas iguales entre el reporte del agente y el recalculado desde el log ÷ celdas totales", "100 %", "Fase de pruebas"],
], [3.8, 6.9, 3.6, 2.2], negrita_primera=True))
add(P("<b>Alcanzable.</b> Ismail y Malik (2022) reportan exactitudes de 99,2 % para manzana y 98,6 % para banano con EfficientNet. Estudios con "
      "procesamiento de imagen clásico en frutas reales reportan alrededor de 89–92 %, frente a 94–95 % con redes convolucionales (Frontiers of Agricultural "
      "Science and Engineering, 2023). Por eso el umbral de 95 % es razonable con un modelo profundo y un conjunto de datos representativo, pero es exigente "
      "para un modelo clásico sobre fruta real; en el simulador se evalúa con una dificultad definida de antemano."))
add(P("<b>Relevante.</b> M1 a M3 miden la calidad de la inspección; M4 y M5 miden la operación en una banda dinámica con fallas; M6 verifica el historial "
      "por proveedor que motiva el diseño."))

add(P("8. Componentes de la arquitectura de IA agéntica (marco IBM)", "h1"))
add(P("Los agentes de IA perciben su entorno, deciden y actúan de manera autónoma. IBM describe siete componentes funcionales; la siguiente tabla indica cómo los "
      "cubre FreshCheck-Agent (el detalle por módulo está en la sección 9.1)."))
add(tabla([
    ["Componente (IBM)", "Cómo lo cubre FreshCheck-Agent"],
    ["Percepción y manejo de entradas", "Valida la imagen, normaliza la iluminación con la banda visible, segmenta la fruta y extrae características."],
    ["Planificación y descomposición de tareas", "Secuencia fija por unidad: percibir → decidir → actuar → registrar; no requiere planificación deliberativa porque el ambiente es episódico."],
    ["Memoria", "Corto plazo: estado de la cámara, banda y velocidad. Largo plazo: contadores por lote y log de ejecución."],
    ["Razonamiento y toma de decisiones", "Modelo entrenado que infiere la clase y regla condición-acción que decide continuar, desviar o derivar."],
    ["Llamada de herramientas y acción", "Interfaz del desviador (simulada en el curso) que recibe la orden con su plazo."],
    ["Comunicación", "Reporte de calidad por lote y proveedor; interfaz de demostración."],
    ["Aprendizaje y adaptación", "Reentrenamiento periódico con muestras corregidas por el personal de calidad."],
], [5.4, 11.1], negrita_primera=True))
add(P("Fuente: IBM, «Components of AI agents», https://www.ibm.com/mx-es/think/topics/components-of-ai-agents", "leyenda"))

# =================================================================== 9. DESARROLLO
add(PageBreak())
add(P("9. Desarrollo del agente", "h1"))
add(P("9.1 Arquitectura", "h2"))
add(P("El agente se implementó en Python como un paquete (<i>freshcheck</i>) con módulos separados para el entorno, la percepción, el modelo, el agente y el "
      "actuador. El ciclo percepción → decisión → acción es explícito en <i>FreshCheckAgent.paso()</i>, que invoca <i>percibir()</i>, <i>decidir()</i> y <i>actuar()</i>."))
add(figura(FIG / "arquitectura.png", 16.4, "Figura 1. Diagrama de componentes de FreshCheck-Agent."))
add(tabla([
    ["Componente IBM", "Módulo o clase del código", "Estado", "Observación"],
    ["Percepción", "perception.py (validar_imagen, procesar_imagen); FreshCheckAgent.percibir()", "Implementado", "Valida, normaliza, segmenta y extrae 38 características."],
    ["Planificación", "FreshCheckAgent.paso()", "Implementado", "Pipeline fijo; sin planificación deliberativa, por diseño."],
    ["Memoria", "ModeloDelMundo; contadores por lote; log CSV", "Implementado", "Corto plazo en memoria; largo plazo en el log."],
    ["Razonamiento", "ClasificadorEstado.predecir_proba(); FreshCheckAgent.decidir()", "Implementado", "Modelo entrenado más reglas condición-acción."],
    ["Llamada de herramientas", "actuator.py (InterfazDesviador, DesviadorSimulado)", "Implementado (simulado)", "Sin integración con un PLC real."],
    ["Comunicación", "reports.py; app.py", "Implementado", "Reporte por proveedor e interfaz Streamlit."],
    ["Aprendizaje", "learning.py (RetroalimentacionCalidad, reentrenar)", "Parcial", "Reentrenamiento manual y offline; la retroalimentación del experimento E2 proviene del simulador, no de personal real."],
], [3.0, 5.6, 2.6, 5.3], negrita_primera=True))

add(P("9.2 Tecnologías", "h2"))
tec = [
    ["Tecnología", "Versión", "Para qué se usó", "Por qué frente a una alternativa", "Quién razona"],
    ["Python", platform.python_version(), "Lenguaje de todo el sistema", "Conocido por el estudiante; ecosistema de IA", "—"],
    ["NumPy", ver("numpy"), "Matrices de imagen y simulación aleatoria", "Estándar; base de OpenCV y scikit-learn", "Cálculo"],
    ["OpenCV (headless)", ver("opencv-python-headless"), "Color, segmentación, contornos y componentes conexas", "Más rápido y ligero que scikit-image para este pipeline", "Código propio sobre OpenCV"],
    ["scikit-learn", ver("scikit-learn"), "Clasificador HistGradientBoosting", "Entrena en minutos en una CPU; frente a una CNN (PyTorch/TensorFlow) exige menos datos y hardware", "El modelo aprende el límite entre clases"],
    ["pandas", ver("pandas"), "Log, reportes y métricas", "Agrupaciones y CSV con pocas líneas", "Cálculo"],
    ["joblib", ver("joblib"), "Guardar y cargar el modelo", "Recomendado por scikit-learn", "—"],
    ["Matplotlib", ver("matplotlib"), "Diagramas de arquitectura y secuencia", "Sin dependencias externas; frente a PlantUML no requiere Java (se incluyen también las fuentes .puml)", "—"],
    ["Streamlit", ver("streamlit"), "Interfaz de demostración y despliegue", "Más rápido que Flask para una demo de datos", "—"],
    ["ReportLab", ver("reportlab"), "Generación del informe PDF", "Reproducible desde el repositorio", "—"],
]
add(tabla(tec, [2.6, 1.7, 3.6, 5.6, 3.0], negrita_primera=True))
add(P("<b>Uso de modelos de lenguaje.</b> El agente en ejecución no usa ningún LLM: todo el razonamiento lo realiza el código y el clasificador entrenado. "
      "Un asistente de IA (Claude, de Anthropic) se empleó para apoyar el desarrollo del código y del informe, como se declara en la bitácora de uso de IA.", "nota"))

add(P("9.3 Entradas (inputs)", "h2"))
ej = log_c.iloc[0]
ent = [
    ["Nombre", "Tipo de dato", "Rango o formato", "Origen", "Ejemplo real (log)"],
    ["imagen", "ndarray uint8, BGR", "160×160×3; valores 0–255", "Sensor simulado (cámara)", f"brillo medio {num(float(ej['brillo']), 2)}; unit_id {int(ej['unit_id'])} de {ej['run_id']}"],
    ["t_s", "float", "segundos ≥ 0", "Reloj simulado", f"{num(float(ej['t_sim_s']), 4)}"],
    ["velocidad_banda", "float", "0,05–3,0 m/s; ruido 2 %", "Encoder simulado", f"{num(float(ej['velocidad_lectura']), 4)}"],
    ["disparo", "bool", "True/False", "Sensor de presencia simulado", "True"],
    ["lot_id, proveedor", "str", "p. ej. «L-001», «Proveedor A»", "Lector de código simulado", f"{ej['lot_id']}, {ej['proveedor']}"],
    ["unit_id", "int", "≥ 0", "Contador del entorno", f"{int(ej['unit_id'])}"],
    ["escenario, n_unidades, semilla", "str, int, int", "13 escenarios; 20–300 unidades", "Usuario (CLI o interfaz)", "N1_nominal, 200, 9000"],
]
add(tabla(ent, [3.1, 2.6, 3.6, 3.4, 3.8], negrita_primera=True))

add(Spacer(1, 0.2 * cm))
add(figura(FIG / "muestras_simulador.png", 8.2, "Figura 2. Percepciones generadas por el simulador. Filas: manzana, naranja, banano y pera (esta última no se usa en el entrenamiento). Columnas: fresca, en maduración y dañada."))
add(P("9.4 Salidas y acciones", "h2"))
ej_d = log_c[(log_c["accion"] == "DESVIAR") & (log_c["ejecutado"].astype(str) == "True")].iloc[0]
ej_r = log_f[log_f["accion"] == "REVISION_MANUAL"].iloc[0]
sal = [
    ["Acción", "Efecto en el entorno", "Ejemplo real (log)"],
    ["CONTINUAR", "La unidad sigue al empaque final.", f"{ej['run_id']}, unidad {int(ej['unit_id'])}: {ej['clase_predicha']} ({num(float(ej['p_fresca']), 4)}) → destino {ej['destino']}"],
    ["DESVIAR", "El desviador acciona y la unidad va al descarte; si hay retraso o recarga, pasa al empaque (desvío fallido).",
     f"{ej_d['run_id']}, unidad {int(ej_d['unit_id'])}: {ej_d['clase_predicha']} ({num(float(ej_d['p_danada']), 4)}) → {ej_d['destino']}"],
    ["REVISION_MANUAL", "La unidad se deriva a inspección humana y se registra el motivo.",
     f"{ej_r['run_id']}, unidad {int(ej_r['unit_id'])}: {ej_r['motivo_decision']} → {ej_r['destino']}"],
    ["NINGUNA", "No hay fruta en la banda; no se actúa.", "Escenario F08_banda_vacia: motivo SIN_FRUTA"],
    ["Registro y reporte", "Escribe una fila por unidad y genera el reporte por lote y proveedor.", "logs/pruebas/log_clasificacion.csv; reportes_proveedor.csv"],
]
add(tabla(sal, [3.0, 6.3, 7.2], negrita_primera=True))

add(PageBreak())
add(P("9.5 Diagramas de secuencia UML", "h2"))
add(P("Se presentan dos escenarios: el típico (clasificación y desvío de una unidad, con los casos de desvío ejecutado y fallido) y uno de excepción (sensor sin "
      "dato, cámara en falla y recuperación). Ambos coinciden con los métodos del código; las fuentes PlantUML están en docs/diagramas/."))
add(figura(FIG / "secuencia_tipico.png", 16.4, "Figura 3. Diagrama de secuencia, escenario típico."))
add(PageBreak())
add(figura(FIG / "secuencia_excepcion.png", 16.4, "Figura 4. Diagrama de secuencia, escenario de excepción."))

add(P("9.6 Diferencias entre diseño e implementación", "h2"))
dif = [
    ["Aspecto", "Diseño", "Implementación", "Justificación"],
    ["Modelo de visión", "Visión por computador entrenada, en la línea de aprendizaje profundo", "Características clásicas (color, anomalías, textura) y HistGradientBoosting",
     "Un solo núcleo de CPU, entrenamiento en ~90 s y características interpretables. Contrapartida: la literatura reporta menor exactitud que las CNN en fruta real."],
    ["Datos", "FruitVision, fotografías propias e imágenes de internet", "Imágenes sintéticas del simulador", "La guía permite un simulador; los datos reales no se integraron. El script --real-data deja la ruta lista."],
    ["Cámara y desviador", "Hardware real conectado a la banda", "Entorno y actuador simulados con plazo y recarga", "No hay instalación industrial."],
    ["Acciones", "Continuar o desviar", "Se agrega REVISION_MANUAL", "Acción segura ante percepción inválida o frutas en contacto."],
    ["Aprendizaje", "Reentrenamiento periódico con corrección del personal de calidad", "Reentrenamiento manual y offline", "No hay personal real que corrija; se prueba con muestras del simulador (E2)."],
    ["Estado interno", "No definido", "ModeloDelMundo", "Coherencia con el tipo de agente (ver tabla de cambios)."],
]
add(tabla(dif, [2.4, 4.0, 4.4, 5.7], negrita_primera=True))

# =================================================================== 9.7 PRUEBAS
elementos.extend(a_horizontal())
add(P("9.7 Pruebas contra las medidas de desempeño", "h2"))
add(P(f"Todas las medidas se probaron con los umbrales fijados de antemano. Se ejecutaron <b>13 escenarios de clasificación</b> (8 nominales y 5 límite) y "
      f"<b>11 de falla</b> (10 dirigidos y 1 de mezcla aleatoria), con <b>5 repeticiones</b> por escenario (semillas 9000 en adelante, distintas de las de entrenamiento y "
      f"validación) y 200 unidades por corrida de clasificación. Como el entorno es estocástico, se reporta la media y la desviación estándar entre corridas (DE)."))
n_nom, n_lim = M["M1_exactitud_nominal"]["n_corridas"], M["M1_exactitud_limite"]["n_corridas"]
tabla_res = [["Medida de desempeño", "Fórmula", "Umbral", "Casos ejecutados", "Resultado obtenido", "¿Cumple?", "Evidencia (log, archivo)"]]


def si(c):
    return "<font color='#1E6B3A'><b>Sí</b></font>" if c else "<font color='#9C1C1C'><b>No</b></font>"


m1n, m1l = M["M1_exactitud_nominal"], M["M1_exactitud_limite"]
m3n, m3l = M["M3_descarte_indebido_nominal"], M["M3_descarte_indebido_limite"]
tabla_res += [
    ["M1. Exactitud (nominal)", "correctas ÷ unidades válidas", "≥ 95 %", f"8 escenarios × 5 rep. × 200 = {n_nom * 200:,} unidades".replace(",", "."),
     f"{pct(m1n['media'], 2)} (DE {num(100 * m1n['sd_entre_corridas'], 2)} pp); peor: {m1n['peor_escenario']} {pct(m1n['valor_peor_escenario'], 1)}", si(m1n["cumple"]),
     "log_clasificacion.csv, estrato = nominal; resultados_por_corrida.csv"],
    ["M1. Exactitud (límite)", "igual", "≥ 85 %", f"5 × 5 × 200 = {n_lim * 200:,} unidades".replace(",", "."),
     f"{pct(m1l['media'], 2)} (DE {num(100 * m1l['sd_entre_corridas'], 2)} pp); peor: {m1l['peor_escenario']} {pct(m1l['valor_peor_escenario'], 1)}", si(m1l["cumple"]),
     "log_clasificacion.csv, estrato = limite"],
    ["M2. Reducción de defectuosas", "(D0 − D1) ÷ D0", "≥ 30 %", f"13 × 5 × 200 = 13.000 unidades; D0 = {d0_total:,}".replace(",", "."),
     f"{pct(M['M2_reduccion_defectuosas']['media'], 2)} (DE {num(100 * M['M2_reduccion_defectuosas']['sd_entre_corridas'], 2)} pp); D0 = {d0_total}, D1 = {d1_total}; peor: {M['M2_reduccion_defectuosas']['peor_escenario']} {pct(M['M2_reduccion_defectuosas']['valor_peor_escenario'], 1)}",
     si(M["M2_reduccion_defectuosas"]["cumple"]), "log_clasificacion.csv (columnas clase_real, destino)"],
    ["M3. Descarte indebido (nominal)", "no dañadas desviadas ÷ no dañadas", "≤ 5 %", "8.000 unidades",
     f"{pct(m3n['media'], 2)} (DE {num(100 * m3n['sd_entre_corridas'], 2)} pp)", si(m3n["cumple"]), "log_clasificacion.csv"],
    ["M3. Descarte indebido (límite)", "igual", "≤ 10 %", "5.000 unidades",
     f"{pct(m3l['media'], 2)} (DE {num(100 * m3l['sd_entre_corridas'], 2)} pp); peor: {m3l['peor_escenario']} {pct(m3l['valor_peor_escenario'], 2)}", si(m3l["cumple"]), "log_clasificacion.csv"],
    ["M4. Latencia de decisión", "p95 de percepción + decisión", "≤ 200 ms", "13.000 decisiones",
     f"p95 = {num(M['M4_latencia_p95_ms']['media'], 1)} ms; máximo {num(M['M4_latencia_p95_ms']['maximo_observado'], 1)} ms", si(M["M4_latencia_p95_ms"]["cumple"]), "log_clasificacion.csv (latencia_ms)"],
    ["M4. Decisiones tardías", "latencia > plazo", "0", "13.000 decisiones", f"{M['M4_decisiones_tardias']['total']} tardías", si(M["M4_decisiones_tardias"]["cumple"]), "log_clasificacion.csv (tardia)"],
    ["M5. Robustez ante fallas", "manejadas ÷ inyecciones", "100 %", f"11 escenarios × 5 rep.; {M['M5_robustez_fallas']['instancias']} inyecciones",
     f"{pct(M['M5_robustez_fallas']['media'], 2)} ({M['M5_robustez_fallas']['manejadas']} de {M['M5_robustez_fallas']['instancias']})", si(M["M5_robustez_fallas"]["cumple"]),
     "log_fallas.csv; resultados_fallas.csv (columna detalle)"],
    ["M6. Consistencia del reporte", "celdas iguales ÷ celdas", "100 %", f"{M['M6_consistencia_reporte']['n_corridas']} corridas",
     f"{pct(M['M6_consistencia_reporte']['media'], 1)}", si(M["M6_consistencia_reporte"]["cumple"]), "reportes_proveedor.csv (consistencia_con_agente)"],
]
add(tabla(tabla_res, [3.4, 3.2, 1.5, 3.7, 5.9, 1.9, 4.6]))
add(P("Tabla de resultados obligatoria. Las medidas con dos estratos (M1, M3) y con dos criterios (M4) ocupan una fila por criterio.", "leyenda"))

elementos.extend(a_vertical())
add(P("Escenarios de prueba", "h3"))
esc_desc = {
    "N1_nominal": "Condiciones nominales", "N2_luz_baja": "Iluminación 20 % menor", "N3_luz_alta": "Iluminación 25 % mayor",
    "N4_ruido": "Ruido alto y desenfoque leve", "N5_posicion": "Fruta muy desplazada; escala y rotación extremas", "N6_banda_lenta": "Banda a 0,3 m/s",
    "N7_una_variedad": "Lote de una sola variedad (naranja)", "N8_lote_critico": "Lote con 40 % de fruta dañada",
    "L1_luz_muy_baja": "Iluminación al 55 % (fuera del rango de entrenamiento)", "L2_sobreexposicion": "Iluminación al 155 % (fuera del rango)",
    "L3_desenfoque": "Desenfoque fuerte y ruido alto", "L4_frontera": "Maduración incipiente y daños pequeños (casos fronterizos)",
    "L5_alta_densidad": "Banda a 1,5 m/s con unidades muy próximas",
}
filas = [["Escenario", "Descripción", "M1 exactitud (media; DE)", "M2 reducción", "M3 descarte indebido", "p95 (ms)"]]
for k, d_ in esc_desc.items():
    v = PE[k]
    filas.append([k, d_, f"{pct(v['M1_exactitud']['media'], 1)}; {num(100 * v['M1_exactitud']['sd'], 2)} pp", pct(v["M2_reduccion"]["media"], 1),
                  pct(v["M3_descarte_indebido"]["media"], 2), num(v["M4_latencia_p95_ms"]["media"], 1)])
add(tabla(filas, [3.0, 5.0, 3.2, 1.9, 2.0, 1.4], negrita_primera=True))
add(Spacer(1, 0.25 * cm))
add(P("Escenarios de falla (M5)", "h3"))
fdesc = {
    "F01_sin_dato": "Sensor sin imagen (None)", "F02_imagen_vacia": "Imagen de tamaño cero", "F03_formato": "Imagen en escala de grises",
    "F04_nan": "Imagen con valores NaN", "F05_tipo": "Cadena de texto en lugar de imagen", "F06_lente_tapado": "Imagen completamente oscura",
    "F07_saturada": "Imagen saturada en blanco", "F08_banda_vacia": "Disparo con la banda vacía", "F09_solapadas": "Dos frutas en contacto",
    "F10_rafaga": "Ráfaga de 5 imágenes sin dato y recuperación", "F11_mezcla": "Mezcla aleatoria (12 % de las unidades)",
}
filas = [["Escenario", "Falla inyectada", "Inyecciones", "Manejadas", "Tasa"]]
for esc_, g in fal.groupby("escenario", sort=False):
    filas.append([esc_, fdesc[esc_], str(int(g["instancias"].sum())), str(int(g["manejadas"].sum())), pct(g["manejadas"].sum() / g["instancias"].sum(), 1)])
add(tabla(filas, [3.6, 6.2, 2.0, 2.0, 2.0], negrita_primera=True))

add(P("Experimentos complementarios (no son medidas del diseño)", "h3"))
e1 = extra["E1_ablacion"]["exactitud"]
rb = extra["E1_ablacion"]["rafaga_de_fallas"]
e2 = extra["E2_reentrenamiento"]
filas = [["Escenario", "Con modelo del mundo", "Sin modelo del mundo (reflejo simple)"]]
for k, v in e1.items():
    filas.append([k, f"{pct(v['con_modelo_del_mundo']['media'], 1)} (DE {num(100 * v['con_modelo_del_mundo']['sd'], 1)})",
                  f"{pct(v['sin_modelo_del_mundo']['media'], 1)} (DE {num(100 * v['sin_modelo_del_mundo']['sd'], 1)})"])
add(tabla(filas, [4.5, 5.5, 6.5], negrita_primera=True))
add(P(f"<b>E1 (ablación, 5 repeticiones × 100 unidades).</b> El estado interno no mejora la exactitud de clasificación: las diferencias (menos de un punto porcentual) "
      f"están dentro de la variación entre corridas. Su aporte se observa en la gestión de fallas: con una ráfaga de cinco cuadros sin dato, el agente con estado marca la cámara "
      f"«{rb['con_modelo_del_mundo']['estado_camara_unidad_13']}» y deriva {rb['con_modelo_del_mundo']['revision_manual_unidades_10_a_16']} unidades de las 10 a la 16 a inspección manual "
      f"(incluye el primer cuadro válido tras la falla, por la histéresis de recuperación), mientras que la versión sin estado reporta la cámara «{rb['sin_modelo_del_mundo']['estado_camara_unidad_13']}» "
      f"y deriva {rb['sin_modelo_del_mundo']['revision_manual_unidades_10_a_16']}."))
add(P(f"<b>E2 (variedad no vista y reentrenamiento).</b> Con la variedad pera, ausente del entrenamiento, la exactitud cae a {pct(e2['exactitud_pera_antes']['media'], 1)} "
      f"(DE {num(100 * e2['exactitud_pera_antes']['sd'], 1)}). Tras reentrenar con {e2['muestras_de_retroalimentacion']} muestras etiquetadas sube a {pct(e2['exactitud_pera_despues']['media'], 1)} "
      f"(DE {num(100 * e2['exactitud_pera_despues']['sd'], 1)}) y la exactitud nominal se mantiene en {pct(e2['exactitud_nominal_tras_reentrenar']['media'], 1)}. Esto muestra tanto la "
      f"limitación (variedades nuevas) como el valor del componente de aprendizaje; las muestras provienen del simulador, no de personal de calidad."))

add(PageBreak())
add(P("Análisis de resultados", "h3"))
add(P(f"<b>Medidas cumplidas.</b> M1 (nominal {pct(m1n['media'], 1)}; límite {pct(m1l['media'], 1)}), M2 ({pct(M['M2_reduccion_defectuosas']['media'], 1)}), M3 "
      f"({pct(m3n['media'], 2)} nominal; {pct(m3l['media'], 2)} límite), M4 (p95 de {num(M['M4_latencia_p95_ms']['media'], 1)} ms y ninguna decisión tardía) y M6 (100 %). Sin agente, "
      f"{pct(M['M2_reduccion_defectuosas']['tasa_defecto_sin_agente_media'], 1)} de las unidades que llegarían al empaque serían dañadas; con el agente, "
      f"{pct(M['M2_reduccion_defectuosas']['tasa_defecto_con_agente_media'], 1)}."))
add(P(f"<b>Medida incumplida: M5 ({pct(M['M5_robustez_fallas']['media'], 2)} frente a 100 %).</b> Las {M['M5_robustez_fallas']['instancias'] - M['M5_robustez_fallas']['manejadas']} inyecciones "
      f"no manejadas corresponden a F09 (frutas en contacto): en {f09_inst - f09_ok} de {f09_inst} casos el agente trató la silueta fusionada como una sola fruta y la clasificó "
      f"(CONTINUAR o DESVIAR) en lugar de derivarla a revisión. <i>Causa probable:</i> la regla de detección usa la solidez de la silueta (&lt; 0,94) y, cuando el cuello entre las frutas es "
      f"poco profundo, la solidez supera el umbral. Bajar el umbral no es gratuito: con la regla actual ya se marcaron como solapamiento {solap_falsos} de {miles(n_validas)} unidades válidas "
      f"({pct(solap_falsos / n_validas, 2)}), que cuentan como error en M1. <i>Corrección propuesta:</i> detectar el contacto con defectos de convexidad del contorno o con una "
      f"transformada de distancia (watershed), y validar el umbral con un conjunto que incluya frutas aisladas y en contacto."))
add(P(f"<b>Otros hallazgos que no incumplen un umbral.</b> (a) En L5 (banda a 1,5 m/s con unidades muy próximas) M2 baja a {pct(PE['L5_alta_densidad']['M2_reduccion']['media'], 1)}: "
      f"{l5_ocupado} de {l5_total} desvíos no se ejecutaron por «DESVIADOR_OCUPADO» (recarga de {int(60)} ms). Correcciones posibles: un segundo desviador, un menor tiempo de recarga o limitar la "
      f"velocidad de la banda. (b) Los escenarios más bajos de M1 fueron L4 ({pct(PE['L4_frontera']['M1_exactitud']['media'], 1)}) y L3 ({pct(PE['L3_desenfoque']['M1_exactitud']['media'], 1)}); "
      f"{err_fm} de los errores confunden fresca con maduración (y viceversa), es decir, ocurren en la frontera entre clases, mientras que solo {err_dan_perdida} unidades dañadas se clasificaron como "
      f"no dañadas. (c) La exactitud fuera del rango de iluminación de entrenamiento se mantiene alta (L1 {pct(PE['L1_luz_muy_baja']['M1_exactitud']['media'], 1)}; L2 {pct(PE['L2_sobreexposicion']['M1_exactitud']['media'], 1)}); es plausible que la normalización con la banda visible lo explique, pero no se probó por separado."))
add(P("<b>Cautela de interpretación.</b> Estos resultados provienen de imágenes sintéticas cuya dificultad se fijó antes de las pruebas. Con fruta real cabe esperar una exactitud menor, "
      "en línea con las cifras de la visión clásica citadas en la sección 7; el estado interno, además, no mejoró la exactitud en el simulador (E1)."))

add(P("9.8 Limitaciones y trabajo futuro", "h2"))
for t in [
    "<b>Datos sintéticos.</b> Validar con FruitVision y fotografías propias: <i>python scripts/train.py --real-data RUTA</i> y repetir <i>run_tests.py</i> con un simulador reemplazado por el conjunto real.",
    "<b>Modelo.</b> Sustituir las características clásicas por una CNN con transferencia de aprendizaje (por ejemplo EfficientNet o MobileNet), manteniendo el contrato <i>predecir_proba</i>.",
    "<b>Segmentación.</b> Asume una banda de color uniforme visible en el borde; con fondos complejos habría que usar detección (por ejemplo YOLO).",
    "<b>Frutas en contacto y alta densidad.</b> Mejorar la detección de contacto (M5) y evaluar un segundo desviador (L5).",
    "<b>Hardware.</b> No se probó con una cámara ni un PLC reales; la interfaz <i>InterfazDesviador</i> deja el punto de integración.",
    "<b>Aprendizaje.</b> El reentrenamiento es manual; falta un ciclo con retroalimentación real del personal de calidad y monitoreo de deriva.",
    "<b>Medida de reducción.</b> M2 usa como línea base «sin agente»; una línea base de inspección manual daría una comparación más exigente.",
]:
    add(B(t))

add(P("9.9 Enlaces y evidencia de funcionamiento", "h2"))
add(tabla([
    ["Elemento", "Enlace"],
    ["Repositorio (README, requirements.txt, código)", "[PEGAR ENLACE DEL REPOSITORIO]"],
    ["Plataforma de despliegue (Streamlit) o video de máximo 20 minutos", "[PEGAR ENLACE]"],
    ["Logs de las pruebas", "logs/pruebas/ (log_clasificacion.csv, log_fallas.csv, resultados_por_corrida.csv, resultados_fallas.csv, reportes_proveedor.csv, resultados_resumen.json, experimentos_extra.json)"],
], [8.0, 8.5], negrita_primera=True))

# =================================================================== 10 CONCLUSIONES
add(P("10. Conclusiones", "h1"))
add(P(f"FreshCheck-Agent se implementó como un agente reflejo basado en modelo coherente con su diseño: ciclo percepción → decisión → acción explícito, estado interno "
      f"que condiciona la decisión, ambiente estocástico y dinámico reproducido por el simulador, y los mismos sensores y actuadores del diseño (con las diferencias documentadas en 9.6). "
      f"En las pruebas se cumplieron las medidas de exactitud, reducción de defectuosas, descarte indebido, latencia y consistencia del reporte, y no se cumplió la robustez ante fallas "
      f"({pct(M['M5_robustez_fallas']['media'], 2)}), por un problema acotado y diagnosticado: la detección de frutas en contacto."))
add(P("La principal limitación es que la evidencia es sintética. Los resultados demuestran que el agente funciona sobre el entorno simulado y que su arquitectura gestiona fallas y reportes, "
      "pero no que alcance el 95 % sobre fruta real; para eso se requiere reentrenar con FruitVision o fotografías propias. La matriz de decisión favoreció a FreshCheck-Agent por un margen estrecho, "
      "y el análisis de sensibilidad lo reconoce. Desde el punto de vista operativo, conviene mantener supervisión humana (la acción REVISION_MANUAL) y vigilar el costo de los descartes indebidos."))

# =================================================================== REFERENCIAS
add(P("Referencias", "h1"))
for r in [
    "Russell, S., &amp; Norvig, P. (2021). <i>Artificial intelligence: A modern approach</i> (4.ª ed., Global Edition). Pearson.",
    "IBM. (s. f.). <i>Components of AI agents</i>. IBM Think. https://www.ibm.com/mx-es/think/topics/components-of-ai-agents",
    "Ismail, N., &amp; Malik, O. A. (2022). Real-time visual inspection system for grading fruits using computer vision and deep learning techniques. <i>Information Processing in Agriculture, 9</i>(1), 24–37. https://doi.org/10.1016/j.inpa.2021.01.005",
    "Machine vision-based automatic fruit quality detection and grading. (2023). <i>Frontiers of Agricultural Science and Engineering</i>. https://doi.org/10.15302/J-FASE-2023532",
    "FruitVision. Conjunto de datos de imágenes de frutas frescas, en descomposición y adulteradas para inspección automatizada de calidad. https://www.sciencedirect.com/science/article/pii/S2352340925004792",
    "Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. <i>Journal of Machine Learning Research, 12</i>, 2825–2830.",
    "Bradski, G. (2000). The OpenCV Library. <i>Dr. Dobb's Journal of Software Tools</i>.",
    "Puertas, E. (2026). <i>Intelligent Agents</i> [Material de clase, curso de Inteligencia Artificial]. Universidad Tecnológica de Bolívar.",
    "IBM HR Analytics Employee Attrition &amp; Performance (Kaggle). https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset",
    "Employee Attrition Prediction: An Explanatory and Statistically Robust Ensemble Learning Model (MDPI). https://www.mdpi.com/2073-431X/15/3/185",
]:
    add(B(r))

# =================================================================== ANEXO
add(P("Anexo. Reproducibilidad", "h1"))
add(P(f"Semillas: entrenamiento {entren['seed_entrenamiento']}, validación {entren['seed_validacion']}, pruebas 9000 en adelante. Modelo: {entren['n_entrenamiento']} muestras de entrenamiento, "
      f"exactitud de validación {pct(entren['exactitud_validacion'], 1)}. Umbrales: config/umbrales.json, SHA-256 {hash_umbrales}. Comandos: <i>python scripts/train.py</i>, "
      f"<i>python scripts/run_tests.py</i>, <i>python scripts/run_extras.py</i>, <i>python scripts/decision_matrix.py</i>, <i>python scripts/build_informe.py</i>."))

doc = crear_documento(RAIZ / "docs" / "Informe_FreshCheck_Agent.pdf", "FreshCheck-Agent — Informe de diseño, desarrollo y pruebas",
                      titulo="FreshCheck-Agent: diseño ajustado, desarrollo y pruebas")
doc.build(elementos)
print("Informe generado:", RAIZ / "docs" / "Informe_FreshCheck_Agent.pdf")
