"""
Genera los diagramas del informe en docs/figuras/ (PNG) y las fuentes PlantUML en docs/diagramas/.

    python scripts/make_diagrams.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle   # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
FIG = RAIZ / "docs" / "figuras"
UML = RAIZ / "docs" / "diagramas"
FIG.mkdir(parents=True, exist_ok=True)
UML.mkdir(parents=True, exist_ok=True)

AZUL, AZUL_CL = "#1F3864", "#DCE6F5"
VERDE_CL, NARANJA_CL, GRIS_CL = "#DDEFE0", "#FBE7C9", "#EEEEEE"


# --------------------------------------------------------------------------- #
# Diagrama de secuencia
# --------------------------------------------------------------------------- #
def secuencia(titulo, participantes, filas, archivo, ancho=15.5, paso=0.62):
    n = len(participantes)
    xs = {p: 1.4 + i * (ancho - 2.8) / (n - 1) for i, p in enumerate(participantes)}
    total = len(filas) * 0.93 + 2
    alto = total * paso + 2.2
    fig, ax = plt.subplots(figsize=(ancho, alto))
    ax.set_xlim(0, ancho)
    ax.set_ylim(0, alto)
    ax.axis("off")
    ax.text(ancho / 2, alto - 0.35, titulo, ha="center", va="center", fontsize=13, fontweight="bold", color=AZUL)
    y_top = alto - 1.2
    for p, x in xs.items():
        ax.add_patch(FancyBboxPatch((x - 0.82, y_top - 0.25), 1.64, 0.62, boxstyle="round,pad=0.02",
                                    fc=AZUL_CL, ec=AZUL, lw=1.2))
        ax.text(x, y_top + 0.06, p, ha="center", va="center", fontsize=7.2, fontweight="bold")
    y = y_top - 0.6
    y_fin = y_top - 0.6 - len(filas) * paso - 0.2
    for p, x in xs.items():
        ax.plot([x, x], [y_top - 0.25, y_fin], color="#777777", lw=0.9, ls=(0, (4, 3)), zorder=0)

    pila, cajas = [], []
    for f in filas:
        k = f[0]
        if k == "msg":
            _, a, b, texto, retorno = f
            y -= paso
            if a == b:
                x = xs[a]
                ax.plot([x, x + 0.55, x + 0.55, x], [y + 0.12, y + 0.12, y - 0.12, y - 0.12], color="black", lw=1.0)
                ax.annotate("", xy=(x, y - 0.12), xytext=(x + 0.2, y - 0.12),
                            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
                ax.text(x + 0.65, y, texto, ha="left", va="center", fontsize=7.6)
            else:
                xa, xb = xs[a], xs[b]
                estilo = dict(arrowstyle="-|>", color="black", lw=1.1, ls="--" if retorno else "-")
                ax.annotate("", xy=(xb, y), xytext=(xa, y), arrowprops=estilo)
                ax.text((xa + xb) / 2, y + 0.09, texto, ha="center", va="bottom", fontsize=7.6,
                        bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.9))
        elif k == "ini":
            _, tipo, etiqueta = f
            y -= paso * 0.8
            pila.append([tipo, etiqueta, y, None])
        elif k == "else":
            _, etiqueta = f
            y -= paso * 0.6
            pila[-1][3] = (y, etiqueta)
        elif k == "fin":
            y -= paso * 0.5
            tipo, etiqueta, y0, sino = pila.pop()
            cajas.append((tipo, etiqueta, y0, y, sino, len(pila)))
    for tipo, etiqueta, y0, y1, sino, nivel in cajas:
        m = 0.35 + 0.18 * nivel
        x0, x1 = 0.4 + m * 0.5, ancho - 0.4 - m * 0.5
        ax.add_patch(Rectangle((x0, y1), x1 - x0, y0 - y1 + 0.3, fill=False, ec="#444444", lw=1.1, zorder=1))
        ax.add_patch(Rectangle((x0, y0 + 0.3 - 0.3), 0.9, 0.3, fc=GRIS_CL, ec="#444444", lw=1.0, zorder=2))
        ax.text(x0 + 0.45, y0 + 0.15, tipo, ha="center", va="center", fontsize=8, fontweight="bold", zorder=3)
        ax.text(x0 + 1.0, y0 + 0.15, f"[{etiqueta}]", ha="left", va="center", fontsize=7.8, zorder=3)
        if sino:
            ys, et2 = sino
            ax.plot([x0, x1], [ys, ys], color="#444444", lw=1.0, ls=(0, (5, 3)), zorder=2)
            ax.text(x0 + 0.1, ys - 0.17, f"[{et2}]", ha="left", va="center", fontsize=7.8, zorder=3)
    fig.savefig(FIG / archivo, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


PART = ["Operador\n(actor externo)", "Interfaz\n(CLI / Streamlit)", "Entorno\n(línea simulada)",
        "Agente:\nPercepción", "Agente:\nDecisión", "Agente:\nAcción", "Desviador\n(actuador)",
        "Registro y reportes\n(servicio externo)"]
O, I, E, P, D, A, V, R = PART


def m(a, b, t, ret=False):
    return ("msg", a, b, t, ret)


TIPICO = [
    m(O, I, "iniciar_simulacion(escenario, n_unidades, seed)"),
    m(I, E, "flujo(n_unidades)"),
    ("ini", "loop", "por cada unidad que llega a la cámara"),
    m(E, P, "Percepto(imagen, t_s, lote, velocidad_banda)"),
    m(P, P, "validar_imagen(); normalizar_iluminacion(); segmentar_fruta()"),
    m(P, P, "ModeloDelMundo.actualizar(válida, banda_ref, velocidad)"),
    m(P, D, "Percepcion(codigo=OK, características)"),
    m(D, D, "predecir_proba(x) → clase"),
    ("ini", "alt", "clase == danada"),
    m(D, A, "Decision(DESVIAR)"),
    m(A, V, "accionar(unit_id, t_act, latencia_ms, plazo_ms)"),
    ("ini", "alt", "latencia ≤ plazo y desviador libre"),
    m(V, A, "Resultado(ejecutado=True) → destino=descarte", True),
    ("else", "DECISION_TARDIA o DESVIADOR_OCUPADO"),
    m(V, A, "Resultado(ejecutado=False) → destino=empaque_final", True),
    ("fin",),
    ("else", "clase == fresca o maduracion"),
    m(D, A, "Decision(CONTINUAR)"),
    ("fin",),
    m(A, R, "registrar(percepción, decisión, acción, latencia)"),
    m(A, E, "efecto(unit_id, destino)", True),
    ("fin",),
    m(I, R, "reporte_por_proveedor()"),
    m(R, I, "tabla por lote y proveedor", True),
    m(I, O, "decisiones, métricas y reporte", True),
]

EXCEPCION = [
    m(O, I, "iniciar_simulacion(escenario='F10_rafaga')"),
    m(I, E, "flujo(n_unidades)"),
    ("ini", "loop", "unidades 10 a 16 (ráfaga de fallas y recuperación)"),
    m(E, P, "Percepto(imagen=None) — sensor sin dato"),
    m(P, P, "validar_imagen() → SENSOR_SIN_DATO"),
    m(P, P, "ModeloDelMundo.actualizar(inválida): invalidos_consecutivos += 1"),
    ("ini", "alt", "invalidos_consecutivos ≥ 3"),
    m(P, P, "estado_camara = EN_FALLA"),
    ("else", "menos de 3 fallas seguidas"),
    m(P, P, "estado_camara = DEGRADADA"),
    ("fin",),
    m(P, D, "Percepcion(codigo=SENSOR_SIN_DATO)"),
    ("ini", "alt", "código ≠ OK, o cámara EN_FALLA"),
    m(D, A, "Decision(REVISION_MANUAL, motivo)"),
    m(A, E, "derivar a inspección manual (sin accionar el desviador)", True),
    ("else", "percepción válida y cámara OK"),
    m(D, A, "(flujo típico: CONTINUAR o DESVIAR)"),
    ("fin",),
    m(A, R, "registrar(motivo, estado_camara, REVISION_MANUAL)"),
    ("ini", "alt", "estado_camara == EN_FALLA"),
    m(A, I, "alerta: CAMARA_EN_FALLA", True),
    m(I, O, "mostrar alerta y unidades derivadas", True),
    ("fin",),
    ("ini", "alt", "2 percepciones válidas seguidas (histéresis)"),
    m(P, P, "estado_camara = OK; se reanuda la clasificación"),
    ("fin",),
    ("fin",),
]


# --------------------------------------------------------------------------- #
# Diagrama de arquitectura (componentes)
# --------------------------------------------------------------------------- #
def arquitectura():
    fig, ax = plt.subplots(figsize=(15, 8.6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0.2, 8.6)
    ax.axis("off")

    def caja(x, y, w, h, texto, fc=AZUL_CL, fs=8.4, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=fc, ec=AZUL, lw=1.3))
        ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal")

    def flecha(x0, y0, x1, y1, texto="", dy=0.12, ha="center"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.3))
        if texto:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, texto, ha=ha, va="bottom", fontsize=7.6,
                    bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.9))

    ax.text(7.5, 8.3, "Arquitectura de componentes de FreshCheck-Agent", ha="center", fontsize=13, fontweight="bold", color=AZUL)

    # Entorno
    ax.add_patch(Rectangle((0.2, 1.0), 3.0, 6.7, fc="#F6F6F6", ec="#888888", lw=1.2, ls="--"))
    ax.text(1.7, 7.5, "ENTORNO (simulado)", ha="center", fontsize=9, fontweight="bold", color="#444444")
    caja(0.4, 5.6, 2.6, 1.3, "LineaEmpaqueSimulada\nsimulator.py\n(banda, lotes, estocástico)", NARANJA_CL)
    caja(0.4, 3.9, 2.6, 1.2, "Sensores simulados\nimagen RGB · disparo\nvelocidad · lote", NARANJA_CL)
    caja(0.4, 1.3, 2.6, 1.7, "Efecto sobre la banda\ndestino de cada unidad:\nempaque_final · descarte\ninspeccion_manual", NARANJA_CL)

    # Agente
    ax.add_patch(Rectangle((3.7, 1.0), 7.1, 6.7, fc="#FAFCFF", ec=AZUL, lw=1.8))
    ax.text(7.25, 7.5, "FreshCheckAgent  (agent.py)", ha="center", fontsize=9.5, fontweight="bold", color=AZUL)
    caja(3.95, 5.3, 2.5, 1.7, "percibir()\nperception.py\nvalidar · normalizar\nsegmentar · características", AZUL_CL, bold=False)
    caja(8.1, 5.3, 2.5, 1.7, "decidir()\nregla condición-acción\nDESVIAR si clase = danada\nREVISION_MANUAL si falla", AZUL_CL)
    caja(8.1, 1.3, 2.5, 1.7, "actuar()\nenvía la orden al\ndesviador y calcula\nel plazo de decisión", AZUL_CL)
    caja(3.95, 3.0, 2.5, 1.7, "ModeloDelMundo\n(estado interno)\nbanda · cámara · velocidad\ncontadores por lote", VERDE_CL, bold=True)
    caja(5.5, 1.3, 2.3, 1.3, "Clasificador\nmodel.py\nHistGradientBoosting", VERDE_CL, fs=7.8)
    caja(3.95, 1.3, 1.4, 1.3, "learning.py\nreentrenar()", VERDE_CL, fs=7.8)

    # Servicios externos
    ax.add_patch(Rectangle((11.4, 1.0), 3.4, 6.7, fc="#F6F6F6", ec="#888888", lw=1.2, ls="--"))
    ax.text(13.1, 7.5, "SERVICIOS EXTERNOS", ha="center", fontsize=9, fontweight="bold", color="#444444")
    caja(11.6, 5.6, 3.0, 1.3, "InterfazDesviador\nactuator.py\n(PLC en producción real)", NARANJA_CL)
    caja(11.6, 3.9, 3.0, 1.2, "Registro (log CSV)\nlogs/pruebas/*.csv", NARANJA_CL)
    caja(11.6, 2.3, 3.0, 1.2, "Reportes por proveedor\nreports.py", NARANJA_CL)
    caja(11.6, 1.2, 3.0, 0.8, "Interfaz: app.py (Streamlit)", NARANJA_CL, fs=8)

    # Flechas
    flecha(1.7, 5.6, 1.7, 5.1, "")
    flecha(3.0, 4.5, 3.95, 5.8, "percepto", dy=0.05)
    flecha(6.45, 6.15, 8.1, 6.15, "Percepcion")
    flecha(4.9, 4.7, 4.9, 5.3, "")
    flecha(5.9, 5.3, 6.4, 2.6, "características", dy=0.0, ha="left")
    flecha(7.8, 1.95, 8.1, 5.3, "")
    flecha(9.35, 5.3, 9.35, 3.0, "Decision")
    flecha(10.6, 2.2, 11.6, 6.0, "accionar()", dy=0.0, ha="right")
    ax.plot([9.35, 9.35, 1.7, 1.7], [1.3, 0.55, 0.55, 1.3], color="#222222", lw=1.3)
    ax.annotate("", xy=(1.7, 1.3), xytext=(1.7, 0.9), arrowprops=dict(arrowstyle="-|>", color="#222222", lw=1.3))
    ax.text(5.5, 0.62, "efecto sobre la banda (destino de la unidad)", ha="center", va="bottom", fontsize=7.6,
            bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.9))
    flecha(10.6, 2.8, 11.6, 4.5, "fila de log", dy=0.0, ha="left")
    flecha(13.1, 3.9, 13.1, 3.5, "")
    flecha(4.65, 2.6, 4.65, 3.0, "")
    flecha(5.35, 1.95, 5.5, 1.95, "")
    fig.savefig(FIG / "arquitectura.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


PLANTUML_TIPICO = """@startuml
title FreshCheck-Agent - escenario tipico
actor Operador
participant "Interfaz (CLI/Streamlit)" as I
participant "Entorno (linea simulada)" as E
participant "Agente: Percepcion" as P
participant "Agente: Decision" as D
participant "Agente: Accion" as A
participant "Desviador (actuador)" as V
database "Registro y reportes" as R
Operador -> I : iniciar_simulacion(escenario, n, seed)
I -> E : flujo(n)
loop por cada unidad
  E -> P : Percepto(imagen, t_s, lote, velocidad)
  P -> P : validar, normalizar, segmentar, caracteristicas
  P -> P : ModeloDelMundo.actualizar()
  P -> D : Percepcion(OK, caracteristicas)
  D -> D : predecir_proba(x)
  alt clase == danada
    D -> A : Decision(DESVIAR)
    A -> V : accionar(unit_id, t_act, latencia, plazo)
    alt latencia <= plazo y desviador libre
      V --> A : ejecutado=True (descarte)
    else DECISION_TARDIA o DESVIADOR_OCUPADO
      V --> A : ejecutado=False (empaque_final)
    end
  else fresca o maduracion
    D -> A : Decision(CONTINUAR)
  end
  A -> R : registrar(percepcion, decision, accion)
  A --> E : efecto(destino)
end
I -> R : reporte_por_proveedor()
R --> I : tabla
I --> Operador : decisiones, metricas, reporte
@enduml
"""

PLANTUML_EXCEPCION = """@startuml
title FreshCheck-Agent - excepcion: sensor sin dato y camara en falla
actor Operador
participant "Interfaz" as I
participant "Entorno" as E
participant "Agente: Percepcion" as P
participant "Agente: Decision" as D
participant "Agente: Accion" as A
database "Registro" as R
Operador -> I : iniciar_simulacion('F10_rafaga')
I -> E : flujo(n)
loop unidades 10 a 16
  E -> P : Percepto(imagen=None)
  P -> P : validar_imagen() -> SENSOR_SIN_DATO
  P -> P : invalidos_consecutivos += 1
  alt invalidos_consecutivos >= 3
    P -> P : estado_camara = EN_FALLA
  else menos de 3
    P -> P : estado_camara = DEGRADADA
  end
  P -> D : Percepcion(SENSOR_SIN_DATO)
  alt codigo != OK o camara EN_FALLA
    D -> A : Decision(REVISION_MANUAL, motivo)
    A --> E : derivar a inspeccion manual
  end
  A -> R : registrar(motivo, estado_camara)
  alt estado_camara == EN_FALLA
    A --> I : alerta CAMARA_EN_FALLA
    I --> Operador : mostrar alerta
  end
  alt 2 percepciones validas seguidas
    P -> P : estado_camara = OK
  end
end
@enduml
"""


def main():
    secuencia("Diagrama de secuencia UML — escenario típico (clasificación y desvío de una unidad)",
              PART, TIPICO, "secuencia_tipico.png")
    secuencia("Diagrama de secuencia UML — escenario de excepción (sensor sin dato, cámara en falla y recuperación)",
              PART, EXCEPCION, "secuencia_excepcion.png")
    arquitectura()
    (UML / "secuencia_tipico.puml").write_text(PLANTUML_TIPICO, encoding="utf-8")
    (UML / "secuencia_excepcion.puml").write_text(PLANTUML_EXCEPCION, encoding="utf-8")
    print("Diagramas generados en", FIG)


if __name__ == "__main__":
    main()
