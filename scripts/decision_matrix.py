"""
Fase 1 - Matriz de decisión ponderada entre FreshCheck-Agent (A) y RetainIQ-Agent (B).

    python scripts/decision_matrix.py

Genera analysis/matriz_decision.csv, analysis/sensibilidad.csv y analysis/matriz_decision.json
a partir de los pesos y puntajes definidos aquí. Cualquier persona que use los mismos pesos y
puntajes obtiene el mismo resultado: S = suma(w_i * p_i).
"""
import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "analysis"
SALIDA.mkdir(exist_ok=True)

CRITERIOS = [
    # nombre, peso, puntaje A, evidencia A, puntaje B, evidencia B
    ("Viabilidad técnica", 0.25,
     4, "5 de 7 componentes IBM con herramientas ya conocidas (Python, pandas, scikit-learn); percepción visual y "
        "razonamiento sobre imágenes requieren OpenCV y un simulador de imágenes (5/7 = 0,71).",
     4, "5 de 7 con pandas y scikit-learn; la memoria de trayectorias y la recalibración trimestral exigen series "
        "mensuales que el dataset IBM HR (transversal) no trae y habría que simular (5/7 = 0,71)."),
    ("Disponibilidad de datos o entorno", 0.20,
     4, "Dataset concreto identificado (FruitVision) más fotos propias, y entorno simulable con imágenes etiquetadas "
        "ilimitadas; falta la banda física.",
     3, "Dataset concreto y descargable (IBM HR, 1.470 empleados), pero transversal y ficticio: no modela el flujo "
        "mensual a 3 meses del diseño."),
    ("Medibilidad del desempeño", 0.20,
     4, "2 de 2 medidas con umbral numérico (exactitud ≥ 95 %, reducción ≥ 30 %); la reducción carecía de línea base y "
        "fórmula, que se definen en el ajuste.",
     3, "1 medida con umbral (precisión ≥ 70 %), pero el horizonte de 3 meses y la evaluación trimestral no son "
        "verificables sin dimensión temporal; habría que reformularla."),
    ("Complejidad de IA", 0.15,
     3, "Agente reflejo basado en modelo (R&N) con percepción aprendida y reentrenamiento offline; sin búsqueda, "
        "planificación ni función de utilidad.",
     5, "Agente de aprendizaje con elemento de desempeño basado en utilidad y estado interno (trayectorias): cubre "
        "tres tipos de la jerarquía de R&N."),
    ("Costo y recursos", 0.10,
     5, "0 COP: Python, OpenCV y scikit-learn gratuitos, sin APIs de pago, CPU suficiente.",
     5, "0 COP: CSV pequeño y scikit-learn, sin APIs de pago."),
    ("Valor del problema", 0.10,
     4, "Usuario: personal de calidad de la línea; beneficio: menos unidades defectuosas en el empaque final "
        "(contexto piloto hipotético).",
     4, "Usuario: líderes de equipo y RR.HH.; beneficio: anticipar renuncias a 3 meses (el agente solo informa, "
        "por restricciones éticas)."),
]


def puntaje(pesos, col):
    return sum(w * p for w, p in zip(pesos, col))


def main():
    nombres = [c[0] for c in CRITERIOS]
    w = [c[1] for c in CRITERIOS]
    pa = [c[2] for c in CRITERIOS]
    pb = [c[4] for c in CRITERIOS]
    assert abs(sum(w) - 1.0) < 1e-9

    sa, sb = puntaje(w, pa), puntaje(w, pb)
    filas = [dict(criterio=n, peso=wi, A_puntaje=a, A_ponderado=round(wi * a, 4), B_puntaje=b,
                  B_ponderado=round(wi * b, 4), diferencia_AB=round(wi * (a - b), 4))
             for n, wi, a, b in zip(nombres, w, pa, pb)]
    pd.DataFrame(filas).to_csv(SALIDA / "matriz_decision.csv", index=False)

    # Criterio más influyente = mayor |w * (pA - pB)|
    influencia = [abs(wi * (a - b)) for wi, a, b in zip(w, pa, pb)]
    k = max(range(len(w)), key=lambda i: influencia[i])

    def reescalar(pesos, i, nuevo):
        resto = 1.0 - pesos[i]
        factor = (1.0 - nuevo) / resto
        return [nuevo if j == i else pj * factor for j, pj in enumerate(pesos)]

    sens = []
    for i, n in enumerate(nombres):
        for delta in (-0.10, 0.10):
            nuevo = w[i] + delta
            if nuevo < 0 or nuevo > 1:
                continue
            ww = reescalar(w, i, nuevo)
            a_, b_ = puntaje(ww, pa), puntaje(ww, pb)
            sens.append(dict(criterio=n, cambio_peso=f"{delta:+.0%}", peso_nuevo=round(nuevo, 4),
                             S_A=round(a_, 4), S_B=round(b_, 4), elegido="A" if a_ > b_ else ("B" if b_ > a_ else "empate")))
    pd.DataFrame(sens).to_csv(SALIDA / "sensibilidad.csv", index=False)

    # Peso de equilibrio del criterio más influyente
    equilibrio = None
    for x in [i / 1000 for i in range(0, 1001)]:
        ww = reescalar(w, k, x)
        if puntaje(ww, pb) > puntaje(ww, pa):
            equilibrio = x
            break

    resumen = dict(S_A=round(sa, 4), S_B=round(sb, 4), elegido="A (FreshCheck-Agent)" if sa > sb else "B (RetainIQ-Agent)",
                   criterio_mas_influyente=nombres[k], peso_base=w[k], peso_equilibrio=equilibrio,
                   sensibilidad_mas_influyente=[s for s in sens if s["criterio"] == nombres[k]])
    (SALIDA / "matriz_decision.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(pd.DataFrame(filas).to_string(index=False))
    print(f"\nS(A) = {sa:.4f}   S(B) = {sb:.4f}   -> {resumen['elegido']}")
    print(f"Criterio más influyente: {nombres[k]} (peso {w[k]:.0%}); B supera a A si su peso es >= {equilibrio}")
    print(pd.DataFrame(sens).to_string(index=False))


if __name__ == "__main__":
    main()
