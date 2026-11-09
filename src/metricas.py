"""Metricas de evaluacion (definidas en el Hito 1)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

from .config import CLASES


def metricas(y_true, y_pred, clases=CLASES) -> dict:
    """Exactitud, macro-F1 (solo sobre clases presentes en y_true), P/R/F1 por clase y matriz de confusion."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    presentes = [c for c in clases if (y_true == c).any()]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(clases), zero_division=0)
    por_clase = {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "soporte": int(s[i])}
                 for i, c in enumerate(clases)}
    return {
        "n": int(len(y_true)),
        "exactitud": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=presentes, average="macro", zero_division=0)),
        "clases_presentes": presentes,
        "recall_danada": por_clase["danada"]["recall"] if por_clase["danada"]["soporte"] > 0 else None,
        "por_clase": por_clase,
        "matriz_confusion": confusion_matrix(y_true, y_pred, labels=list(clases)).tolist(),
    }


def guardar_matriz_confusion(cm, clases, ruta, titulo="Matriz de confusion"):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(clases)), clases, rotation=30)
    ax.set_yticks(range(len(clases)), clases)
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    ax.set_title(titulo, fontsize=10)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def estado_umbrales(m_todas: dict, m_fv: dict | None, m_int: dict | None, umbrales: dict) -> list[dict]:
    """Compara con los umbrales del Hito 1. En el Hito 2 se evalua en VALIDACION, por lo tanto es indicativo."""
    def fila(nombre, valor):
        u = umbrales[nombre]
        cumple = None if valor is None else bool(valor >= u["umbral"])
        return {"metrica": nombre, "valor": valor, "umbral": u["umbral"], "cumple": cumple}
    return [
        fila("macro_f1_prueba_todas_fuentes", m_todas["macro_f1"]),
        fila("exactitud_prueba_fruitvision", None if m_fv is None else m_fv["exactitud"]),
        fila("macro_f1_prueba_internet", None if m_int is None else m_int["macro_f1"]),
        fila("recall_danada_prueba", m_todas["recall_danada"]),
    ]
