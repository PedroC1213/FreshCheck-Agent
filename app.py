"""
Interfaz de demostración de FreshCheck-Agent.

    streamlit run app.py

Pestañas:
  1. Simulación de la línea: ejecuta un escenario y muestra decisiones, métricas y reporte por proveedor.
  2. Caso de falla: inyecta una falla de percepción y muestra cómo responde el agente.
  3. Imagen propia: clasifica una fotografía subida por el usuario.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from freshcheck.actuator import DesviadorSimulado
from freshcheck.agent import FreshCheckAgent
from freshcheck.config import IMG_SIZE
from freshcheck.evaluation import (m1_exactitud, m2_reduccion_defectuosas, m3_descarte_indebido,
                                   m4_latencia)
from freshcheck.model import ClasificadorEstado
from freshcheck.reports import reporte_desde_agente
from freshcheck.scenarios import FALLAS, LIMITE, NOMINALES, por_nombre
from freshcheck.simulator import LineaEmpaqueSimulada, Percepto

RAIZ = Path(__file__).resolve().parent
RUTA_MODELO = RAIZ / "models" / "freshcheck_model.joblib"
COLOR = {"CONTINUAR": (60, 160, 60), "DESVIAR": (40, 40, 220), "REVISION_MANUAL": (30, 150, 240), "NINGUNA": (150, 150, 150)}

st.set_page_config(page_title="FreshCheck-Agent", layout="wide")


@st.cache_resource
def cargar_modelo() -> ClasificadorEstado:
    if not RUTA_MODELO.exists():
        st.error("No se encontró models/freshcheck_model.joblib. Ejecute `python scripts/train.py` primero.")
        st.stop()
    return ClasificadorEstado.cargar(RUTA_MODELO)


def anotar(img, fila) -> np.ndarray:
    if not isinstance(img, np.ndarray) or img.ndim != 3 or img.dtype != np.uint8 or img.size == 0:
        vacia = np.full((IMG_SIZE, IMG_SIZE, 3), 70, np.uint8)
        cv2.putText(vacia, "sin imagen", (22, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        img = vacia
    out = cv2.copyMakeBorder(img, 4, 28, 4, 4, cv2.BORDER_CONSTANT, value=COLOR[fila["accion"]])
    texto = f"{fila['clase_predicha'] or fila['motivo_decision']} -> {fila['accion']}"
    cv2.putText(out, texto[:26], (6, out.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def correr(clf, esc, n, seed, max_imagenes=12):
    linea = LineaEmpaqueSimulada(esc, seed)
    agente = FreshCheckAgent(clf, DesviadorSimulado())
    filas, galeria = [], []
    for p in linea.flujo(n):
        fila = agente.paso(p)
        v = linea.verdad[p.unit_id]
        fila.update(clase_real=v.clase or "", falla_inyectada=v.falla or "")
        filas.append(fila)
        if len(galeria) < max_imagenes:
            galeria.append(anotar(p.imagen, fila))
    return pd.DataFrame(filas), galeria, agente


clf = cargar_modelo()
st.title("FreshCheck-Agent — inspección de calidad de frutas en línea de empaque")
st.caption("Agente reflejo basado en modelo. Entorno simulado: imágenes sintéticas, banda con dinámica y fallas de percepción.")

tab1, tab2, tab3 = st.tabs(["Simulación de la línea", "Caso de falla", "Imagen propia"])

with tab1:
    esc_lista = list(NOMINALES) + list(LIMITE)
    c1, c2, c3 = st.columns(3)
    nombre = c1.selectbox("Escenario", [e.nombre for e in esc_lista])
    n = c2.slider("Unidades", 20, 300, 100, 10)
    seed = c3.number_input("Semilla", 0, 10_000_000, 12345)
    st.caption(por_nombre()[nombre].descripcion)
    if st.button("Ejecutar simulación", type="primary"):
        df, galeria, agente = correr(clf, por_nombre()[nombre], n, int(seed))
        r2, d0, d1 = m2_reduccion_defectuosas(df)
        p95, tard = m4_latencia(df)
        m = st.columns(5)
        m[0].metric("Exactitud (M1)", f"{m1_exactitud(df):.1%}")
        m[1].metric("Reducción de defectuosas (M2)", f"{r2:.1%}", f"{d0} → {d1}")
        m[2].metric("Descarte indebido (M3)", f"{m3_descarte_indebido(df):.1%}")
        m[3].metric("Latencia p95 (M4)", f"{p95:.1f} ms")
        m[4].metric("Decisiones tardías", tard)
        st.subheader("Primeras unidades procesadas")
        cols = st.columns(6)
        for i, g in enumerate(galeria):
            cols[i % 6].image(g, width="stretch")
        st.subheader("Reporte de calidad por lote y proveedor")
        st.dataframe(reporte_desde_agente(agente), width="stretch")
        st.subheader("Registro de ejecución (log)")
        st.dataframe(df, width="stretch", height=300)
        st.download_button("Descargar log (CSV)", df.to_csv(index=False).encode("utf-8"), "log_simulacion.csv")

with tab2:
    esc_f = {e.nombre: e for e in FALLAS}
    nombre_f = st.selectbox("Falla a inyectar", list(esc_f))
    st.caption(esc_f[nombre_f].descripcion)
    if st.button("Inyectar falla"):
        df, galeria, agente = correr(clf, esc_f[nombre_f], 20 if nombre_f.startswith("F10") else 60, 4242, max_imagenes=60)
        inicio = 9
        sub = df.iloc[inicio:18]
        cols = st.columns(9)
        for i, g in enumerate(galeria[inicio:18]):
            cols[i % 9].image(g, width="stretch")
        st.dataframe(sub[["unit_id", "estado_percepcion", "estado_camara", "clase_predicha", "accion",
                          "motivo_decision", "destino"]], width="stretch")
        st.info("El agente nunca se detiene: cada percepción inválida se registra y se deriva a inspección manual.")

with tab3:
    st.warning("El modelo incluido se entrenó con imágenes sintéticas. Para fotografías reales reentrene con "
               "`python scripts/train.py --real-data RUTA` (carpetas fresca/, maduracion/, danada/).")
    arch = st.file_uploader("Fotografía de una fruta sobre fondo uniforme", type=["jpg", "jpeg", "png"])
    if arch is not None:
        datos = np.frombuffer(arch.read(), np.uint8)
        im = cv2.imdecode(datos, cv2.IMREAD_COLOR)
        if im is None:
            st.error("No se pudo leer la imagen.")
        else:
            im = cv2.resize(im, (IMG_SIZE, IMG_SIZE))
            agente = FreshCheckAgent(clf, DesviadorSimulado())
            fila = agente.paso(Percepto(0.0, 0, "L-USR", "Usuario", im, 0.5))
            st.image(anotar(im, fila), caption="Resultado", width=260)
            st.json({k: fila[k] for k in ("estado_percepcion", "clase_predicha", "p_fresca", "p_maduracion",
                                            "p_danada", "accion", "motivo_decision", "latencia_ms")})
