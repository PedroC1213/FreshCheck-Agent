"""Prototipo web (Streamlit) del proyecto final: clasifica una foto de fruta como fresca o dañada.

Uso:  streamlit run app_v2.py
"""
import streamlit as st
from PIL import Image

from src import predecir

st.set_page_config(page_title="FreshCheck-Agent v2", page_icon="🍎", layout="centered")
st.title("FreshCheck-Agent v2")
st.caption("Clasificación del estado de una fruta (fresca o dañada) con visión por computador.")


@st.cache_resource(show_spinner="Cargando el modelo…")
def _modelo(nombre):
    return predecir.cargar_modelo(nombre)


disponibles = list(predecir.modelos_disponibles())
if not disponibles:
    st.error("No hay modelos en `models_v2/`. Ejecute `python -m src.baseline_final` (línea base) o `python -m src.run_final`.")
    st.stop()

with st.sidebar:
    eleccion = st.selectbox("Modelo", ["auto"] + disponibles,
                            help="«auto» usa el mejor modelo ajustado si existe; si no, la línea base.")
    st.markdown(predecir.AVISO)

archivo = st.file_uploader("Suba una foto de la fruta", type=["jpg", "jpeg", "png", "webp"])
foto = st.camera_input("…o tome una foto") if archivo is None else None
origen = archivo or foto

if origen is not None:
    imagen = Image.open(origen)
    st.image(imagen, caption="Imagen recibida", use_container_width=True)
    try:
        r = _modelo(eleccion).predecir(imagen)
    except Exception as e:  # mensaje claro en la demostración
        st.error(f"No se pudo clasificar la imagen: {e}")
        st.stop()
    etiqueta = "DAÑADA" if r["clase"] == "danada" else "FRESCA"
    (st.error if r["clase"] == "danada" else st.success)(f"**{etiqueta}**  (confianza {r['confianza']:.0%})")
    st.progress(min(1.0, max(0.0, r["p_danada"])), text=f"Probabilidad de daño: {r['p_danada']:.1%}")
    st.caption(f"Modelo: {r['modelo']}")
    for adv in r["advertencias"]:
        st.warning(adv)
