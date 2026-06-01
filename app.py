from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

CLASSES = ["algas", "anomalia_termica", "nuvens", "oceano_normal", "tempestade"]
MODEL_OPTIONS = {
    "CNN v1": "models/oceanmind_cnn_v1.keras",
    "CNN v2": "models/oceanmind_cnn_v2.keras",
}

st.set_page_config(page_title="OceanMind ACV", page_icon="🌊", layout="wide")
st.title("🌊 OceanMind — Visão Computacional")
st.caption("Classificação de imagens oceânicas/satelitais com CNNs treinadas do zero")

st.markdown(
    """
Este protótipo demonstra o módulo de **Applied Computer Vision** do OceanMind.
A CNN classifica imagens relacionadas ao monitoramento oceânico em categorias ambientais, apoiando a identificação de padrões visuais como anomalias, nuvens, tempestades e algas.
"""
)

col_info_1, col_info_2, col_info_3 = st.columns(3)
col_info_1.metric("Arquiteturas", "2 CNNs")
col_info_2.metric("Pré-treinamento", "Não usado")
col_info_3.metric("Demonstração", "Upload de imagem")

st.divider()
model_label = st.selectbox("Modelo", list(MODEL_OPTIONS.keys()), index=1)
model_path = Path(MODEL_OPTIONS[model_label])

uploaded = st.file_uploader("Envie uma imagem oceânica/satelital", type=["png", "jpg", "jpeg"])

@st.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path)

if not model_path.exists():
    st.warning("Modelo treinado não encontrado. Rode primeiro: `python src/train.py --dataset dataset --epochs 12`.")
else:
    model = load_model(str(model_path))
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        preview, result = st.columns([1, 1])
        with preview:
            st.image(img, caption="Imagem enviada", use_container_width=True)
        resized = img.resize((96, 96))
        arr = np.expand_dims(np.asarray(resized), axis=0)
        probs = model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(probs))
        pred_class = CLASSES[idx]
        confidence = float(probs[idx])
        with result:
            st.subheader("Resultado da CNN")
            st.metric("Classificação", pred_class.replace("_", " ").title())
            st.metric("Confiança", f"{confidence:.2%}")
            st.write("Probabilidades por classe")
            st.bar_chart({CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))})
    else:
        st.info("Envie uma imagem para realizar a classificação.")

st.divider()
st.subheader("Classes do modelo")
st.write(", ".join([c.replace("_", " ").title() for c in CLASSES]))
