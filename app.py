from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

DEFAULT_CLASSES = ["algas", "anomalia_termica", "nuvens", "oceano_normal", "tempestade"]
MODEL_OPTIONS = {
    "CNN v1": "models/oceanmind_cnn_v1.keras",
    "CNN v2": "models/oceanmind_cnn_v2.keras",
}

st.set_page_config(page_title="OceanMind ACV", layout="wide")
st.title("OceanMind - Visao Computacional")
st.caption("Classificacao de imagens oceanicas/satelitais com CNNs treinadas do zero")

st.markdown(
    """
Este prototipo demonstra o modulo de Applied Computer Vision do OceanMind.
A CNN classifica imagens relacionadas ao monitoramento oceanico em categorias ambientais,
apoiando a identificacao de padroes visuais como anomalias, nuvens, tempestades e algas.
"""
)

col_info_1, col_info_2, col_info_3 = st.columns(3)
col_info_1.metric("Arquiteturas", "2 CNNs")
col_info_2.metric("Pre-treinamento", "Nao usado")
col_info_3.metric("Demonstracao", "Upload de imagem")

st.divider()
model_label = st.selectbox("Modelo", list(MODEL_OPTIONS.keys()), index=0)
model_path = Path(MODEL_OPTIONS[model_label])
classes_path = Path("models/class_names.json")
if classes_path.exists():
    CLASSES = json.loads(classes_path.read_text(encoding="utf-8")).get("classes", DEFAULT_CLASSES)
else:
    CLASSES = DEFAULT_CLASSES

uploaded = st.file_uploader("Envie uma imagem oceanica/satelital", type=["png", "jpg", "jpeg"])


@st.cache_resource
def load_model(path: str):
    return tf.keras.models.load_model(path)


def model_image_size(model):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    return int(shape[1]), int(shape[2])


if not model_path.exists():
    st.warning("Modelo treinado nao encontrado. Rode primeiro: `python src/train.py --dataset dataset --epochs 12`.")
else:
    model = load_model(str(model_path))
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        preview, result = st.columns([1, 1])
        with preview:
            st.image(img, caption="Imagem enviada", use_container_width=True)
        resized = img.resize(model_image_size(model))
        arr = np.expand_dims(np.asarray(resized), axis=0)
        probs = model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(probs))
        pred_class = CLASSES[idx]
        confidence = float(probs[idx])
        with result:
            st.subheader("Resultado da CNN")
            st.metric("Classificacao", pred_class.replace("_", " ").title())
            st.metric("Confianca", f"{confidence:.2%}")
            st.write("Probabilidades por classe")
            st.bar_chart({CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))})
    else:
        st.info("Envie uma imagem para realizar a classificacao.")

st.divider()
st.subheader("Classes do modelo")
st.write(", ".join([c.replace("_", " ").title() for c in CLASSES]))
