"""Predição de uma imagem com o modelo treinado."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

DEFAULT_CLASSES = ["algas", "anomalia_termica", "nuvens", "oceano_normal", "tempestade"]


def predict(model_path: str, image_path: str, image_size: int = 96, class_names=None):
    class_names = class_names or DEFAULT_CLASSES
    model = tf.keras.models.load_model(model_path)
    img = Image.open(image_path).convert("RGB").resize((image_size, image_size))
    arr = np.expand_dims(np.asarray(img), axis=0)
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return {
        "image": image_path,
        "class": class_names[idx],
        "confidence": float(probs[idx]),
        "probabilities": {class_names[i]: float(probs[i]) for i in range(len(class_names))},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/oceanmind_cnn_v2.keras")
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-size", type=int, default=96)
    args = parser.parse_args()
    result = predict(args.model, args.image, args.image_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
