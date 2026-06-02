"""Predicao de uma imagem com o modelo treinado."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

DEFAULT_CLASSES = ["algas", "anomalia_termica", "nuvens", "oceano_normal", "tempestade"]


def load_class_names(path: str | None = None):
    candidates = [Path(path)] if path else [Path("models/class_names.json"), Path("reports/class_names.json")]
    for candidate in candidates:
        if candidate and candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8")).get("classes", DEFAULT_CLASSES)
    return DEFAULT_CLASSES


def model_image_size(model, fallback: int):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    if len(shape) >= 3 and shape[1] and shape[2]:
        return int(shape[1]), int(shape[2])
    return fallback, fallback


def predict(model_path: str, image_path: str, image_size: int | None = None, class_names=None):
    class_names = class_names or load_class_names()
    model = tf.keras.models.load_model(model_path)
    fallback = image_size or 96
    img = Image.open(image_path).convert("RGB").resize(model_image_size(model, fallback))
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
    parser.add_argument("--model", default="models/oceanmind_cnn_v1.keras")
    parser.add_argument("--image", required=True)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--classes-json", default=None)
    args = parser.parse_args()
    result = predict(args.model, args.image, args.image_size, load_class_names(args.classes_json))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
