"""Treinamento das duas CNNs do OceanMind."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from models import build_cnn_v1, build_cnn_v2


def plot_history(history, title: str, out_path: Path):
    plt.figure(figsize=(9, 5))
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title(title)
    plt.xlabel("Epoca")
    plt.ylabel("Valor")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_confusion_matrix(cm, class_names, title: str, out_path: Path):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.ylabel("Classe real")
    plt.xlabel("Classe prevista")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _load_directory(path: Path, image_size: int, batch_size: int, shuffle: bool, seed: int):
    return tf.keras.utils.image_dataset_from_directory(
        path,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=shuffle,
        seed=seed,
    )


def make_datasets(dataset_dir: str, image_size: int, batch_size: int, seed: int):
    dataset_path = Path(dataset_dir)
    split_dirs = [dataset_path / "train", dataset_path / "validation", dataset_path / "test"]
    if all(path.exists() for path in split_dirs):
        train_ds = _load_directory(dataset_path / "train", image_size, batch_size, True, seed)
        val_ds = _load_directory(dataset_path / "validation", image_size, batch_size, False, seed)
        test_ds = _load_directory(dataset_path / "test", image_size, batch_size, False, seed)
        class_names = train_ds.class_names
    else:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_dir,
            validation_split=0.30,
            subset="training",
            seed=seed,
            image_size=(image_size, image_size),
            batch_size=batch_size,
            label_mode="categorical",
        )
        temp_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_dir,
            validation_split=0.30,
            subset="validation",
            seed=seed,
            image_size=(image_size, image_size),
            batch_size=batch_size,
            label_mode="categorical",
        )
        class_names = train_ds.class_names

        # Divide validacao temporaria em validacao e teste.
        val_batches = tf.data.experimental.cardinality(temp_ds).numpy()
        test_ds = temp_ds.take(max(1, val_batches // 2))
        val_ds = temp_ds.skip(max(1, val_batches // 2))

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=seed).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names


def count_split_images(dataset_dir: str) -> dict:
    dataset_path = Path(dataset_dir)
    split_dirs = [dataset_path / "train", dataset_path / "validation", dataset_path / "test"]
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    if all(path.exists() for path in split_dirs):
        return {
            split.name: {
                class_dir.name: len([p for p in class_dir.iterdir() if p.suffix.lower() in image_exts])
                for class_dir in sorted(split.iterdir())
                if class_dir.is_dir()
            }
            for split in split_dirs
        }
    return {
        "single_directory_with_validation_split": {
            class_dir.name: len([p for p in class_dir.iterdir() if p.suffix.lower() in image_exts])
            for class_dir in sorted(dataset_path.iterdir())
            if class_dir.is_dir()
        }
    }


def copy_dataset_metadata(dataset_dir: str, reports_dir: Path):
    src = Path(dataset_dir) / "dataset_metadata.json"
    if src.exists():
        shutil.copy2(src, reports_dir / "dataset_metadata.json")


def save_run_metadata(dataset_dir: str, image_size: int, batch_size: int, epochs: int, seed: int, class_names: list[str], reports_dir: Path):
    metadata = {
        "dataset": dataset_dir,
        "image_size": image_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
        "classes": class_names,
        "split_counts": count_split_images(dataset_dir),
        "architectures": ["cnn_v1", "cnn_v2"],
        "pretrained_models": False,
    }
    with open(reports_dir / "training_run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def save_class_names(class_names: list[str], reports_dir: Path):
    payload = {"classes": class_names}
    with open("models/class_names.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(reports_dir / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _legacy_make_datasets(dataset_dir: str, image_size: int, batch_size: int, seed: int):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.30,
        subset="training",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="categorical",
    )
    temp_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.30,
        subset="validation",
        seed=seed,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="categorical",
    )
    class_names = train_ds.class_names

    # Divide validacao temporaria em validacao e teste.
    val_batches = tf.data.experimental.cardinality(temp_ds).numpy()
    test_ds = temp_ds.take(max(1, val_batches // 2))
    val_ds = temp_ds.skip(max(1, val_batches // 2))

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000, seed=seed).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    test_ds = test_ds.cache().prefetch(autotune)
    return train_ds, val_ds, test_ds, class_names


def evaluate_model(model, test_ds, class_names, model_name: str, reports_dir: Path):
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, f"Matriz de Confusao - {model_name}", reports_dir / f"confusion_matrix_{model_name}.png")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    with open(reports_dir / f"classification_report_{model_name}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(args.seed)
    Path("models").mkdir(exist_ok=True)
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    train_ds, val_ds, test_ds, class_names = make_datasets(args.dataset, args.image_size, args.batch_size, args.seed)
    num_classes = len(class_names)
    save_class_names(class_names, reports_dir)
    save_run_metadata(args.dataset, args.image_size, args.batch_size, args.epochs, args.seed, class_names, reports_dir)
    copy_dataset_metadata(args.dataset, reports_dir)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.5),
    ]

    results = {}
    for name, builder in [("cnn_v1", build_cnn_v1), ("cnn_v2", build_cnn_v2)]:
        print(f"\nTreinando {name}...")
        model = builder(input_shape=(args.image_size, args.image_size, 3), num_classes=num_classes)
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
        test_loss, test_acc = model.evaluate(test_ds, verbose=0)
        model.save(f"models/oceanmind_{name}.keras")
        plot_history(history, f"Treinamento - {name}", reports_dir / f"training_history_{name}.png")
        report = evaluate_model(model, test_ds, class_names, name, reports_dir)
        results[name] = {
            "test_accuracy": float(test_acc),
            "test_loss": float(test_loss),
            "classification_report": report,
        }

    with open(reports_dir / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump({"classes": class_names, "dataset": args.dataset, "results": results}, f, ensure_ascii=False, indent=2)

    best = max(results.items(), key=lambda x: x[1]["test_accuracy"])[0]
    print("\nTreinamento concluido.")
    print("Classes:", class_names)
    print("Melhor modelo:", best)
    print("Resultados salvos em models/ e reports/")


if __name__ == "__main__":
    main()
