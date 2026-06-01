"""
OceanMind ACV - Gerador de dataset sintético de imagens oceânicas.

Este script cria imagens simples e controladas para demonstrar o pipeline de
Visão Computacional da Global Solution. As imagens simulam padrões visuais
relacionados ao monitoramento oceânico por satélite.

Classes:
- oceano_normal
- anomalia_termica
- tempestade
- nuvens
- algas

Uso:
python src/generate_synthetic_ocean_dataset.py --output dataset --images-per-class 220 --size 96
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CLASSES = ["oceano_normal", "anomalia_termica", "tempestade", "nuvens", "algas"]


def _base_ocean(size: int, rng: random.Random) -> Image.Image:
    """Cria fundo oceânico com gradiente e ruído leve."""
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    base_blue = rng.randint(95, 145)
    for y in range(size):
        for x in range(size):
            wave = int(10 * math.sin((x + rng.random() * 3) / 8.0) + 8 * math.cos(y / 11.0))
            noise = rng.randint(-8, 8)
            arr[y, x] = [15 + noise // 2, 70 + wave + noise, base_blue + wave + noise]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _draw_swirl(draw: ImageDraw.ImageDraw, center: Tuple[int, int], radius: int, color: Tuple[int, int, int], turns: int = 3):
    cx, cy = center
    points = []
    for i in range(160):
        t = i / 160 * turns * 2 * math.pi
        r = radius * i / 160
        x = cx + r * math.cos(t)
        y = cy + r * math.sin(t)
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=color, width=max(2, radius // 12))


def make_image(label: str, size: int, seed: int) -> Image.Image:
    rng = random.Random(seed)
    img = _base_ocean(size, rng).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    if label == "oceano_normal":
        # Ondas suaves e poucas variações.
        for _ in range(rng.randint(4, 8)):
            y = rng.randint(8, size - 8)
            draw.arc([rng.randint(-20, 10), y - 8, size + rng.randint(-10, 25), y + 10], 0, 180, fill=(180, 220, 240, 55), width=1)

    elif label == "anomalia_termica":
        # Mancha vermelho/laranja simulando concentração térmica.
        for _ in range(rng.randint(2, 4)):
            cx, cy = rng.randint(20, size - 20), rng.randint(20, size - 20)
            rx, ry = rng.randint(14, 28), rng.randint(10, 22)
            draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=(255, rng.randint(80, 145), 30, rng.randint(95, 140)))
        img = img.filter(ImageFilter.GaussianBlur(radius=0.7))

    elif label == "tempestade":
        # Nuvens densas em espiral e área escura.
        draw.rectangle([0, 0, size, size], fill=(20, 35, 60, 75))
        center = (rng.randint(size//3, 2*size//3), rng.randint(size//3, 2*size//3))
        for r in range(size//8, size//2, 6):
            _draw_swirl(draw, center, r, (230, 235, 240, rng.randint(80, 135)), turns=2)
        draw.ellipse([center[0]-5, center[1]-5, center[0]+5, center[1]+5], fill=(15, 25, 45, 180))
        img = img.filter(ImageFilter.GaussianBlur(radius=0.4))

    elif label == "nuvens":
        # Muitas manchas brancas dispersas.
        for _ in range(rng.randint(8, 16)):
            cx, cy = rng.randint(0, size), rng.randint(0, size)
            rx, ry = rng.randint(8, 24), rng.randint(5, 15)
            draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=(245, 245, 245, rng.randint(110, 170)))
        img = img.filter(ImageFilter.GaussianBlur(radius=1.0))

    elif label == "algas":
        # Mancha verde irregular simulando proliferação de algas.
        for _ in range(rng.randint(3, 6)):
            cx, cy = rng.randint(15, size - 15), rng.randint(15, size - 15)
            rx, ry = rng.randint(12, 30), rng.randint(7, 18)
            draw.ellipse([cx-rx, cy-ry, cx+rx, cy+ry], fill=(rng.randint(25, 75), rng.randint(145, 220), rng.randint(45, 90), rng.randint(95, 145)))
        img = img.filter(ImageFilter.GaussianBlur(radius=0.8))

    # Ruído final leve
    arr = np.asarray(img).astype(np.int16)
    noise = np.random.default_rng(seed).normal(0, 5, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dataset", help="Pasta de saída do dataset")
    parser.add_argument("--images-per-class", type=int, default=220)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    for label in CLASSES:
        (out / label).mkdir(parents=True, exist_ok=True)
        for i in range(args.images_per_class):
            img = make_image(label, args.size, args.seed + hash(label) % 10000 + i)
            img.save(out / label / f"{label}_{i:04d}.png")

    print(f"Dataset gerado em: {out.resolve()}")
    print(f"Total de imagens: {len(CLASSES) * args.images_per_class}")
    print("Classes:", ", ".join(CLASSES))


if __name__ == "__main__":
    main()
