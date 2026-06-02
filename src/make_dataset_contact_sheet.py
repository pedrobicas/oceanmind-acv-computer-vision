"""Gera uma folha de contato para revisar visualmente o dataset."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load_font(size: int):
    for candidate in ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf"):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def find_class_dirs(dataset: Path):
    if (dataset / "train").exists():
        return sorted([p for p in (dataset / "train").iterdir() if p.is_dir()])
    return sorted([p for p in dataset.iterdir() if p.is_dir()])


def find_images(dataset: Path, class_name: str):
    exts = {".jpg", ".jpeg", ".png"}
    if (dataset / "train").exists():
        images = []
        for split in ("train", "validation", "test"):
            images.extend([p for p in (dataset / split / class_name).glob("*") if p.suffix.lower() in exts])
        return images
    return [p for p in (dataset / class_name).glob("*") if p.suffix.lower() in exts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset_real")
    parser.add_argument("--output", default="reports/dataset_real_contact_sheet.jpg")
    parser.add_argument("--samples-per-class", type=int, default=8)
    parser.add_argument("--thumb-size", type=int, default=140)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    dataset = Path(args.dataset)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    class_dirs = find_class_dirs(dataset)
    if not class_dirs:
        raise FileNotFoundError(f"Nenhuma classe encontrada em {dataset}")

    font = load_font(18)
    label_font = load_font(22)
    margin = 24
    gap = 12
    label_h = 34
    thumb = args.thumb_size
    width = margin * 2 + args.samples_per_class * thumb + (args.samples_per_class - 1) * gap
    height = margin * 2 + len(class_dirs) * (label_h + thumb + gap)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    y = margin
    for class_dir in class_dirs:
        draw.text((margin, y), class_dir.name, font=label_font, fill="#1d1d1d")
        y += label_h
        images = find_images(dataset, class_dir.name)
        rng.shuffle(images)
        for i, path in enumerate(images[: args.samples_per_class]):
            img = Image.open(path).convert("RGB")
            img.thumbnail((thumb, thumb))
            x = margin + i * (thumb + gap)
            tile = Image.new("RGB", (thumb, thumb), "#eeeeee")
            tile.paste(img, ((thumb - img.width) // 2, (thumb - img.height) // 2))
            sheet.paste(tile, (x, y))
            draw.text((x, y + thumb - 20), path.stem[:18], font=font, fill="#222222")
        y += thumb + gap

    sheet.save(output, quality=92)
    print(f"Folha de contato gerada: {output}")


if __name__ == "__main__":
    main()
