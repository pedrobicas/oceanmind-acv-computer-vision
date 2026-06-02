"""Baixa imagens reais/observacionais do NASA GIBS para treino supervisionado.

Os rotulos gerados aqui sao aproximados: cada classe usa uma combinacao de
camada, regiao e periodo. Antes de usar para uma entrega final, revise
visualmente uma amostra das imagens baixadas.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable
from uuid import uuid4

import requests
from PIL import Image, ImageStat

BASE_URL = "https://gibs.earthdata.nasa.gov/wms/epsg4326/best/wms.cgi"
DEFAULT_CLASSES = ["oceano_normal", "anomalia_termica", "tempestade", "nuvens", "algas"]


@dataclass(frozen=True)
class ClassConfig:
    layer_candidates: tuple[str, ...]
    bboxes: tuple[tuple[float, float, float, float], ...]
    description: str


CLASS_CONFIGS: dict[str, ClassConfig] = {
    "oceano_normal": ClassConfig(
        layer_candidates=(
            "VIIRS_SNPP_CorrectedReflectance_TrueColor",
            "MODIS_Aqua_CorrectedReflectance_TrueColor",
            "MODIS_Terra_CorrectedReflectance_TrueColor",
        ),
        bboxes=(
            (-45, -40, -20, -15),
            (-170, -25, -140, 0),
            (50, -35, 85, -10),
            (-35, 5, -15, 25),
            (-130, -10, -100, 15),
        ),
        description="Recortes de oceano aberto em true color.",
    ),
    "nuvens": ClassConfig(
        layer_candidates=(
            "VIIRS_SNPP_CorrectedReflectance_TrueColor",
            "MODIS_Aqua_CorrectedReflectance_TrueColor",
            "MODIS_Terra_CorrectedReflectance_TrueColor",
        ),
        bboxes=(
            (-60, -30, -30, 0),
            (-40, -5, -10, 20),
            (110, -30, 145, 5),
            (40, -20, 75, 10),
            (-160, -5, -125, 20),
        ),
        description="Regioes oceanicas com alta chance de cobertura de nuvens.",
    ),
    "tempestade": ClassConfig(
        layer_candidates=(
            "VIIRS_SNPP_CorrectedReflectance_TrueColor",
            "MODIS_Aqua_CorrectedReflectance_TrueColor",
            "MODIS_Terra_CorrectedReflectance_TrueColor",
        ),
        bboxes=(
            (-85, 5, -55, 35),
            (115, 0, 155, 30),
            (45, -30, 85, 0),
            (-120, 5, -90, 30),
            (130, -25, 170, 5),
        ),
        description="Bacias tropicais com maior probabilidade de sistemas convectivos.",
    ),
    "algas": ClassConfig(
        layer_candidates=(
            "MODIS_Aqua_L2_Chlorophyll_A",
            "MODIS_Terra_L2_Chlorophyll_A",
            "VIIRS_SNPP_L2_Chlorophyll_A",
            "VIIRS_NOAA20_Chlorophyll_a",
            "OCI_PACE_Chlorophyll_a",
        ),
        bboxes=(
            (-85, 20, -60, 35),
            (-20, 10, 5, 25),
            (5, -35, 25, -20),
            (-80, -20, -55, 0),
            (120, -10, 145, 10),
        ),
        description="Camadas de clorofila-a em regioes costeiras e de ressurgencia.",
    ),
    "anomalia_termica": ClassConfig(
        layer_candidates=(
            "GHRSST_L4_MUR_Sea_Surface_Temperature_Anomalies",
            "GHRSST_L4_MUR25_Sea_Surface_Temperature_Anomalies",
            "MODIS_Aqua_L3_SST_Thermal_4km_Day_Daily",
            "MODIS_Terra_L3_SST_Thermal_4km_Day_Daily",
            "VIIRS_SNPP_L2_Sea_Surface_Temp_Day",
        ),
        bboxes=(
            (-85, 5, -55, 30),
            (-50, -45, -25, -25),
            (30, -35, 60, -10),
            (120, 0, 155, 25),
            (-130, -5, -95, 20),
        ),
        description="Camadas de temperatura da superficie do mar em regioes dinamicas.",
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Baixa e organiza dataset real do NASA GIBS.")
    parser.add_argument("--output", default="dataset_real", help="Pasta final do dataset.")
    parser.add_argument("--raw-output", default="raw_nasa_downloads", help="Pasta local para manter os downloads brutos.")
    parser.add_argument("--images-per-class", type=int, default=80)
    parser.add_argument("--size", type=int, default=160, help="Largura/altura baixada do WMS.")
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES, choices=DEFAULT_CLASSES)
    parser.add_argument("--days-back", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--max-attempt-factor", type=int, default=8)
    parser.add_argument("--sleep", type=float, default=0.25)
    parser.add_argument("--keep-existing", action="store_true", help="Nao limpa a pasta de saida antes de montar o split.")
    return parser.parse_args()


def random_date(rng: random.Random, days_back: int) -> str:
    delta = rng.randint(10, days_back)
    return (date.today() - timedelta(days=delta)).isoformat()


def jitter_bbox(rng: random.Random, bbox, min_size=8.0, max_size=18.0, max_shift=3.0):
    min_lon, min_lat, max_lon, max_lat = bbox
    center_lon = rng.uniform(min_lon, max_lon)
    center_lat = rng.uniform(min_lat, max_lat)
    width = rng.uniform(min_size, max_size)
    height = rng.uniform(min_size, max_size)

    new_min_lon = max(-180, min(180, center_lon - width / 2 + rng.uniform(-max_shift, max_shift)))
    new_max_lon = max(-180, min(180, center_lon + width / 2 + rng.uniform(-max_shift, max_shift)))
    new_min_lat = max(-90, min(90, center_lat - height / 2 + rng.uniform(-max_shift, max_shift)))
    new_max_lat = max(-90, min(90, center_lat + height / 2 + rng.uniform(-max_shift, max_shift)))

    if new_min_lon > new_max_lon:
        new_min_lon, new_max_lon = new_max_lon, new_min_lon
    if new_min_lat > new_max_lat:
        new_min_lat, new_max_lat = new_max_lat, new_min_lat
    return new_min_lon, new_min_lat, new_max_lon, new_max_lat


def wms_params(layer: str, bbox, image_date: str, size: int) -> dict[str, str | int]:
    min_lon, min_lat, max_lon, max_lat = bbox
    return {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "STYLES": "",
        "FORMAT": "image/jpeg",
        "TRANSPARENT": "false",
        "CRS": "EPSG:4326",
        # WMS 1.3.0 com EPSG:4326 usa ordem lat,lon.
        "BBOX": f"{min_lat},{min_lon},{max_lat},{max_lon}",
        "WIDTH": size,
        "HEIGHT": size,
        "TIME": image_date,
    }


def is_useful_image(path: Path, min_stddev: float = 4.0) -> bool:
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")
            if img.size[0] < 32 or img.size[1] < 32:
                return False
            stat = ImageStat.Stat(img)
            return sum(stat.stddev) / len(stat.stddev) >= min_stddev
    except OSError:
        return False


def download_image(session: requests.Session, output_path: Path, layer: str, bbox, image_date: str, size: int) -> bool:
    response = session.get(BASE_URL, params=wms_params(layer, bbox, image_date, size), timeout=45)
    if response.status_code != 200:
        return False
    if "image" not in response.headers.get("content-type", ""):
        return False
    output_path.write_bytes(response.content)
    if not is_useful_image(output_path):
        output_path.unlink(missing_ok=True)
        return False
    return True


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_class(
    session: requests.Session,
    label: str,
    config: ClassConfig,
    class_dir: Path,
    images_per_class: int,
    size: int,
    days_back: int,
    max_attempt_factor: int,
    sleep: float,
    rng: random.Random,
) -> list[dict]:
    class_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    seen_hashes = set()
    attempts = 0
    max_attempts = max(images_per_class * max_attempt_factor, images_per_class)

    while len(rows) < images_per_class and attempts < max_attempts:
        attempts += 1
        bbox = jitter_bbox(rng, rng.choice(config.bboxes))
        layer = rng.choice(config.layer_candidates)
        image_date = random_date(rng, days_back)
        filename = f"{label}_{len(rows) + 1:04d}_{image_date}_{uuid4().hex[:8]}.jpg"
        output_path = class_dir / filename

        ok = download_image(session, output_path, layer, bbox, image_date, size)
        if ok:
            file_hash = sha1_file(output_path)
            if file_hash in seen_hashes:
                output_path.unlink(missing_ok=True)
            else:
                seen_hashes.add(file_hash)
                rows.append(
                    {
                        "label": label,
                        "file": str(output_path),
                        "date": image_date,
                        "layer": layer,
                        "bbox": list(bbox),
                        "source": BASE_URL,
                    }
                )
                print(f"[{label}] {len(rows):03d}/{images_per_class} {filename}")
        time.sleep(sleep)

    return rows


def split_files(files: Iterable[Path], train_ratio: float, val_ratio: float, rng: random.Random):
    files = list(files)
    rng.shuffle(files)
    n_total = len(files)
    if n_total >= 3:
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio))
        if n_train + n_val >= n_total:
            n_train = max(1, n_total - 2)
            n_val = 1
    else:
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
    return {
        "train": files[:n_train],
        "validation": files[n_train : n_train + n_val],
        "test": files[n_train + n_val :],
    }


def build_split(raw_run_dir: Path, output_dir: Path, classes: list[str], args, rows: list[dict]):
    if output_dir.exists() and not args.keep_existing:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    split_counts: dict[str, dict[str, int]] = {}

    for label in classes:
        class_files = sorted((raw_run_dir / label).glob("*.jpg"))
        split = split_files(class_files, args.train_ratio, args.val_ratio, rng)
        split_counts[label] = {}
        for split_name, files in split.items():
            dst_dir = output_dir / split_name / label
            dst_dir.mkdir(parents=True, exist_ok=True)
            split_counts[label][split_name] = len(files)
            for src in files:
                shutil.copy2(src, dst_dir / src.name)

    metadata = {
        "dataset_type": "real_nasa_gibs_weak_labels",
        "created_at": date.today().isoformat(),
        "source": BASE_URL,
        "classes": classes,
        "image_size": args.size,
        "requested_images_per_class": args.images_per_class,
        "split": {
            "train_ratio": args.train_ratio,
            "validation_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "counts": split_counts,
        },
        "labeling_note": "Rotulos aproximados por camada/regiao/data. Recomenda-se revisao visual.",
        "downloads": rows,
    }
    (output_dir / "dataset_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main():
    args = parse_args()
    if round(args.train_ratio + args.val_ratio + args.test_ratio, 6) != 1:
        raise ValueError("A soma de --train-ratio, --val-ratio e --test-ratio precisa ser 1.0")

    rng = random.Random(args.seed)
    raw_root = Path(args.raw_output)
    run_id = date.today().isoformat() + "_" + uuid4().hex[:6]
    raw_run_dir = raw_root / run_id
    raw_run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fonte WMS: {BASE_URL}")
    print(f"Download bruto: {raw_run_dir}")
    print(f"Dataset final: {Path(args.output)}")

    rows = []
    with requests.Session() as session:
        for label in args.classes:
            rows.extend(
                download_class(
                    session=session,
                    label=label,
                    config=CLASS_CONFIGS[label],
                    class_dir=raw_run_dir / label,
                    images_per_class=args.images_per_class,
                    size=args.size,
                    days_back=args.days_back,
                    max_attempt_factor=args.max_attempt_factor,
                    sleep=args.sleep,
                    rng=rng,
                )
            )

    metadata = build_split(raw_run_dir, Path(args.output), args.classes, args, rows)
    print("\nDataset real montado.")
    print(json.dumps(metadata["split"]["counts"], ensure_ascii=False, indent=2))
    print(f"Metadados: {Path(args.output) / 'dataset_metadata.json'}")


if __name__ == "__main__":
    main()
