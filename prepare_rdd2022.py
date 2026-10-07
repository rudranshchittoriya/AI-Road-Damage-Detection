from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASS_MAP = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}


def xml_to_yolo(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        return []
    width = float(size.findtext("width", "0"))
    height = float(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        return []

    labels = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if name not in CLASS_MAP:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue

        xmin = max(0.0, min(float(box.findtext("xmin", "0")), width))
        ymin = max(0.0, min(float(box.findtext("ymin", "0")), height))
        xmax = max(0.0, min(float(box.findtext("xmax", "0")), width))
        ymax = max(0.0, min(float(box.findtext("ymax", "0")), height))
        if xmax <= xmin or ymax <= ymin:
            continue

        xc = ((xmin + xmax) / 2) / width
        yc = ((ymin + ymax) / 2) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        labels.append(f"{CLASS_MAP[name]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    return labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="datasets/raw/RDD2022")
    p.add_argument("--out", default="datasets/rdd2022")
    p.add_argument("--countries", nargs="+", default=["India"])
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)

    if not raw.exists():
        raise SystemExit(f"Raw dataset not found: {raw}")

    pairs = []
    for country in args.countries:
        image_dir = raw / country / "train" / "images"
        xml_dir = raw / country / "train" / "annotations" / "xmls"
        if not image_dir.exists() or not xml_dir.exists():
            raise SystemExit(
                f"Missing folders for {country}. Expected {image_dir} and {xml_dir}"
            )

        for image in image_dir.glob("*.jpg"):
            xml = xml_dir / f"{image.stem}.xml"
            if xml.exists():
                pairs.append((image, xml, country))

    if not pairs:
        raise SystemExit("No image/XML pairs found.")

    random.Random(args.seed).shuffle(pairs)
    split = int(len(pairs) * (1 - args.val_ratio))
    splits = {"train": pairs[:split], "val": pairs[split:]}

    for split_name, items in splits.items():
        image_out = out / "images" / split_name
        label_out = out / "labels" / split_name
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for image, xml, country in items:
            safe_name = f"{country}_{image.name}"
            shutil.copy2(image, image_out / safe_name)
            labels = xml_to_yolo(xml)
            (label_out / f"{Path(safe_name).stem}.txt").write_text(
                "\n".join(labels), encoding="utf-8"
            )

    print(f"Prepared {len(splits['train'])} training images and {len(splits['val'])} validation images.")
    print(f"Output: {out.resolve()}")


if __name__ == "__main__":
    main()
