# Dataset setup

This project uses RDD2022 (Road Damage Dataset 2022). The official dataset contains 47,420 road images from six countries and more than 55,000 annotated damage instances. It includes longitudinal cracks, transverse cracks, alligator cracks and potholes.

Official sources:
- Figshare DOI: 10.6084/m9.figshare.21431547.v1
- Official project: https://github.com/sekilab/RoadDamageDetector

The full dataset is intentionally not committed to this repository.

## Recommended phone-only setup

Start with the India subset in Google Colab. The official project lists that archive at about 502 MB.

After extraction, the expected structure is:

datasets/raw/RDD2022/India/train/images/
datasets/raw/RDD2022/India/train/annotations/xmls/

Then run:

    python scripts/prepare_rdd2022.py --countries India

This creates an 80/20 train/validation split in datasets/rdd2022.

## Training

    python train.py --data datasets/data.yaml --epochs 30 --imgsz 640 --batch 8

The best checkpoint is written to:

    runs/road_damage/weights/best.pt

Copy it to:

    models/best.pt

## Evaluation

    python evaluate.py --model models/best.pt --data datasets/data.yaml

Do not add invented accuracy numbers to the README. Only report metrics produced by your own run.
