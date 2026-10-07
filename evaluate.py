from ultralytics import YOLO
import argparse


def main():
    p = argparse.ArgumentParser(description="Evaluate the trained road-damage detector.")
    p.add_argument("--model", default="models/best.pt")
    p.add_argument("--data", default="datasets/data.yaml")
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    model = YOLO(args.model)
    metrics = model.val(data=args.data, imgsz=args.imgsz)

    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()
