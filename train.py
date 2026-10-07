from ultralytics import YOLO
import argparse


def main():
    p = argparse.ArgumentParser(description="Train the road-damage detector.")
    p.add_argument("--data", default="datasets/data.yaml")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--device", default=None, help="0 for GPU, cpu for CPU")
    args = p.parse_args()

    model = YOLO(args.model)
    kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project="runs",
        name="road_damage",
        pretrained=True,
        patience=10,
    )
    if args.device:
        kwargs["device"] = args.device

    model.train(**kwargs)
    print("Training complete.")
    print("Best checkpoint: runs/road_damage/weights/best.pt")


if __name__ == "__main__":
    main()
