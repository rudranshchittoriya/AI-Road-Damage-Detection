from ultralytics import YOLO
import argparse


def main():
    p = argparse.ArgumentParser(description="Run inference on an image, folder or video.")
    p.add_argument("--source", required=True)
    p.add_argument("--model", default="models/best.pt")
    p.add_argument("--conf", type=float, default=0.25)
    args = p.parse_args()

    model = YOLO(args.model)
    model.predict(source=args.source, conf=args.conf, save=True)
    print("Prediction complete. Check the runs/detect/ directory.")


if __name__ == "__main__":
    main()
