from ultralytics import YOLO
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="datasets/data.yaml")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--imgsz", type=int, default=640)
args = parser.parse_args()

model = YOLO("yolo11n.pt")
model.train(
    data=args.data,
    epochs=args.epochs,
    imgsz=args.imgsz,
    project="runs",
    name="road_damage",
)
print("Training complete. Copy the best weights to models/best.pt")
