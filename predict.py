from ultralytics import YOLO
import argparse

p = argparse.ArgumentParser()
p.add_argument("--source", required=True)
p.add_argument("--model", default="models/best.pt")
args = p.parse_args()

model = YOLO(args.model)
model.predict(source=args.source, conf=0.25, save=True)
print("Prediction complete. Check the runs/ directory.")
