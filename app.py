import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Road Damage AI", page_icon="🚧", layout="wide")
st.title("🚧 AI Road Damage Detection")
st.caption("YOLO-based academic demo for road-damage detection and severity prioritization.")

MODEL_PATH = "models/best.pt"


@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    return YOLO(path)


def severity(box, w, h):
    x1, y1, x2, y2 = box
    ratio = max(0.0, (x2 - x1) * (y2 - y1) / float(w * h))
    if ratio >= 0.12:
        level = "High"
    elif ratio >= 0.04:
        level = "Medium"
    else:
        level = "Low"
    return level, round(ratio * 100, 2)


uploaded_model = st.sidebar.file_uploader(
    "Optional: upload best.pt",
    type=["pt"],
    help="Use this if models/best.pt is not present."
)

if uploaded_model:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        f.write(uploaded_model.read())
        runtime_model_path = f.name
else:
    runtime_model_path = MODEL_PATH

model = load_model(runtime_model_path)

if model is None:
    st.warning("No trained model found.")
    st.info("Train the model in Google Colab, then place best.pt in models/ or upload it in the sidebar.")
    st.stop()

confidence = st.sidebar.slider("Confidence threshold", 0.10, 0.90, 0.25, 0.05)
source = st.radio("Input", ["Image", "Video"], horizontal=True)

if source == "Image":
    uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        arr = np.array(image)
        result = model.predict(arr, conf=confidence, verbose=False)[0]

        st.image(result.plot(), caption="AI detection result", use_container_width=True)

        h, w = arr.shape[:2]
        rows = []
        for box, cls, conf in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
        ):
            level, area = severity(box, w, h)
            rows.append(
                {
                    "Damage": result.names[int(cls)],
                    "Confidence": round(float(conf), 2),
                    "Severity": level,
                    "Area %": area,
                }
            )

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Damages", len(df))
            c2.metric("High", int((df.Severity == "High").sum()))
            c3.metric("Medium", int((df.Severity == "Medium").sum()))
            c4.metric("Low", int((df.Severity == "Low").sum()))
        else:
            st.success("No damage detected above the confidence threshold.")

else:
    uploaded = st.file_uploader("Upload a road video", type=["mp4", "mov", "avi"])

    if uploaded:
        suffix = os.path.splitext(uploaded.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(uploaded.read())
            input_path = f.name

        output_path = input_path + "_out.mp4"
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w, h),
        )

        total = high = medium = low = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = model.predict(frame, conf=confidence, verbose=False)[0]
            for box in result.boxes.xyxy.cpu().numpy():
                level, _ = severity(box, w, h)
                total += 1
                if level == "High":
                    high += 1
                elif level == "Medium":
                    medium += 1
                else:
                    low += 1

            writer.write(result.plot())

        cap.release()
        writer.release()

        st.video(output_path)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Detections", total)
        c2.metric("High", high)
        c3.metric("Medium", medium)
        c4.metric("Low", low)
