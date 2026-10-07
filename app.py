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
st.caption("YOLO-based academic demo for road-damage detection and severity estimation.")

MODEL_PATH = "models/best.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return YOLO(MODEL_PATH)

def severity(box, w, h):
    x1, y1, x2, y2 = box
    ratio = max(0.0, (x2-x1) * (y2-y1) / float(w*h))
    if ratio >= 0.12:
        level = "High"
    elif ratio >= 0.04:
        level = "Medium"
    else:
        level = "Low"
    return level, round(ratio * 100, 2)

model = load_model()

if model is None:
    st.warning("No trained model found. Add your trained file as `models/best.pt` and restart the app.")
    st.info("For phone-only development, use the included Google Colab notebook to train the model.")
    st.stop()

source = st.radio("Input", ["Image", "Video"], horizontal=True)

if source == "Image":
    uploaded = st.file_uploader("Upload a road image", type=["jpg","jpeg","png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        result = model.predict(np.array(image), conf=0.25, verbose=False)[0]
        plotted = result.plot()
        st.image(plotted, caption="AI detection result", use_container_width=True)

        h, w = np.array(image).shape[:2]
        rows = []
        for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(),
                                  result.boxes.cls.cpu().numpy(),
                                  result.boxes.conf.cpu().numpy()):
            label = result.names[int(cls)]
            level, area = severity(box, w, h)
            rows.append({"Damage": label, "Confidence": round(float(conf), 2),
                         "Severity": level, "Area %": area})
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Damages", len(df))
            c2.metric("High", int((df.Severity=="High").sum()))
            c3.metric("Medium", int((df.Severity=="Medium").sum()))
        else:
            st.success("No damage detected above the confidence threshold.")

else:
    uploaded = st.file_uploader("Upload a road video", type=["mp4","mov","avi"])
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
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))

        total = 0
        high = 0
        medium = 0
        low = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = model.predict(frame, conf=0.25, verbose=False)[0]
            for box in result.boxes.xyxy.cpu().numpy():
                level, _ = severity(box, w, h)
                total += 1
                if level == "High": high += 1
                elif level == "Medium": medium += 1
                else: low += 1
            writer.write(result.plot())
        cap.release()
        writer.release()

        st.video(output_path)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Detections", total)
        c2.metric("High", high)
        c3.metric("Medium", medium)
        c4.metric("Low", low)
