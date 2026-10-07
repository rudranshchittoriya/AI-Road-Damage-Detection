# AI-Based Road Damage Detection and Severity Classification

A beginner-friendly deep-learning project for detecting road damage from images/video and estimating damage severity.

## Project title
**AI-Based Road Damage Detection and Severity Classification System**

## Features
- Road-damage detection using YOLO
- Supports images and videos
- Detects common RDD-style classes: longitudinal crack, transverse crack, alligator crack and pothole
- Estimates severity from detected damage size (demo/academic heuristic)
- Streamlit web interface
- Training script for a custom YOLO dataset
- Google Colab notebook for phone-only development

## Quick start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Put a trained YOLO model here
Create:
`models/best.pt`

You can train your own model using `train.py` and the dataset instructions in `docs/DATASET.md`.

### 3. Run the app
```bash
streamlit run app.py
```

## Important
The included severity score is an academic heuristic based on bounding-box area relative to image area. It is **not an engineering safety assessment**.

## Suggested GitHub repository
`AI-Road-Damage-Detection`
