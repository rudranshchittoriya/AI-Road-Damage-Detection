# AI-Based Road Damage Detection and Severity Classification System

## Abstract

This project demonstrates an end-to-end computer-vision pipeline for detecting common road-surface damage in images and videos. A YOLO object-detection model is trained on RDD2022 and deployed through a Streamlit interface. The application reports damage classes, confidence scores, and an academic severity estimate based on relative bounding-box area.

## Objectives

1. Detect road damage automatically from images and video.
2. Classify four common damage types.
3. Visualize detections with bounding boxes.
4. Provide a simple severity-prioritization heuristic.
5. Provide a reproducible training and deployment workflow.

## Classes

0 — Longitudinal crack (D00)
1 — Transverse crack (D10)
2 — Alligator crack (D20)
3 — Pothole (D40)

## Methodology

1. Obtain RDD2022 data.
2. Convert PASCAL-VOC XML annotations to YOLO labels.
3. Split images into training and validation sets.
4. Fine-tune a lightweight YOLO model using transfer learning.
5. Evaluate on validation data.
6. Run inference on unseen images/videos.
7. Display results in Streamlit.

## Severity heuristic

Low: less than 4% of image area.
Medium: 4% to less than 12%.
High: 12% or more.

This is an academic prioritization heuristic, not a civil-engineering safety assessment.

## Limitations

Performance depends on the training subset and number of epochs. Very small, occluded, or unusual defects may be missed. Bounding-box area is not a reliable engineering measure of pavement severity. The project does not claim production or safety certification.

## Reproducibility

The repository contains dataset conversion, training, evaluation, inference, application, requirements, and documentation. Large datasets and model weights are excluded from Git history and should be downloaded/generated separately.
