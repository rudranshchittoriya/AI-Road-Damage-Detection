# Google Colab training — phone-only workflow

You can complete the training part without a laptop.

## Step 1 — Open Colab

Open Google Colab in your phone browser and create a new notebook.

## Step 2 — Clone the repository

Run:

    !git clone https://github.com/rudranshchittoriya/AI-Road-Damage-Detection.git
    %cd AI-Road-Damage-Detection

## Step 3 — Install packages

    !pip install -q -r requirements.txt

## Step 4 — Download RDD2022

Use the official RDD2022 source. For the first experiment, use the India subset because it is much smaller than the complete dataset.

Official source:
https://github.com/sekilab/RoadDamageDetector

After downloading and extracting, make sure this path exists:

    datasets/raw/RDD2022/India/train/images

and this path exists:

    datasets/raw/RDD2022/India/train/annotations/xmls

## Step 5 — Prepare YOLO labels

    !python scripts/prepare_rdd2022.py --countries India

## Step 6 — Train

Enable a GPU in Colab first: Runtime → Change runtime type → GPU.

Then run:

    !python train.py --data datasets/data.yaml --epochs 30 --imgsz 640 --batch 8 --device 0

For a quick demonstration, 10 epochs is enough to test the pipeline. For the final academic experiment, use more epochs if the available GPU time allows.

## Step 7 — Evaluate

    !python evaluate.py --model runs/road_damage/weights/best.pt --data datasets/data.yaml

Save the printed mAP50, mAP50-95, precision and recall. These are your actual project results.

## Step 8 — Test a road image

    !python predict.py --model runs/road_damage/weights/best.pt --source /content/test-road.jpg

## Step 9 — Use the Streamlit app

Copy the checkpoint to models/best.pt:

    !mkdir -p models
    !cp runs/road_damage/weights/best.pt models/best.pt

Then run:

    !streamlit run app.py &>/content/streamlit.log &

For a public browser demo in Colab, use a tunneling tool such as Cloudflare Tunnel or another service permitted by your institution. The GitHub repository itself does not need to contain the large checkpoint.
