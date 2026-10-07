# Model weights

Place the trained YOLO checkpoint here:

models/best.pt

The checkpoint is intentionally not stored in Git because model weights can be large.
After training in Google Colab, copy `runs/road_damage/weights/best.pt` to this folder.

Do not claim any accuracy value until you run `evaluate.py` on your own trained checkpoint.
