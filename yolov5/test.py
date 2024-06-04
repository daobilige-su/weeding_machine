import torch

# Model
model = torch.hub.load("ultralytics/yolov5", "custom")  # or yolov5n - yolov5x6, custom

# Images
img = "/home/zhong/Desktop/CV/TrackR-CNN/data/LETTUCE_MOTS/train/images/0001/000000.png"  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.