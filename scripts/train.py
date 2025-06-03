from ultralytics import YOLO
import os
# Load a YOLO11 classification model
model = YOLO("yolo11n-cls.pt")  # nano model for faster training

# Train the model on your custom dataset
dataset_path = os.path.abspath("../dataset")  # or full path like "/Users/hrushikreddy/Desktop/yolo-weed-detection/dataset"

# Train with absolute path
results = model.train(
    data=dataset_path,
    epochs=100,
    imgsz=224,
    batch=16
)