from ultralytics import YOLO
from PIL import Image
import os

# Load your trained model
model = YOLO("../runs/classify/train5/weights/best.pt")

# Path to your test image
image_path = "../pigweed.jpeg"  # Replace with your image path

# Run prediction
results = model(image_path)

# Get the prediction results
for result in results:
    # Get class names
    class_names = result.names
    
    # Get predicted class index
    predicted_class_idx = result.probs.top1
    
    # Get confidence score
    confidence = result.probs.top1conf.item()
    
    # Get class name
    predicted_class = class_names[predicted_class_idx]
    
    print(f"Image: {image_path}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    print(f"All probabilities: {result.probs.data}")

# Show the image with prediction (optional)
result.show()  # This will display the image with prediction overlay