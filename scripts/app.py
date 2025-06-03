import cv2
from ultralytics import YOLO
import time

class WeedDetectorApp:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    def run(self):
        print("Starting Weed Detection App...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Run prediction
            results = self.model(frame, verbose=False)
            result = results[0]
            
            # Get prediction details
            predicted_class = result.names[result.probs.top1]
            confidence = result.probs.top1conf.item()
            
            # Colors for different predictions
            color = (0, 255, 0) if predicted_class == 'normal' else (0, 0, 255)
            
            # Add prediction text
            text = f"{predicted_class.upper()}: {confidence:.2%}"
            cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # Add confidence bar
            bar_width = int(confidence * 300)
            cv2.rectangle(frame, (10, 60), (10 + bar_width, 80), color, -1)
            cv2.rectangle(frame, (10, 60), (310, 80), (255, 255, 255), 2)
            
            # Add status indicator
            status_text = "🌿 WEED DETECTED!" if predicted_class == 'weed' else "✅ Normal Plant"
            cv2.putText(frame, status_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Add instructions
            cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Real-time Weed Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                filename = f"detection_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved as {filename}")
        
        self.cap.release()
        cv2.destroyAllWindows()

# Run the app
if __name__ == "__main__":
    model_path = "../runs/classify/train5/weights/best.pt"
    app = WeedDetectorApp(model_path)
    app.run()