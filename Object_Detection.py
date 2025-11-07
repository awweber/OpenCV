import os
import cv2
import sys
from ultralytics import YOLO #type: ignore
import numpy as np

# Initialize video source
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

try:
    source = cv2.VideoCapture(s)
    if not source.isOpened():
        print("Error: Could not open camera")
        sys.exit(1)
except Exception as e:
    print(f"Error initializing camera: {e}")
    sys.exit(1)

win_name = "YOLO Object Detection"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Load the pre-trained YOLO11n model
try:
    model = YOLO("yolo11n.pt")
    print("YOLO11n model loaded successfully")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    source.release()
    sys.exit(1)

# YOLO class names (COCO dataset)
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Configuration parameters
conf_threshold = 0.5

print("Press ESC to exit...")
print("Starting object detection...")

# Process frames from the camera
try:
    while cv2.waitKey(1) != 27:  # Escape key
        has_frame, frame = source.read()
        if not has_frame:
            print("Error: Could not read frame from camera")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Run YOLO inference
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # Process detections
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Get confidence score
                    confidence = box.conf[0].cpu().numpy()
                    
                    # Get class index and name
                    class_idx = int(box.cls[0].cpu().numpy())
                    class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class_{class_idx}"
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Create label with class name and confidence
                    label = f"{class_name}: {confidence:.2f}"
                    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    
                    # Draw background rectangle for label
                    cv2.rectangle(
                        frame,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1),
                        (0, 255, 0),
                        cv2.FILLED,
                    )
                    
                    # Put text label
                    cv2.putText(
                        frame, 
                        label, 
                        (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 0, 0),
                        1
                    )
        
        # Add FPS information
        fps_label = "Press ESC to exit"
        cv2.putText(frame, fps_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow(win_name, frame)

except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"An error occurred during detection: {e}")

# Clean up
source.release()
cv2.destroyWindow(win_name)
print("Object detection stopped.")