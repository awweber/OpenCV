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

win_name = "YOLO Segmentation"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Load the pre-trained YOLO11n segmentation model
try:
    model = YOLO("yolo11n-seg.pt")
    print("YOLO11n segmentation model loaded successfully")
except Exception as e:
    print(f"Error loading YOLO segmentation model: {e}")
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

# Generate random colors for each class
np.random.seed(42)  # For reproducible colors
colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)

print("Press ESC to exit...")
print("Starting segmentation...")

# Process frames from the camera
try:
    while cv2.waitKey(1) != 27:  # Escape key
        has_frame, frame = source.read()
        if not has_frame:
            print("Error: Could not read frame from camera")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        original_frame = frame.copy()
        
        # Run YOLO segmentation inference
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # Process detections and segmentation masks
        for result in results:
            boxes = result.boxes
            masks = result.masks
            
            if masks is not None and boxes is not None:
                # Get the masks data
                masks_data = masks.data.cpu().numpy()  # Shape: (n, H, W)
                
                # Create overlay for masks
                mask_overlay = np.zeros_like(frame)
                
                for i, (box, mask) in enumerate(zip(boxes, masks_data)):
                    # Get class information
                    class_idx = int(box.cls[0].cpu().numpy())
                    confidence = box.conf[0].cpu().numpy()
                    class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class_{class_idx}"
                    
                    # Get color for this class
                    color = colors[class_idx % len(colors)].tolist()
                    
                    # Resize mask to match frame dimensions
                    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                    
                    # Create colored mask
                    colored_mask = np.zeros_like(frame)
                    colored_mask[mask_resized > 0.5] = color
                    
                    # Add to overlay
                    mask_overlay = cv2.addWeighted(mask_overlay, 1.0, colored_mask, 0.7, 0)
                
                # Blend original frame with mask overlay (only masks, no boxes/labels)
                frame = cv2.addWeighted(original_frame, 0.6, mask_overlay, 0.4, 0)
        
        # Add instruction text
        info_label = "Press ESC to exit | Mask-only Segmentation"
        cv2.putText(frame, info_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow(win_name, frame)

except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"An error occurred during segmentation: {e}")

# Clean up
source.release()
cv2.destroyWindow(win_name)
print("Segmentation stopped.")