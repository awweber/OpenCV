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

win_name = "YOLO Pose Estimation"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Load the pre-trained YOLO11n pose model
try:
    model = YOLO("yolo11n-pose.pt")
    print("YOLO11n pose model loaded successfully")
except Exception as e:
    print(f"Error loading YOLO pose model: {e}")
    source.release()
    sys.exit(1)

# COCO pose keypoints (17 keypoints for human pose estimation)
pose_keypoints = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Define skeleton connections for drawing pose lines
pose_skeleton = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # legs
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],         # torso and arms
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],         # arms and face
    [2, 4], [3, 5], [4, 6], [5, 7]                    # face to shoulders
]

# Configuration parameters
conf_threshold = 0.5

print("Press ESC to exit...")
print("Starting pose estimation...")

# Process frames from the camera
try:
    while cv2.waitKey(1) != 27:  # Escape key
        has_frame, frame = source.read()
        if not has_frame:
            print("Error: Could not read frame from camera")
            break
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Run YOLO pose estimation inference
        results = model(frame, conf=conf_threshold, verbose=False, task="pose")
        
        # Process pose detections
        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints
            
            if keypoints is not None and boxes is not None:
                # Get keypoints data
                keypoints_data = keypoints.xy.cpu().numpy()  # Shape: (n, 17, 2)
                keypoints_conf = keypoints.conf.cpu().numpy()  # Shape: (n, 17)
                
                for i, (box, kpts, kpts_conf) in enumerate(zip(boxes, keypoints_data, keypoints_conf)):
                    # Get confidence score for person detection
                    person_confidence = box.conf[0].cpu().numpy()
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    
                    # Draw bounding box around person
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw person detection label
                    label = f"Person: {person_confidence:.2f}"
                    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    
                    cv2.rectangle(
                        frame,
                        (x1, y1 - label_size[1] - 10),
                        (x1 + label_size[0], y1),
                        (0, 255, 0),
                        cv2.FILLED,
                    )
                    
                    cv2.putText(
                        frame, 
                        label, 
                        (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (0, 0, 0),
                        1
                    )
                    
                    # Draw keypoints (joints)
                    for j, (x, y) in enumerate(kpts):
                        if kpts_conf[j] > 0.5:  # Only draw confident keypoints
                            # Draw keypoint as circle
                            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
                            
                            # Optionally draw keypoint label
                            if j < len(pose_keypoints):
                                cv2.putText(
                                    frame,
                                    pose_keypoints[j],
                                    (int(x) + 5, int(y) - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.3,
                                    (255, 255, 255),
                                    1
                                )
                    
                    # Draw skeleton connections
                    for connection in pose_skeleton:
                        kpt1_idx = connection[0] - 1  # Convert to 0-based index
                        kpt2_idx = connection[1] - 1  # Convert to 0-based index
                        
                        if (kpt1_idx < len(kpts) and kpt2_idx < len(kpts) and
                            kpts_conf[kpt1_idx] > 0.5 and kpts_conf[kpt2_idx] > 0.5):
                            
                            x1, y1 = int(kpts[kpt1_idx][0]), int(kpts[kpt1_idx][1])
                            x2, y2 = int(kpts[kpt2_idx][0]), int(kpts[kpt2_idx][1])
                            
                            # Draw line between connected keypoints
                            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Add instruction text
        fps_label = "Press ESC to exit | Pose Estimation"
        cv2.putText(frame, fps_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow(win_name, frame)

except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"An error occurred during pose estimation: {e}")

# Clean up
source.release()
cv2.destroyWindow(win_name)
print("Pose estimation stopped.")