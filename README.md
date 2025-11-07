# OpenCV

OpenCV  tutorials and projects using Python


## Installation

To install OpenCV, you can use pip. Run the following command:

```
pip install opencv-python
```
For additional functionalities, you may also want to install the contrib package:

```
pip install opencv-contrib-python
```
## Usage
You can import OpenCV in your Python scripts as follows:

```python
import cv2 as cv
```
or for specific modules:

```python
from cv2 import cv2 as cv
```

## Tutorial

see 'tutorial.ipynb' for tutorials on (based on Udemy course):
- Basic Image Operations
- Image Filtering
- Face Detection with Haar Cascades

## Projects

### Image Filtering & Edge Detection
- **File:** `Image_Filtering_Edge_Detection.py`
- **Description:** Real-time camera filters and edge detection
- **Features:**
  - Live webcam feed processing
  - Multiple filter modes:
    - **Preview Mode (P):** Original camera feed
    - **Canny Edge Detection (C):** Black and white edge detection
    - **Blur Filter (B):** Gaussian blur effect (13x13 kernel)
    - **Corner Feature Detection (F):** Green circles marking detected corners
  - Interactive keyboard controls
  - Horizontal mirror effect
- **Controls:**
  - `P` - Preview mode (original)
  - `C` - Canny edge detection
  - `B` - Blur filter
  - `F` - Feature/corner detection
  - `Q` or `ESC` - Exit

### Face Detection
- **File:** `Face_Detection.py`
- **Description:** Real-time face detection using pre-trained Caffe model
- **Model:** `res10_300x300_ssd_iter_140000_fp16.caffemodel`
- **Features:**
  - Live webcam feed processing
  - Face detection with confidence scores
  - Green bounding boxes around detected faces
  - Adjustable confidence threshold (default: 0.7)

### Object Detection
- **File:** `Object_Detection.py`
- **Description:** Real-time object detection using YOLO11n
- **Model:** `yolo11n.pt` (80 COCO classes)
- **Features:**
  - Live webcam feed processing
  - Detects 80 different object classes (person, car, bicycle, etc.)
  - Green bounding boxes with class labels and confidence scores
  - Adjustable confidence threshold (default: 0.5)

### Instance Segmentation
- **File:** `Segmentation.py`
- **Description:** Real-time instance segmentation using YOLO11n-seg
- **Model:** `yolo11n-seg.pt`
- **Features:**
  - Live webcam feed processing
  - Pixel-level object segmentation
  - Color-coded masks for different object classes
  - No bounding boxes - pure mask visualization
  - Transparent overlay (40% mask, 60% original image)

### Pose Estimation
- **File:** `Pose_Estimation.py`
- **Description:** Real-time human pose estimation using YOLO11n-pose
- **Model:** `yolo11n-pose.pt`
- **Features:**
  - Live webcam feed processing
  - 17-point human pose detection (COCO keypoints)
  - Skeleton visualization with connected joints
  - Red keypoint markers and blue skeleton lines
  - Person detection with green bounding boxes

### Image Object Detection
- **File:** `Image_Object_Detection.py`
- **Description:** Static image object detection using YOLO11n
- **Model:** `yolo11n.pt`
- **Features:**
  - Processes single images
  - Saves results with annotations
  - Performance timing measurements

## Hardware Requirements
- Computer with webcam (for real-time detection scripts)
- Python 3.x
- Sufficient RAM for model loading (minimum 4GB recommended)

## Software Requirements
- OpenCV (`opencv-python`)
- ultralytics (`pip install ultralytics`)
- NumPy
- matplotlib (for image display)

## Installation

1. Install OpenCV:
```bash
pip install opencv-python
```

2. Install ultralytics (for YOLO models):
```bash
pip install ultralytics
```

3. Install additional dependencies:
```bash
pip install numpy matplotlib
```

## Usage

### Real-time Detection Scripts
All real-time scripts support webcam input and can be run with:

```bash
# Image Filtering & Edge Detection
python Image_Filtering_Edge_Detection.py

# Object Detection
python Object_Detection.py

# Segmentation
python Segmentation.py

# Pose Estimation
python Pose_Estimation.py

# Face Detection
python Face_Detection.py
```

**Controls:**
- **Image Filtering:** Use `P`, `C`, `B`, `F` keys to switch filters; `Q` or `ESC` to exit
- **All other scripts:** Press `ESC` to exit
- Webcam feed is horizontally flipped for mirror effect

### Image Processing
```bash
# Process static images
python Image_Object_Detection.py
```

## Models

The scripts automatically download required models on first run:
- `yolo11n.pt` - Object detection (6.3MB)
- `yolo11n-seg.pt` - Instance segmentation 
- `yolo11n-pose.pt` - Pose estimation (6.0MB)
- `res10_300x300_ssd_iter_140000_fp16.caffemodel` - Face detection (Caffe model)

### Object Detection using OpenCV and pre-trained YOLO models (Yolo11n).

Hardware Requirements:
- A computer with a webcam (optional for real-time detection)
- Python 3.x

Software Requirements:
- OpenCV
- NumPy
- Pre-trained models (e.g., YOLOv5, YOLOv8)

### Segmentation using OpenCV and pre-trained YOLO models (Yolo11n-seg).
Hardware Requirements:
- A computer with a webcam (optional for real-time segmentation)
- Python 3.x

Software Requirements:
- OpenCV
- ultralytics
- NumPy
- Pre-trained segmentation models (automatically downloaded)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
