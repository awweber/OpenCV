import os
import sys
import cv2 as cv
import matplotlib.pyplot as plt
from ultralytics import YOLO
import time

# ====================================================================
# Object Detection using Pre-trained DNN model in OpenCV
# ====================================================================
def prediction(model_type, img_path, display_result = False, task : str = None):

    model = YOLO(model_type)
    result = model(img_path, save = True, conf=0.5)

    for r in result:
        # Use just the image name for the predicted image path since YOLO saves it with original name
        pred_img_path = f"{r.save_dir}/{img_path.split('/')[-1]}"
        pred = cv.cvtColor(cv.imread(pred_img_path), cv.COLOR_BGR2RGB)
        plt.imshow(pred)
        plt.axis('off')
        plt.title(f"YOLO11 - {task}")
    plt.show()

    # To print the results
    if display_result:
       print(result)


# Load the pre-trained YOLOv11-Nano model
img_name = "bild.jpg"
model_type = "yolo11n.pt"
img_path = f"data/{img_name}"

start_time = time.time()
prediction(model_type, img_path, task = "Object Detection")
end_time = time.time()

print(f"YOLOv11-Nano Inference Time: {end_time - start_time:.2f} seconds")