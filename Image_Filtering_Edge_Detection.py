import cv2
import sys
import numpy

# Define filter modes
PREVIEW  = 0  # Preview Mode
BLUR     = 1  # Blurring Filter
FEATURES = 2  # Corner Feature Detector
CANNY    = 3  # Canny Edge Detector

# Feature detection parameters
maxCorners:int = 500
feature_params = dict(maxCorners=maxCorners, qualityLevel=0.2, minDistance=15, blockSize=9)

# Get video source from command line argument or default to camera 0
s = 0
if len(sys.argv) > 1:
    s = sys.argv[1]

# Initialize video source
image_filter = PREVIEW
alive = True

# Create window for display
win_name = "Camera Filters"
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
result = None

# Initialize video capture
source = cv2.VideoCapture(s)

# Main processing loop
while alive:
    has_frame, frame = source.read()
    if not has_frame:
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)

    # Apply selected filter
    if image_filter == PREVIEW:
        result = frame
    elif image_filter == CANNY:
        result = cv2.Canny(frame, 50, 150)
    elif image_filter == BLUR:
        result = cv2.blur(frame, (13, 13))
    elif image_filter == FEATURES:
        result = frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(frame_gray, 
                                        maxCorners=feature_params['maxCorners'], # type: ignore
                                        qualityLevel=feature_params['qualityLevel'], 
                                        minDistance=feature_params['minDistance'], 
                                        blockSize=feature_params['blockSize']) # type: ignore
        if corners is not None:
            for x, y in numpy.float32(corners).reshape(-1, 2):
                cv2.circle(result, (int(x), int(y)), 10, (0, 255, 0), 1)

    cv2.imshow(win_name, result) # type: ignore

    # Handle key inputs
    key = cv2.waitKey(1)
    if key == ord("Q") or key == ord("q") or key == 27:
        alive = False
    elif key == ord("C") or key == ord("c"):
        image_filter = CANNY
    elif key == ord("B") or key == ord("b"):
        image_filter = BLUR
    elif key == ord("F") or key == ord("f"):
        image_filter = FEATURES
    elif key == ord("P") or key == ord("p"):
        image_filter = PREVIEW

source.release()
cv2.destroyWindow(win_name)