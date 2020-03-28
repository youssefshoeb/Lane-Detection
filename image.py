import matplotlib.image as mpimg
from camera_calibration import Camera
from detect_lanes_utils import *


c = Camera(display=False)

# Image Pipline
img = mpimg.imread('./test_images/test3.jpg')
undist = c.undistort(img, display=True)
wraped = c.birds_eye(undist, display=True)

result = apply_thresholds(wraped, display=True)
left, right = detect_lines(result, display=True)
draw_lane(undist, result, left, right, c, display=True)
