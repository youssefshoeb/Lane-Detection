import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip
from camera_calibration import Camera
from lines import Line
from detect_lanes_utils import *


def process_image(img, camera, left, right):
    """Processing each image in video pipeline

    Arguments:
        img -- Input Image
        camera -- camera
        left -- x coordinates of left lane pixels
        right -- x coordinates of right lane pixels

    Returns:
        result -- Image with drawn lane
    """
    # undistort image
    undist_img = camera.undistort(img)

    # get perspective View
    wraped = c.birds_eye(undist_img, display=False)

    # apply color threshold on image
    warped_binary = apply_thresholds(wraped, display=False)

    nonzerox, nonzeroy = np.nonzero(np.transpose(warped_binary))

    # Get lanes
    if left.detected is True:
        leftx, lefty, left.detected = left.quick_search(nonzerox, nonzeroy)
    if right.detected is True:
        rightx, righty, right.detected = right.quick_search(nonzerox, nonzeroy)
    if left.detected is False:
        leftx, lefty, left.detected = left.blind_search(
            nonzerox, nonzeroy, warped_binary)
    if right.detected is False:
        rightx, righty, right.detected = right.blind_search(
            nonzerox, nonzeroy, warped_binary)

    left.y = np.array(lefty).astype(np.float32)
    left.x = np.array(leftx).astype(np.float32)
    right.y = np.array(righty).astype(np.float32)
    right.x = np.array(rightx).astype(np.float32)

    # Based on searched x and y coordinates, polyfit with second order.
    left_fit, left_fitx, left_fity = left.get_fit()
    right_fit, right_fitx, right_fity = right.get_fit()

    # Visualize Results
    image_shape = img.shape
    ploty = np.linspace(0, image_shape[0] - 1, image_shape[0])
    offset, mean_curv = get_car_pos(
        left_fit, right_fit, ploty, (image_shape[0], image_shape[1]))

    result = draw_lane(undist_img, warped_binary, left_fit,
                       right_fit, camera, display=False)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text1 = 'Radius of Curvature: %d(m)'
    text2 = 'Offset from center: %.2f(m)'
    text3 = 'Radius of Curvature: Inf (m)'

    if mean_curv < 3000:
        cv2.putText(result, text1 % (int(mean_curv)),
                    (60, 100), font, 1.0, (255, 255, 255), thickness=2)
    else:
        cv2.putText(result, text3,
                    (60, 100), font, 1.0, (255, 255, 255), thickness=2)
    cv2.putText(result, text2 % (-offset),
                (60, 130), font, 1.0, (255, 255, 255), thickness=2)

    return result


c = Camera(display=False)

# Video Pipline
frame_num = 15   # latest number of frames for good detection
left = Line(False, frame_num)
right = Line(True, frame_num)

video_output = 'result.mp4'
input_path = './test_videos/project_video.mp4'

clip1 = VideoFileClip(input_path).subclip(0, 30)

final_clip = clip1.fl_image(lambda image: process_image(image, c, left, right))
final_clip.write_videofile(video_output, audio=False)
