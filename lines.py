import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import deque
from detect_lanes_utils import *


class Line:
    """
    A class to receive the characteristics of each line detection
    """

    def __init__(self, right_lane, frame_num):
        # was the line detected in the last iteration?
        self.detected = False
        self.right = right_lane

        # x and y values in last frame
        self.x = None
        self.y = None

        # x intercepts for average smoothing
        self.bottom_x = deque(maxlen=frame_num)
        self.top_x = deque(maxlen=frame_num)

        # Record last x intercept
        self.current_bottom_x = None
        self.current_top_x = None

        # Record radius of curvature
        self.radius = None

        # Polynomial coefficients: x = A*y**2 + B*y + C
        self.A = deque(maxlen=frame_num)
        self.B = deque(maxlen=frame_num)
        self.C = deque(maxlen=frame_num)
        self.fit = None
        self.fitx = None
        self.fity = None

    def get_curv(self):
        self.radius = get_curvature(self.fit)
        return self.radius

    def get_intercepts(self):
        bottom = self.fit[0] * 720 ** 2 + self.fit[1] * 720 + self.fit[2]
        top = self.fit[2]
        return bottom, top

    def quick_search(self, nonzerox, nonzeroy):
        """
        Assuming in last frame, lane has been detected.
        Based on last x/y coordinates, quick search current lane.
        """
        x_inds = []
        y_inds = []
        if self.detected:
            win_bottom = 720
            win_top = 630
            while win_top >= 0:
                yval = np.mean([win_top, win_bottom])
                xval = (np.median(self.A)) * yval ** 2 + \
                    (np.median(self.B)) * yval + (np.median(self.C))
                x_idx = np.where((((xval - 50) < nonzerox)
                                  & (nonzerox < (xval + 50))
                                  & ((nonzeroy > win_top)
                                     & (nonzeroy < win_bottom))))
                x_window, y_window = nonzerox[x_idx], nonzeroy[x_idx]
                if np.sum(x_window) != 0:
                    np.append(x_inds, x_window)
                    np.append(y_inds, y_window)
                win_top -= 90
                win_bottom -= 90

        # If no lane pixels were detected then perform blind search
        if np.sum(x_inds) == 0:
            self.detected = False
        return x_inds, y_inds, self.detected

    def blind_search(self, nonzerox, nonzeroy, image):
        """
        Sliding window search method, start from blank.
        """
        x_inds = []
        y_inds = []
        if self.detected is False:
            win_bottom = 720
            win_top = 630
            while win_top >= 0:
                histogram = np.sum(image[win_top:win_bottom, :], axis=0)
                if self.right:
                    base = np.argmax(histogram[640:]) + 640
                else:
                    base = np.argmax(histogram[:640])
                x_idx = np.where((((base - 50) < nonzerox) &
                                  (nonzerox < (base + 50)) &
                                  ((nonzeroy > win_top) &
                                      (nonzeroy < win_bottom))))
                x_window, y_window = nonzerox[x_idx], nonzeroy[x_idx]
                if np.sum(x_window) != 0:
                    x_inds.extend(x_window)
                    y_inds.extend(y_window)
                win_top -= 90
                win_bottom -= 90
        if np.sum(x_inds) > 0:
            self.detected = True
        else:
            y_inds = self.y
            x_inds = self.x
        return x_inds, y_inds, self.detected

    def sort_idx(self):
        """
        Sort x and y according to y index
        """
        sorted_idx = np.argsort(self.y)
        sorted_x_inds = self.x[sorted_idx]
        sorted_y_inds = self.y[sorted_idx]

        return sorted_x_inds, sorted_y_inds

    def get_fit(self):
        """
        Based on searched x and y coordinates, polyfit with second order.
        Take median value in previous frames to smooth.
        """
        self.fit = np.polyfit(self.y, self.x, 2)

        self.current_bottom_x, self.current_top_x = self.get_intercepts()

        self.bottom_x.append(self.current_bottom_x)
        self.top_x.append(self.current_top_x)
        self.current_bottom_x = np.median(self.bottom_x)
        self.current_top_x = np.median(self.top_x)

        self.x = np.append(self.x, self.current_bottom_x)
        self.x = np.append(self.x, self.current_top_x)
        self.y = np.append(self.y, 720)
        self.y = np.append(self.y, 0)

        self.x, self.y = self.sort_idx()
        self.fit = np.polyfit(self.y, self.x, 2)
        self.A.append(self.fit[0])
        self.B.append(self.fit[1])
        self.C.append(self.fit[2])
        self.fity = self.y
        self.fit = [np.median(self.A), np.median(self.B), np.median(self.C)]
        self.fitx = self.fit[0] * self.fity ** 2 + \
            self.fit[1] * self.fity + self.fit[2]

        return self.fit, self.fitx, self.fity
