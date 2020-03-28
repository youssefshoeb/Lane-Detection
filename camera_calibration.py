import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

board_w = 9
board_h = 6
board_size = (board_w, board_h)
board_n = board_w * board_h


class Camera():
    def __init__(self, display=False):
        img_shape = (0, 0)
        obj = []
        for ptIdx in range(0, board_n):
            obj.append(
                np.array([[ptIdx/board_w, ptIdx % board_w, 0.0]], np.float32))
        obj = np.vstack(obj)
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        images = glob.glob('camera_calibration/calibration*')

        for idx, fname in enumerate(images):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            found, corners = cv2.findChessboardCorners(
                gray, board_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH |
                cv2.CALIB_CB_FILTER_QUADS)

            # If found, add object points, image points
            if found:
                cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                 (cv2.TermCriteria_EPS |
                                  cv2.TERM_CRITERIA_MAX_ITER,
                                  30, 0.1))
                objpoints.append(obj)
                imgpoints.append(corners)
            if display:
                # Draw and display the corners
                cv2.drawChessboardCorners(img, board_size, corners, found)
                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
                ax1.imshow(cv2.cvtColor(
                    mpimg.imread(fname), cv2.COLOR_BGR2RGB))
                ax1.set_title('Original Image', fontsize=18)
                ax2.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax2.set_title('With Corners', fontsize=18)
                plt.show()

        self.objpoints = objpoints
        self.imgpoints = imgpoints

    def undistort(self, img, display=False):
        """Undistort a given image

        Arguments:
            img {cv2.imread} -- input image

        Keyword Arguments:
            display {bool} -- flag to display the results (default: {False})

        Returns:
            undist -- undistorted image
        """
        img_size = (img.shape[1], img.shape[0])

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, img_size, None, None)

        undist = cv2.undistort(img, mtx, dist, None, mtx)
        if display:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            ax1.imshow(img)
            ax1.set_title('Original Image', fontsize=18)
            ax2.imshow(undist)
            ax2.set_title('Undistorted', fontsize=18)
            plt.show()

        return undist

    # Perform perspective transform
    def birds_eye(self, img, display=False, inverse=False):
        """Perform perspective transfrom

        Arguments:
            img {cv2.imread} -- input image

        Keyword Arguments:
            display {bool} -- flag to display the result (default: {False})
            inverse {bool} -- flag to perform the inverse
                              function (default: {False})

        Returns:
            warped -- perspective view of the image
        """

        img_size = (img.shape[1], img.shape[0])
        offset = 0
        src = np.float32([[490, 482], [810, 482],
                          [1250, 720], [40, 720]])
        dst = np.float32([[0, 0], [1280, 0],
                          [1250, 720], [40, 720]])
        if inverse:
            M = cv2.getPerspectiveTransform(dst, src)
        else:
            M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(img, M, img_size)
        if display:
            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            f.tight_layout()
            ax1.imshow(img)
            ax1.set_title('Normal View ', fontsize=18)
            ax2.imshow(warped)
            ax2.set_title('Perspective View', fontsize=18)
            plt.show()

        return warped
