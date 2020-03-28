import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from camera_calibration import Camera


def apply_thresholds(img, display=False):
    """Create binary thresholded images to isolate lane line pixels

    Arguments:
        image {cv2.imread} -- input image

    Keyword Arguments:
        display {bool} -- flag to display the results (default: {False})

    Returns:
        combined_binary -- resulting image
    """
    # Color Channels
    l_channel = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)[:, :, 0]
    b_channel = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)[:, :, 2]

    # Thresholds
    b_thresh_min = 140
    b_thresh_max = 200
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_thresh_min) & (b_channel <= b_thresh_max)] = 1

    l_thresh_min = 195
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    combined_binary = np.zeros_like(l_binary)
    combined_binary[(l_binary == 1) | (
        b_binary == 1)] = 1

    # Plotting thresholded images
    if display:
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
            2, 2, sharey='col', sharex='row', figsize=(12, 6))
        f.tight_layout()

        ax1.set_title('Input Image', fontsize=12)
        ax1.imshow(img)

        ax2.set_title('B threshold', fontsize=12)
        ax2.imshow(b_binary, cmap='gray')

        ax3.set_title('L threshold', fontsize=12)
        ax3.imshow(l_binary, cmap='gray')

        ax4.set_title('Combined color thresholds', fontsize=16)
        ax4.imshow(combined_binary, cmap='gray')
        plt.show()

    return combined_binary


def detect_lines(img, W_Number=9, margin=100, minpix=40,
                 display=False):
    """detect lanes using peaks in a histogram and sliding window approach

    Arguments:
        img {[type]} -- input image (bird view)

    Keyword Arguments:
        W_Number {int} -- number of sliding windows (default: {9})
        margin {int} -- the width of each window (default: {100})
        minpix {int} -- used as a threshold to recenter the
                        next sliding window (default: {40})
        display {bool} -- flag to display the results (default: {False})

    Returns:
        left_fit -- x coordinates of left lane
        right_fit -- x coordinates of right lane
    """
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[int(img.shape[0]/2):, :], axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    left_base = np.argmax(histogram[:midpoint])
    right_base = np.argmax(histogram[midpoint:]) + midpoint

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    left_current = left_base
    right_current = right_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_idx = []
    right_lane_idx = []

    # Set height of windows
    window_height = np.int(img.shape[0]/W_Number)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img))*255

    # Step through the windows one by one
    for window in range(W_Number):

        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = left_current - margin
        win_xleft_high = left_current + margin
        win_xright_low = right_current - margin
        win_xright_high = right_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (255, 0, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (255, 0, 0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_idx = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                         & (nonzerox >= win_xleft_low) &
                         (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_idx = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high)
                          & (nonzerox >= win_xright_low) &
                          (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_idx.append(good_left_idx)
        right_lane_idx.append(good_right_idx)

        # If found pixels > threshold, recenter next window
        #  on their mean position
        if len(good_left_idx) > minpix:
            left_current = np.int(np.mean(nonzerox[good_left_idx]))
        if len(good_right_idx) > minpix:
            right_current = np.int(np.mean(nonzerox[good_right_idx]))

    # Concatenate the arrays of indices
    left_lane_idx = np.concatenate(left_lane_idx)
    right_lane_idx = np.concatenate(right_lane_idx)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_idx]
    lefty = nonzeroy[left_lane_idx]
    rightx = nonzerox[right_lane_idx]
    righty = nonzeroy[right_lane_idx]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Visualization Finding the lines
    if display:

        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + \
            right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_idx],
                nonzerox[left_lane_idx]] = [0, 255, 0]
        out_img[nonzeroy[right_lane_idx],
                nonzerox[right_lane_idx]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='white')
        plt.plot(right_fitx, ploty, color='white')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()

    return left_fit, right_fit


def get_curvature(fit, ploty):
    """Measure Radius of Curvature for a given lane line

    Arguments:
        fit -- x coordinates of the given lane

    Returns:
        curvature -- curvature of the given lane line (in meters unit)
    """

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 18 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 70  # meters per pixel in x dimension
    y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]
    fit_cr = np.polyfit(ploty*ym_per_pix, fitx*xm_per_pix, 2)

    # Calculate the new radii of curvature
    curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2)
                ** 1.5) / \
        np.absolute(2 * fit_cr[0])

    return curverad


def get_car_pos(left_fit, right_fit, ploty, img_shape):
    """Calculate the position of car on left and right lane base
     (converted to real unit meter)

    Arguments:
        left_fit -- x coordinates of the left lane
        right_fit -- x coordinates of the right lane
        ploty -- y coordinates of the lane
        img_shape -- image shape (2D)

    Returns:
        offset -- distance (meters) of car offset from the middle
                of left and right lane
        mean -- mean curvature of both lanes
    """

    xleft_eval = left_fit[0] * np.max(ploty) ** 2 + \
        left_fit[1] * np.max(ploty) + left_fit[2]
    xright_eval = right_fit[0] * np.max(ploty) ** 2 + \
        right_fit[1] * np.max(ploty) + right_fit[2]

    # meters per pixel in x dimension
    xm_per_pix = 3.7 / abs(xleft_eval - xright_eval)
    xmean = np.mean((xleft_eval, xright_eval))

    # +: car in right; -: car in left side
    offset = (img_shape[1]/2 - xmean) * xm_per_pix

    left_curverad = get_curvature(left_fit, ploty)

    right_curverad = get_curvature(right_fit, ploty)

    mean_curv = np.mean([left_curverad, right_curverad])

    return offset, mean_curv


def draw_lane(img, bird_eye, left_fit, right_fit, camera, display=False):
    """Draw Lane, radius of curvature, and position of vehicle

    Arguments:
        img -- street view of image
        bird_eye {[type]} -- bird_eye view of image (binary)
        left_fit {[type]} -- x coordinates of left lane
        right_fit {[type]} -- x coordinates of right lane

    Keyword Arguments:
        display {bool} -- flag to display output (default: {False})

    Returns:
        output -- output image
    """
    tmp_image = np.copy(img)
    if right_fit is None or left_fit is None:
        return img

    zero = np.zeros_like(bird_eye).astype(np.uint8)
    layered_image = np.dstack((zero, zero, zero))

    ploty = np.linspace(0, bird_eye.shape[0]-1, bird_eye.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # formatting the points
    left = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
    right = np.array(
        [np.flipud(np.transpose(np.vstack([left_fitx, ploty])))])
    points = np.hstack((left, right))

    # form lane
    cv2.fillPoly(layered_image, np.int_([points]), (0, 255, 0))
    cv2.polylines(layered_image, np.int32(
        [right]), isClosed=False, color=(255, 0, 0), thickness=20)
    cv2.polylines(layered_image, np.int32(
        [left]), isClosed=False, color=(255, 0, 0), thickness=20)

    # Use the inverse perspective transform back to street view
    inversed = camera.birds_eye(layered_image, inverse=True)

    # Combine the result with the original image
    output = cv2.addWeighted(tmp_image, 1, inversed, 0.5, 0)

    if display:
        left_curverad = get_curvature(left_fit, ploty)
        right_curverad = get_curvature(right_fit, ploty)

        center, mean = get_car_pos(left_fit, right_fit, ploty, img.shape)
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 6))
        ax1.imshow(img)
        ax1.set_title('Original Image', fontsize=20)
        ax2.imshow(output)
        ax2.set_title('Image With Drawn Lane', fontsize=20)
        if center < 0:
            ax2.text(200, 100, 'Vehicle is {:.2f}m left of center'.
                     format(np.abs(center)),
                     style='italic', color='white', fontsize=10)
        else:
            ax2.text(200, 100, 'Vehicle is {:.2f}m right of center'.format
                     (np.abs(center)),
                     style='italic', color='white', fontsize=10)
        ax2.text(200, 175, 'Radius of curvature is {}m'.format(
            int((left_curverad + right_curverad)/2)),
            style='italic', color='white', fontsize=10)
        plt.show()

    return output
