[image1]: ./Images/corners.png "Corners Chessboard"
[image2]: ./Images/undistorted.png "Undistorted Input"
[image3]: ./Images/perspective.png "perspective image"
[image4]: ./Images/binary.png "binary threshold images"
[image5]: ./Images/detected_lines.png "detected lines images"
[image6]: ./Images/output.png "output images"
[image7]: ./Images/project.gif "origin video"
[image8]: ./Images/project_result.gif "project video"
[image9]: ./Images/challenge_result.gif "challenge video"

# Lane Detection
The goal of this project is to identify lane lines, vehicle position, and radius of curvature from a video stream. To do this, it is important to first define a pipeline to process still images.
| Original Video            | Result Video                  |
| ------------------------- | ----------------------------- |
| ![Original Video][image7] | ![Final Result Video][image8] |
## Steps
  1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
  2. Apply distortion correction to input images.
  3. Apply a perspective transformation to rectify input image.
  4. Use color transforms to create a binary thresholded image. 
  5. Identify the lane line pixels and fit polynomials to the lane boundaries.
  6. Determine the curvature of the lane and vehicle position with respect to center.
  7. Output visual display of the lane boundaries and estimation of lane curvature and vehicle position.

 ## Code
 image.py contains the source code to process an image

 video.py contains the source code to process a video 
 ### Dependencies
- [OpenCV](http://opencv.org/)
- [NumPy](http://www.numpy.org/)
- [matplotlib](http://matplotlib.org/)
- [MoviePy](http://zulko.github.io/moviepy/)

## Still Image Processing Pipeline
### Step 1&2: Camera Calibration and Distortion Correction
 OpenCV functions `cv2.findChessboardCorners()` and `cv2.drawChessboardCorners()` are used to identify the locations of corners on a series of pictures of a chessboard taken from different angles, and generate the 2D *imgpoint*, and the 3D *objpoints*.![Original Image, and Image with detected Corners][image1]
 Then the resulting *imgpoints* and *objpoints* are used to compute the *camera calibration* and *distortion coefficients* using the `cv2.calibrateCamera()` function. The *camera calibration* and *distortion coefficients* are then used to undistort the input image using `cv2.undistort()`.

 ![Original Image, and undistorted Image][image2]

 Notice that if you compare the two images around the edges, there are obvious differences between the original and undistorted image.
 
 ### Step 3: Perspective Transform
 This step is used to transform the undistorted image to a "birds eye view" of the road which focuses only on the lane lines and displays them in such a way that they appear to be relatively parallel to eachother. To achieve the perspective transformation the OpenCV functions `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()` which take a matrix of four source points on the undistorted image and remaps them to four destination points on the warped image. The source and destination points were selected manually by visualizing the locations of the lane lines on a series of test images.
 
 ![Original Image, and bird view Image][image3]

 ### Step 4: Binary Threshold Image
 In this step convert the warped image to different color spaces and create binary thresholded images which highlight only the lane lines and ignore everything else. The following color channels and thresholds did a good job of identifying the lane lines on the test images:

- The L Channel from the LUV color space, with a min threshold of 195 and a max threshold of 255, did an almost perfect job of picking up the white lane lines, but completely ignored the yellow lines.
- The B channel from the Lab color space, with a min threshold of 140 and an upper threshold of 200, did a good job in identifying the yellow lines, but completely ignored the white lines.
   
 A combined binary threshold based on the two above mentioned binary thresholds was used to create one combination thresholded image, which does a great job of highlighting almost all of the white and yellow lane lines.
 
 ![ Binary threshold images][image4]

 ### Step 5: Detect Lane Lines
 To detect the lane lines the following approach was applied on the binary wraped image:
#### Peaks in a Histogram and Sliding Windows
* The two highest peaks from the histogram are used as a starting point for determining where the left, and right lane lines are, then the sliding windows approach is used moving upward in the image to determine where the lane lines go.
* If the lane pixels were previously detected in last frame, a quick search could be applied based on last detected x/y pixel positions with a proper margin.
##### Steps:
  1. Split the histogram into two sides, one for each lane line.
  2. Set up sliding windows and window hyperparameters:
     * set a few hyperparameters related to our sliding windows, and set them up to iterate across the binary activations in the image. These hyperparameters are:
        1. **W_Number**; number of sliding windows.
        2. **Margin**; the width of each window.
        3. **Minimum_pixels**; used as a threshold to recenter the next sliding window.
        4. **Window_Height**; computed from number of pixels and image height.
  3. Loop through each window in W_Number.
  4. Find the boundaries of our current window. This is based on a combination of the current window's starting point, as well as the margin set in the hyperparameters.
  5. `cv2.rectangle()` is used to draw these window boundaries onto the visualization image.
  6. Within the boundaries of the window, find which activated(non zero) pixels actually fall into the window.
  7. Append non zero pixels to two different arrays one for the right line and the other for the left line.
  8. If the number of pixels you found in Step **6** are greater than your hyperparameter Minimum_pixels, re-center the next frame window based on the mean position of these pixels.
   
Then a 2nd order polynomial line is fitted on the detected lane lines. ![ Detected Lines][image5]

### Step 6: Lane Curvature and Vehicle Position
Using the x and y pixel positions of the lane line pixels compute the radius of the curvature, and the vehicle position

**Radius of Curvature Equation:**

$$f(y) = y^2+By+c $$

$$R\_Curve = \frac{[1+(\frac{dx}{dy})^2]^{3/2}}{|\frac{d^2x}{dy^2}|}$$

$$f'(y) = \frac{dx}{dy} = 2Ay+B$$

$$f''(y) = \frac{d^2x}{dy^2} = 2A$$

**Position of Vehicle:**

To connect pixel unit with real world meter unit, A conversions in x and y from pixels space to meters is defined. In order to calculate precisely.

- Calculated the average of the x intercepts from each of the two polynomials `position = (rightx_int+leftx_int)/2`
- Calculated the distance from center by taking the absolute value of the vehicle position minus the halfway point along the horizontal axis `distance_from_center = abs(image_width/2 - position)`
- If the horizontal position of the car was greater than `image_width/2` than the car was considered to be left of center, otherwise right of center. 

### Step 7: Output Visual Display
The final step in processing the images is to plot the polynomials on to the warped image, fill the space between the polynomials to highlight the lane that the car is in, use the inverse perspective trasformation to unwarp the image from birds eye back to its original perspective, and print the distance from center and radius of curvature on to the final annotated image.![ Detected Lines][image6]

## Video Processing Pipeline

After establishing a pipeline to process still images, the final step is to expand the pipeline to process videos frame-by-frame, to simulate what it would be like to process an image stream in real time on an actual vehicle.

The goal in developing a video processing pipeline is to create as smooth of an output as possible. To achieve this, the **Line** class created a class for each of the left and right lane lines and it stores features of each lane for averaging across frames.

The video pipeline first checks whether or not the lane was detected in the previous frame. If it was, then it only checks for lane pixels in close proximity to the polynomial calculated in the previous frame. This way, the pipeline does not need to scan the entire image, and the pixels detected have a high confidence of belonging to the lane line because they are based on the location of the lane in the previous frame.

If at any time, the pipeline fails to detect lane pixels based on the the previous frame, it will go back in to blind search mode and scan the entire binary image for nonzero pixels to represent the lanes, using the peaks in a histogram and sliding windows approach.

| Project Video                 | Challenge Video            |
| ----------------------------- | -------------------------- |
| ![Final Result Video][image8] | ![Challenge Video][image9] |