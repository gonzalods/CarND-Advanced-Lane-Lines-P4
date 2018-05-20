
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify undistort color image ("birds-eye view").
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image_1_1]: ./output_images/camera_cal_test.jpg "Origin Camera Calibration Example"
[image_1_2]: ./output_images/camera_cal_undistort.jpg "Undistort Camera Calibration Example"

[image_2_1]: ./output_images/distor_test.jpg "Origin Undistor Example"
[image_2_2]: ./output_images/undistor_test.jpg "Transformed Undistor Example"
[image_3_1]: ./output_images/bep_origin.jpg "Origin Bird-Eye Example"
[image_3_2]: ./output_images/bep_dest.jpg "Transformed Bird-Eye Example"

[image_4_1]: ./output_images/mask_yw_origin.jpg "Origin Binary Example"
[image_4_2]: ./output_images/mask_w_lab_b.jpg "LAB Channel B Binary Example"
[image_4_3]: ./output_images/mask_w_thresh.jpg "White Threshod Binary Example"
[image_4_4]: ./output_images/mask_y_xyz_z.jpg "XYZ Channel Z Binary Example"
[image_4_5]: ./output_images/mask_y_thresh.jpg "Yellow Threshod Binary Example"
[image_4_6]: ./output_images/mask_yw_thresh.jpg "Threshold Binary Example"

[image_5_1]: ./output_images/first_frame.png "Fit Lane Lines First Frame Example"
[image_5_2]: ./output_images/next_frame.png "Fit Lane Lines Next Frames Example"

[image_6_1]: ./output_images/output_frame.jpg "Output Frame Example"


All the code is in the Ipython notebook [project](./Advanced_Lane_Lines.ipynb). The code is structured in 4 classes:

* `CameraCalibrator`: Cell 2 
* `BirdEyeTransformer`: Cell 3
* `Line`: Cell 4
* `LaneLines`: Cell 5
---

### Camera Calibration

To calculate the calibration matrix of the camera and the distortion coefficients I created the class 'Calibrator'.

First, an `objpoints` matrix of 3D points `(x, y, z)` is created, where it is assumed that `z` equals `0`, which represent the intersections of the chess board
that you have to find out for each of the provided images to calibrate the camera.

An 'imgpoints' matrix of 2D points `(x, y)` is also created to store the intersections found in the provided images to calibrate the camera.

For each of the test images, in grayscale, the intersections of the chessboard are searched by invoking the method `cv2.findChessboardCorners()`, if you have found them, the 3D coordinates and the 2D intersections in the respective matrices are saved.

You get the matrix of the `mtx` camera and the distortion coefficients` dist` by invoking `cv2.calibrateCamera()` by passing the matrices `objpoints` and `imgpoints`.

|                       |                        |
|:---------------------:|:----------------------:|
|![alt text][image_1_1] |![alt text][image_1_2] |

### Pipeline

#### 1. Apply a distortion correction to raw images.

To undistor images I invoked `undistor` method of the `Calibrator` class. This method uses the `cv2.undistort ()` method, the camera matrix `mtx` and the distortion coefficients `dist` and returns the image without distortions.

|                       |                        |
|:---------------------:|:----------------------:|
|![alt text][image_2_1] |![alt text][image_2_2] |

#### 2. Apply a perspective transform to rectify undistort color image ("birds-eye view").

To obtain better color filters, I've first applied the perspective transform of the frames than the colors transform. This reduces the noise of the environment to better apply the color transform and obtain a good binary representation.

The perspective transformation is carried out by the class `BirdEyeTransformer`.

I have defined the following hardcoded source reference points:
```python
self.src = np.float32([[582, 460],
                       [705, 460],
                       [210, 720],
                       [1110, 720]])
```
The destination points are the following:
```python
self.offset = 325
self.dst = np.float32([[self.offset, 0],
                       [w - (self.offset), 0],
                       [self.offset, h ],
                       [w - self.offset, h ]])
```
where `h ` and `w` are the height and width of the image.

This resulted in the following source and destination points:

|Source     | Destination|
|:---------:|:----------:|
| 582, 460  | 325, 0     |
| 705, 460  | 955, 0     |
| 210, 720  | 325, 720   |
| 1110, 720 | 955, 720   |

These points are passed in to the `cv2.getPerspectiveTransform()` function to get the transformation matrix `M`.
To transform the perspective of the image, the `cv2.warpPerspective()` method is invoked, passing the image to it, the transformation matrix `M` and the size of the image (`h` and `w`).

The bird's-eye transformation is done by calling the `transform()` method of the class the class `BirdEyeTransformer`.

The inverse transformation, from bird-eye view to normal perspective, is done by invoking the `inv_transform()` method of the class the class `BirdEyeTransformer`.

The following images show the perspective tranformation by drawing the source and destination points onto them:

|                       |                        |
|:---------------------:|:----------------------:|
|![alt text][image_3_1] |![alt text][image_3_2] |

#### 3. Use color transforms, gradients, etc., to create a thresholded binary image.

In this implementation I have only used color transformations to create the binary image. The code is found in the method `color_and_gradient()` of the class `LaneLines`.

To obtain a binary image of the lane lines I used the `cv2.threshold()` method that requires a channel of a color scale of the image and the threshold that you want to apply. The channel is preprocessed by invoking `cv2.GaussianBlur()` to eliminate noise, using a `5x5` kernel and a standard deviation of `0`.

To filter the white lines I used the `Z` channel of the `XYZ` color scale.

To find the threshold to apply, I used a histogram of the `Z` channel of the image to obtain the proportion of pixels with values greater than `180`. I have used the following table of combersión to obtain the values of the threshold.

|Proportion           | theshold  |
|:-------------------:|:----------:|
| `> 0.25`            |   `220`    |
|`<= 0.25 & > 0.002`  |  `210`     |
|`<= 0.02 & > 0.001`  |  `190`     |
|`<= 0.01`            |  `180`     |


|                       |                        |                      |
|:---------------------:|:----------------------:|:---------------------:
|![alt text][image_4_1] |![alt text][image_4_2]  |![alt text][image_4_3]|

To filter the yellow lines I used channel 'B' of the 'LAB' color scale. To find out the threshold to apply, I applied the following formula, that results in a minimum threshold of `132`:
```python
    thresh_y = max(np.mean(lab_b) * ((100 + np.std(lab_b)) / 100), 132)
```
|                       |                        |                      |
|:---------------------:|:----------------------:|:---------------------:
|![alt text][image_4_1] |![alt text][image_4_4]  |![alt text][image_4_5]|

Finally I have merged the two filtered images by means of the `v2.bitwise_or()` method to obtain the final binary image.

|                       |                        |
|:---------------------:|:----------------------:|
|![alt text][image_4_1] |![alt text][image_4_6] |


#### 4. Detect lane pixels and fit to find the lane boundary.

The identification of the lane-line pixels and the fitting of their positions with a polynomial is performed by `Line` class.

First, two instances of the `Line` class are created, one for the left line and one for the right line.

In the first frame of the video you have to locate the pixels that represent the two lane-lines in the binary image.

For this we obtain a histogram of the bottom half of the first frame and divide it in half. Then, we obtain the maximum value of each half of the histogram that we will use as a base x coordinate for each lane line.

This first step of the detection of the pixels is done in the `pipeline()` method of the `LaneLines` class.
```python
    self.left_line = Line()
    self.right_line = Line()
    histogram = np.sum(transformed[binary_image.shape[0]//2:,:], axis=0)
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[115:midpoint]) + 115
    rightx_base = np.argmax(histogram[midpoint:(w - 15)]) + midpoint    
```
These x base coordinates are passed to the `get_first_line()` method of the corresponding instances of `Line`, where the rest of the location process is performed and the lane lines of the first frame fit.

The process of locating the pixels of the first frame is done in the following way. Starting from the base x coordinate we create some windows, centered on the base x coordinate, of determined size, in this case the binary image is divided into `9` windows of `100` pixels wide, and we look for the indexes of the pixels of the binary image whose value is not zero and they lay inside these windows.

```python
  x_current = x_base
  
  nwindows = 9
  window_height = np.int(h//nwindows)
  margin = 100
  
  win_y_low  = h - (window+1)*window_height
  win_y_high = h - window*window_height
  win_x_low  = x_current - (margin * factor)
  win_x_high = x_current + (margin * factor)
  
  good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                        (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]
``` 
If no index is found in a window, we extend the width of the next window in a certain cumulative `factor`, in this case `0.25`. Otherwise, we add the indices to the list of previous window indices and restar de `factor`.

```python
  if len(good_inds) == 0:
      factor += 0.25
  else:
      line_inds.append(good_inds)
      factor = 1
```
On the other hand, if the number of valid indices is higher than a threshold (`minpix`), `50` in this case, we calculate the average of the indices to recenter the window.
```python
  if len(good_inds) > minpix:
    x_current = np.int(np.mean(nonzerox[good_inds]))
```

Once obtained all the indices that fall inside the windows, we obtain the `x` and `y` coordinates of the pixels inside the windows and we fit a polynomial of second degree using the method `np.polyfit()`. This method returns the coefficients of the fitted second degree polynomial (`lane_fit`).
```python
  linex = nonzerox[line_inds]
  liney = nonzeroy[line_inds]
  
  self.line_fit = np.polyfit(liney, linex, 2)
```
![alt text][image_5_1]
The process of locating the pixels of the following frames is done in the `get_line()` method of the `Line` class. 

This process take adventage of the coefficients of the fitted polynomial of the previous frame to find the indexes of the pixels of the binary image whose value is not zero with a margin of 100.

```python
  nonzeroy = np.array(nonzero[0])
  nonzerox = np.array(nonzero[1])
  margin = 100
  line_inds = ((nonzerox > (self.line_fit[0]*(nonzeroy**2) + self.line_fit[1]*nonzeroy + 
              self.line_fit[2] - margin)) & (nonzerox < (self.line_fit[0]*(nonzeroy**2) + 
              self.line_fit[1]*nonzeroy + self.line_fit[2] + margin)))
```

If there is not a minimum number of line pixels in a certain margin at the top and bottom of the frame, we extend the line pixels with the last coordinates of the fitted polynomial in the previous frame to avoid that the new polynomial fit of the current frame can deviate a lot.

```python
    thresh_extra = 75 # top and bottom minimum number of line pixels
    extra_h = 10 # height of the extra line
    top_bottom_margin = 160 # top and bottom margin

    limit_down = h - top_bottom_margin
    nindx_down = len((liney[(liney > limit_down)]))
    if nindx_down <= thresh_extra:
        ploty_down = np.linspace(711 - extra_h , 711, 711 - extra_h)
        fitx_down = self.line_fit[0]*ploty_down**2 + self.line_fit[1]*ploty_down + self.line_fit[2]

        extrax_down = [x for y in fitx_down for x in range(int(y) - 10, int(y) + 11)]
        extray_down = np.concatenate([[y] * 21 for y in ploty_down])
        linex = np.append(linex, np.asarray(extrax_down, dtype=np.int32))
        liney = np.append(liney, np.asarray(extray_down, dtype=np.int32))

    limit_up = top_bottom_margin
    nindx_up = len((liney[(liney < limit_up)]))
    if nindx_up <= thresh_extra:
        ploty_up = np.linspace(0, extra_h - 1, extra_h)
        fitx_up = self.line_fit[0]*ploty_up**2 + self.line_fit[1]*ploty_up + self.line_fit[2]

        extrax_up = [x for y in fitx_up for x in range(int(y) - 10, int(y) + 11)]
        extray_up = np.concatenate([[y] * 21 for y in ploty_up])
        linex = np.append(np.asarray(extrax_up, dtype=np.int32), linex)
        liney = np.append(np.asarray(extray_up, dtype=np.int32), liney)
```

Then we fit a second-degree polynomial using the `np.polyfit()` method. This method returns the coefficients of the fitted second degree polynomial.

![alt text][image_5_2]

#### 5. Determine the curvature of the lane and vehicle position with respect to center.

The calculation of the curvature of the lane and the position of the vehicle with respect to the center of the lane is done by the `get_curvature_and_position ()` method of the `LaneLine` class.

To calculate the curvature of the lane, you must first convert the measurements into pixels in units of the real world, meters in this case. For this, it is assumed that in the bird-eye view the 720 pixels of the y coordinate correspond to 28 meters and the lane width, the difference between the last coordinate of the right and left polynomial, are 3.7 meters.

First you get the maximum value of the `y` coordinate, and you fit the polynomials with the coefficients obtained previously.
```python
    y_eval = np.max(ploty)

    # Fit polinomials to x, y in pixel space
    left_fitx, left_ploty = self.get_good_line(left_fit, ploty)
    right_fitx, right_ploty = self.get_good_line(right_fit, ploty)
```
Next, you convert the measurements into pixels in measures in meters
```python
    # Calcute x and y in meters per pixel
    ym_per_pix = 28/(y_eval + 1) # meters per pixel in y dimension
    xm_per_pix = 3.7/(right_fitx[-1] - left_fitx[-1]) # meters per pixel in x dimension
```
Then, you re-fit a polynomial with coordinates of the real world space.
```python
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
```
And finally, you get the curvature in meters of the lane averaging the curvature of the two lane lines calculated in meters using the equation 

```python
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    curverad = (left_curverad + right_curverad) / 2.0
```

The position of the vehicle with respect to the center of the lane is calculated by subtracting the displacement used in the corversion from bird-eye view to the last `x` coordinate of the polynomial of the left line, multiplied by the conversion of pixels to meters in the `x` coordinate.
```python
    shift = (left_fitx[-1] - self.transformer.get_offset()) * xm_per_pix
```
If the displacement is positive, the vehicle is displaced to the left, if it is negative to the right, and if it is zero it is centered.

#### 6. Warp the detected lane boundaries back onto the original image.

The display of the detected limits of the lane is made in the `get_lane_region()` method of the `LaneLines` class.

First we create a blank mask of the frame and we generate the polynomials that represent the lane lines.
```python
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((img, img, img))*255
    window_img = np.zeros_like(out_img)

    # Generate a polygon to illustrate the search window area
    ploty = np.linspace(0, h-1, h)
    left_fitx, left_ploty = self.get_good_line(left_fit, ploty)
    right_fitx, right_ploty = self.get_good_line(right_fit, ploty)
```

Next, we recast the x and y coordinates into usable format for `cv2.fillPoly()`.
```python
    # Recast the x and y points into usable format for cv2.fillPoly()
    left_line_window = np.array([np.transpose(np.vstack([left_fitx, left_ploty]))])
    right_line_window = np.array([np.flipud(np.transpose(np.vstack([right_fitx, right_ploty])))])
    line_pts = np.hstack((left_line_window, right_line_window))    
```

And finally, we draw the green region onto the warped blank image
```python
    # Draw the region onto the warped blank image
    cv2.fillPoly(window_img, np.int_(line_pts), (0,255, 0))
```

The last step in the pipeline is performed in the the `create_output_frame()` method of the `LaneLines` class.

First, we unwarp the lane region using the `inv_transform ()` method of the `BirdEyeTransformer` class, which will use the same transformation matrix, and source and destion points that were used to warp the original frame.
```python
    inv = self.transformer.inv_transform(region)
```
Then, we merge the original frame with the semi-transparent image of the lane region using the `cv2.addWeighted()` method. And finally, we write the text of the curvature and position in the frame with the method `cv2.putText ()`
```python
    result = cv2.addWeighted(img, 1, inv, 0.3, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    shift_text = 'Vehicle is in the center of lane'
    if shift < 0.0:
        shift_text = 'Vehicle is {:.2f}m right of center'.format(abs(shift))
    elif shift > 0.0:
        shift_text = 'Vehicle is {:.2f}m left of center'.format(shift)
    radius_text = 'Radius of curvature = {:.0f} (m)'.format(radius)
    cv2.putText(result,radius_text,(10,40), font, 1.3,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(result,shift_text,(10,90), font, 1.3,(255,255,255),2,cv2.LINE_AA)
```
This is a frame example:

![alt text][image_6_1]

---

### Pipeline (video)

Here's a [link to project video result](./output_videos/project_video.mp4)

This es a [link to challenge video result](./output_videos/challenge_video.mp4)

---

### Discussion

The biggest difficulty I have found is to obtain a clear binary image that will clearly determine the lane lines. I have tried different combinations of color transformations; color spaces and channels, and different gradients; horizontal, magnitude, direction and thresholds. The combination I have chosen has an acceptable performance for the first two videos, but it does not work for the hardest video. 
