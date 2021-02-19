## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/image1.jpg "Undistorted"
[image2]: ./examples/image2.jpg "Road Transformed"
[image3]: ./examples/image3.jpg "Binary Example"
[image4]: ./examples/image4.jpg "Warp Example"
[image5]: ./examples/image5.jpg "Fit Visual"
[image6]: ./examples/image6.jpg "Output"
[image7]: ./examples/image_pitfall.jpg "Pitfall"
[video1]: ./project_video_out.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb".  


```python
def undistort(img,objpoints, imgpoints):
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)


    dst = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('../output_images/camera_cal_out.jpg',dst)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump( dist_pickle, open( "../output_images/wide_dist_pickle.p", "wb" ) )
    return img,dst
```
I start by preparing "object points(objpoints)", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
The procedures are basically same with above. This time we apply the undistortion operation on real image of the road.
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
```python
    img,mag_binary = thresh_image(img,ksize=3,threshx=(0,255),threshy=(0,255),thresh_mag=(50, 200))
    # img is the original image, mag_binary is the magnitude of x_derivative and y_derivative
```
I used a combination of color and gradient thresholds to generate a binary image ().  Here's an example of my output for this step.  

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.
```python
top_down, perspective_M, Minv = corners_unwarp(mag_binary, mtx, dist)
```
The code for my perspective transform includes a function called `corners_unwarp(mag_binary, mtx, dist)` I chose the hardcode the source and destination points in the following manner:

```python
        src = np.float32(
            [[(img_size[0] / 2) - 63, img_size[1] / 2 + 100],
            [((img_size[0] / 6) - 20), img_size[1]],
            [(img_size[0] * 5 / 6) + 60, img_size[1]],
            [(img_size[0] / 2 + 65), img_size[1] / 2 + 100]])
        dst = np.float32(
            [[(img_size[0] / 4), 0],
            [(img_size[0] / 4), img_size[1]],
            [(img_size[0] * 3 / 4), img_size[1]],
            [(img_size[0] * 3 / 4), 0]])
```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 557, 460      | 320, 0        | 
| 193.33, 720   | 320, 720      |
| 1126.66 720   | 960, 720      |
| 705, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:
```python
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
```

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in `measure_curvature_pixels(ploty,left_fit, right_fit, ym_per_pix=30/720, xm_per_pix=3.7/700)`  in my code in

the previous fit passed to this function is done on the pixel space and the coeficient is respective to pixel space. To convert back to the real world space. I used the following knowledge from a insightful student in the lecture. 
 > if the parabola is x= a*(y**2) +b*y+c; and mx and my are the scale for the x and y axis, respectively (in meters/pixel); then the scaled parabola is x= mx / (my ** 2) *a*(y**2)+(mx/my)*b*y+c


```python
    left_a_new = xm_per_pix/(ym_per_pix**2)*left_fit[0]
    left_b_new = xm_per_pix/(ym_per_pix)*left_fit[1]
    right_a_new = xm_per_pix/(ym_per_pix**2)*right_fit[0]
    right_b_new = xm_per_pix/(ym_per_pix)*right_fit[1]
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_a_new*y_eval + left_b_new)**2)**1.5) / np.absolute(2*left_a_new)
    right_curverad = ((1 + (2*right_a_new*y_eval + right_b_new)**2)**1.5) / np.absolute(2*right_a_new)
```


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function `draw_lane_fit`. Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

Pitfalls:
* Cannot recognize the lanes in some special cases in project videos and especially in challeng videos:  
![alt text][image7]

possible solutions are: tune the image filter parameter, instead in RGB space but HLS space to elimanate the influence from sunshine, etc. 
To do the averaging over the time series or space, like reject certain recognized line if the new line is a big position change repect to the last line.

 
