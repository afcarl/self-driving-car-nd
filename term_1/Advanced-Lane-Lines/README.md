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

[image1]: ./output_images/undistorted_1.png "Undistorted"
[image2]: ./output_images/undistorted_2.png "Undistorted"
[image3]: ./output_images/undistorted_3.png "Undistorted"
[image4]: ./output_images/src_dst.png "Perspevtive Points"
[image5]: ./output_images/yellow_white.png "Binary Example"
[image6]: ./output_images/gradient_saturation.png "Binary Example"
[image7]: ./output_images/gradient_color.png "Binary Example"
[image8]: ./output_images/lane_detection.png "Warp Example"
[image9]: ./output_images/final.png "Fit Visual"
[video1]: ./project_video_out.mp4 "Video"
[video2]: ./challenge_video_out.mp4 "Video"

###Camera Calibration

The code for this step is contained in `calibration.py`). I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1] ![alt text][image2]

The calibration process is only done once, the calibration data is then saved in a pickle file which is read every time the program is run.

###Pipeline (single images)

####1. Image distortion correction.

Here is a example of distortion-corrected image using one of the test_images. The code can be found in the `pipeline` function  in `run.py`

![alt text][image3]

####2. Perspective transform

The code for my perspective transform includes a function called `warp()`, which appears in lines 35 through 41 in the file `processing.py`  The `warp()` function uses `get_perspective_points()` internally to get source and destination points. Source and destination points are calculated using the following as a guide and manually fine tuned so that only the target lane region are displayed and centered in the warped images.
```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. The points are shown in the below image where red rectangle is `dst` and blue is `src`.

![alt text][image4]

####3. Image processing
All of the image processing algorithm can be found in `processing.py`. I use combination of different techniques to produce binary image. They are yellow_white, gradient_saturation, and gradient_color. yellow_white uses HSV channel to detect yellow line and L channel of HLS to detect white line. gradient_saturation uses sobel gradient to detectvertical line and the result is combined with threshoded s channel of HLS. gradient_color is special because it detect pixels only if both yellow_white and gradient threshold condition are met as this can be an important criteria for something to be considered as a lane line in real life. Here's an example of my output for this step.

![alt text][image5] ![alt text][image6] ![alt text][image7]

####4. Lane detection and polynomials fit

The detection code can be found in `detector.py`. The main detection algorithm driver is called `detect_lane` It basically iterate through different image algorithm to get the best possible outcome in different conditions, i.e. sometimes the white line can not be easily distinguishable from surrounding road color or lane looks split with different color other than yellow and white. So it acts like a complementary filter to utilise the strength of different algorithm in different situations.

In detail, it breaks down to a few steps:
1. Locate the base positions of left and right line by finding two peak values in binary_warped histogram.
2. Use sliding windows search to return a line object
3. Check if found lines are good candidates for real lane. (i.e. lane width, parallel lines by checking angles in hough space. etc.)
4. Update the lane line member with detected result if detection is considered good. This is effectively a filter which avoids jittering as we only draw the lane using the most recent good detection. This is how human makes decision anyway, when we drive, our eyes could focus away from the road for a second but our steering will not change simply because road curvature won't change shapely in most of the time.

The sliding window approach can be foudn in `find_line()`. If detection was robustly found, rather than searching the entire next frame for the lines, just a window around the previous detection could be searched. Once the lines pixels position (x,y) are found, a polynomials is fit in `get_line()` function.

![alt text][image8]

####5. Curvature calculation and vehicle position.

Curvature calculation code is from lines 169 through 182 in `calculate_curvature()` in `detector.py`, while Vehicle position code can be found in `calculate_car_position()`

####6. A final image with lane detection.

I implemented this step in lines # through # in my code in `detector.py` in the function `draw_final_lane()`.  Here is an example of my result on a test image:

![alt text][image9]

---

###Pipeline (video)

Here's a link for [project_video](./project_video_out.mp4)

Here's a link for [challenge_video](./challenge_video_out.mp4)

---

###Discussion

There are a few issues I encountered during the project. One is to reliably estimate the lane width as this is important parameter to determine if a detected lines are good. I did this by averaging the distance between the first 5 line fit I found. This might not be robust enough as We don't even know if the first 5 fits are good. Another one is the image processing algorithm. It is hard to design a generic algorithm which works for all kinds of road conditions (lighting condition, lane line partially invisible or hard to be distinguished, dirts on the road, and split lane, etc.) For example gradient and saturation might work well for project video but function poorly in challenge video. So I design a algorithm which combine the strength of all image processing technique I can come up with to solve this. If the detector fails to find lines or found line does not meet criteria, it will try another algorithm to see if passes and saves the result. From the output, it can be observed that this approach effectively improves the robustness.

However, my algorithm will fail when none of the three image algorithm can reliably gives me good result to search for lines, which results in parallel condition failure and being considered as a bad candidate. My algorithm can be improved by returning the best output(best fits the lane criteria) of three algorithm detection rather than returning the first fit.

I have not tested on harder_challenge video, I know it will certainly fails because there are sharp turns and extreme lighting. To make it more robust, further work will be making prediction based on current lines data(this will help when camera has complete lost vision in extreme lighting. It would be easier if camera can wear a sun glasses :>). Maybe some learning algorithm can be deployed to help us distinguish lane from other object.
