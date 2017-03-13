**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use trained classifier to search for vehicles in images.
* Run pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/hog.png
[image3]: ./output_images/window_scales.png
[image4]: ./output_images/search_windows.png
[image5]: ./output_images/windows_heatmap.png
[video1]: ./project_video_out.mp4

###Histogram of Oriented Gradients (HOG)

####1. Extracted HOG features from the training images.

The code for this step is contained in function `show_data_sample` in line 78 through 93 of the file called `classifier.py`.  I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

####2. Final choice of HOG parameters.

I tried various combinations of parameters and found `orient=9`, `pix_per_cell=8`, and `cell_per_block=2` gives the best shape description and clearly distinguishes car from not cars. all features parameter are store in a dictionary called `settings` in `classifier`

####3. Training SVM classifier.

I trained a linear SVM using LinearSVC(), the code can be found in function `train_model()` of `classifier.py` The features used for training are spatial, color histogram, and hog features. Features are scaled to zero mean and unit variance using `StandardScaler()`. Three channel of YCrCb are all used for Hog as I found it gives me the best training accuracy. The accuracy I obtained is around 99.1% This accuracy is high because training data are sequence of video stream (most of the time are the same), which could be leak into test dataset even after random split. It is possible to manually divide the data, however, for this project this is enough to get a good estimation of classifier performance.
The model and the scaler is then saved in a pickle file for later use. This can be foudn in `save_model()` of `classifier.py`

###Sliding Window Search

####1. Multi-scale sliding window search

The code for sliding window search can be found in `find_multiscale()` function in `detector.py`. I actually started with single scale factor 1.0 and use the function `search_windows`. It pre-generates the sliding windows and compute features for each window and make prediction. It gives me very good result on test_images with little false positive. However, when I try on video stream with frame where cars are far away, it fails to detect well. Then I moved onto another approach learnt in the lesson `find_car`, which optimizes the performance by getting Hog feature for the entire ROI only once for a given frame and retrieve feature for each slide window from the feature vector. It also allows me to call the function multiple times to combine the search result at different scale.

The image below shows the multi-scale window I choose, the scaling factors can be found in `self.ypos_scales` in `detector`. The overlaps between ROI for each scale could be fine tuned, so it only search with larger scale for lower portion and smaller scale for upper portion. However, the reason I choose 5 different scales is I want more overlaps of detected car windows, so that it increase the confidence of detection result. Even though it might increase the chance of showing false positive, but further filtering algorithm approach can be used later to eliminated. It is possible to use less scale but from experiment my choice show good detection result.  

![alt text][image3]

This image below draws all windows and its effective region in different colours.

![alt text][image4]

####2. Example of test image result.

Ultimately I searched on five scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are my result on the test images. It shows original images, heatmap and final boundingbox.  

![alt text][image5]
---

### Video Implementation

####1. Video Link

[link to my video result](./project_video_out.mp4)

####2. Detection and filtering on video stream

The main difference between detection on images is in video false positive are very unlikely to be found on the a few subsequent frames, which is the key information for filtering. My video pipeline works as follow which can be found in `detect()` function of `detector.py`:

1. For each frame of the video. First, I obtain the windows search result from `find_multiscale`.they are just multiple bounding boxes including false positive.

2. From the positive detections I created a heatmap using `utils.get_heatmap(image, out_windows)` from `utils.py`

3. Then the constructed heatmap is pass to a filter called `smoother` defined in  line 23 of `detector.py` which is a simple moving average filter. This is implemented in `moving_average()` function in `utils.py`. filter's window/period is defined as 10, it averages the heatmap for 10 consecutive frames. This filter also reduces the bounding box wobbling effect as well as filtering out false positive when used with threshold later.

4. The output of the filtered heatmap is then thresholded using `apply_threshold()` in `utils.py`, which effectively reduces false positive because highly confident detection is only found near the same position in several subsequent frames(assuming the object is not travelling at the speed of light. :) just kidding.)

5. The heatmap from step 4 is then pass onto `utils.get_labeled_boxes()` which converts a heatmap to labels, and then
bounding rectangle. Finally it is drawn to the original image using `utils.draw_boxes()`.

###Discussion

The algorithm overall performs well on the project video clip. The problem I encountered during the project is dealing with false positive, and complexity of the find_car algorithm. It could be improved by reducing scale options. The exact execution time of the function should be measured in the future. Another improvement could be made is to combine adjacent bounding boxes where in certain frame I found the classifier only detects small portions of the car even when it is quite near, so the final bounding box is broken down to two pieces. This could lead to inaccurate detection of the number of cars. The algorithm does not distinguish between two cars if they are very close together, instead it will draw a combined bounding box, which is not a big issue as it does not cause dangerous decision like emergency brake. However, this could be improved by recording/learning the specific feature of a car once it can be considered as a high confident detection, so that it can only search the car with specific feature like TLD (tracking learning detection algorithm).
