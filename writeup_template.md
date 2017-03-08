##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/not_car.png
[image3]: ./output_images/HOG.jpg
[image4]: ./output_images/windows.jpg
[image5]: ./output_images/result_1.jpg
[image6]: ./output_images/result_3.jpg
[video1]: ./output_1.mp4
[video2]: ./output_2.mp4
[video3]: ./output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

HOG features were extracted using the built in hog function `skimage.hog()` (lines 149 to 159 in `P5.py`). This was made for all `vehicle` (GTI and KITTI images) and `non-vehicle` (GTI and Extras) images - resulting in a 50%-50% car-not car ratio. Examples of car and not car images as follow:

![alt text][image1]
![alt text][image2]

Afterwards I tested with different hog function parameters. My final hog parameters where: 9 `orientations`, 8 `pix_per_cell`, 2 `cells_per_block`
and 'ALL' hog_channels. The final color space was `YCrCb`. Here an example of a hog features image:

![alt text][image3]

I settled in this parameters because they gave me enough features to have a high accuracy classifier. Also, using 32x32 spatial features instead of 16x16 made the training super slow, and my computer run out of memory.

####2. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using color features (histogram with 16 bins per color channel and spatial features with 16x16 pixel sizes) and hog features with the paramteres mentioned above. The accuracy of my classifier in the test set was in the range of 98-99%. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Initially, I tried with six scales (0.5, 0.8, 1, 1.5, 2), but made the search very slow, so I ended up using three scales (1, 1.5, 2).

![alt text][image4]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched for cars iterating in three scales using YCrCb 3-channel hog features, plus spatially binned color (16x16) and histograms (16 bin), which gave a feature vector of size 6108. Here are some results from my window search:

![alt text][image5]
![alt text][image6]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here are links [to one video result](./output_1.mp4), [other video result](./output_2.mp4), [and video result combining with my lane line detector](./output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In each frame of the video I saved the positive detections. From this detections I created a thresholded heatmap, later combined with the `scipy.ndimage.measurements.label()` (line 390 in `P5.py`) to find individual cars in the image. To smooth out the cars detected, for the heatmap I used the windows found in the previous 10 frames and a threshold of 8 counts per pixel. By tuning the threshold value was possible to remove outliers as much as possible. I even tried with 15 frames and a threshold of 15 counts, but the search was too slow and the gain not much.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

A problem that was foreseeable with the heatmap was when two cars are in the same line-of-view of the camera. In this scenario, the heatmap merges both cars and hence the algorithm looses its ability to track individual cars. My pipeline also failed to recognize the far away cars. A reason for this could be the low spatial bining (16x16) that used by the classifier.

More work and fine tuning could be done in order to make car windows transition more smoothly between adjacent frames and also to remove more outliers.

