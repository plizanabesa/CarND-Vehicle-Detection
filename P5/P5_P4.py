import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from IPython import get_ipython
ipython = get_ipython()
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label

########### P4 ##################

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None

def find_chessboard_corners(nx, ny):
    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    calibration_images = []

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = mpimg.imread(fname)
        calibration_images.append(img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            #cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            #write_name = 'corners_found'+str(idx)+'.jpg'
            #cv2.imwrite(write_name, img)
            #plt.imshow(img)
            #cv2.imshow('img',img)
            #cv2.waitKey(500)

    cv2.destroyAllWindows()
    return calibration_images,objpoints, imgpoints

def calibrate_camera(img, objpoints, imgpoints):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

    return mtx, dist

def undistort_image(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # TODO: plot undistorted image
    #cv2.imshow('undist',undist)
    return undist

def warper(img, undist_img, src, dst):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(undist_img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    #warped = cv2.warpPerspective(undist_img, M, img_size, flags=cv2.INTER_LINEAR)

    return warped, M, Minv

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):    
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    if orient!='x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel,dtype=np.float)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1.0
    return binary_output

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
    mag_sobel = np.sqrt(np.square(sobelx)+np.square(sobely))
    scaled_sobel = np.uint8(255*mag_sobel/np.max(mag_sobel))
    binary_output = np.zeros_like(scaled_sobel,dtype=np.float)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1.0
    return binary_output

def dir_thresh(img, sobel_kernel=3, dir_thresh=(0, np.pi/2)):  
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel)
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    dir_sobel = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(dir_sobel)
    binary_output[(dir_sobel >= dir_thresh[0]) & (dir_sobel <= dir_thresh[1])] = 1.0
    return binary_output

def color_thresh(img, thresh=(90, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    binary_output = np.zeros_like(S, dtype=np.float)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1.0
    return binary_output

def gray_thresh(img, thresh=(180, 255)):
    gray_thresh = (180, 255)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    binary_output = np.zeros_like(gray)
    binary_output[(gray > gray_thresh[0]) & (gray <= gray_thresh[1])] = 1
    return binary_output

def get_binary_image(img):
    ksize = 3 
    ksize2 = 9
    ksize3 = 15

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(img, orient='x', sobel_kernel=ksize2, thresh=(10, 120))
    grady = abs_sobel_thresh(img, orient='y', sobel_kernel=ksize2, thresh=(40, 80))
    mag_binary = mag_thresh(img, sobel_kernel=ksize2, mag_thresh=(30, 100))
    dir_binary = dir_thresh(img, sobel_kernel=ksize3, dir_thresh=(0.7, 1.3))
    color_binary = color_thresh(img,  thresh=(100, 255))
    #color_binary = color_thresh(img,  thresh=(80, 255))  #Harder Challenge
    gray_binary = gray_thresh(img,  thresh=(130, 255))   #Project & Challenge 1 
    #gray_binary = gray_thresh(img,  thresh=(230, 255))   #Harder Challenge 1
    
    # Get the combined binary image
    combined = np.zeros_like(dir_binary)
    #combined[((gradx == 1)  & (dir_binary == 1) & ((color_binary == 1))| (gray_binary == 1))] = 1.0   #Challenge
    combined[((gradx == 1)  & (dir_binary == 1)) | ((color_binary == 1) & (gray_binary == 1))] = 1.0   #Project and Harder Challenge

    """
    combined_stack_1 = np.dstack(( np.zeros_like(gradx), color_binary, gradx))
    combined_stack_2 = np.dstack(( np.zeros_like(gradx), color_binary, gray_binary))
    combined_stack_3 = np.dstack(( np.zeros_like(gradx), mag_binary, dir_binary))

    mpimg.imsave('stack_color_gradx',combined_stack_1)
    mpimg.imsave('stack_color_gray',combined_stack_2)
    mpimg.imsave('stack_mag_dir',combined_stack_3)
    mpimg.imsave('stack_mag_dir',combined_stack_3)
    mpimg.imsave('stack_mag_dir',combined_stack_3)
    """
    """
    combined_stack_1 = np.dstack(( np.zeros_like(gradx), np.zeros_like(gradx), gradx))
    combined_stack_2 = np.dstack(( np.zeros_like(gradx), np.zeros_like(gradx), grady))
    combined_stack_3 = np.dstack(( np.zeros_like(gradx), np.zeros_like(gradx), mag_binary))
    combined_stack_4 = np.dstack(( np.zeros_like(gradx), np.zeros_like(gradx), gray_binary))
    combined_stack_5 = np.dstack(( np.zeros_like(gradx), np.zeros_like(gradx), dir_binary))
    combined_stack_6 = np.dstack(( np.zeros_like(gradx), np.zeros_like(gradx), color_binary))

    mpimg.imsave('stack_gradx',combined_stack_1)
    mpimg.imsave('stack_grady',combined_stack_2)
    mpimg.imsave('stack_mag',combined_stack_3)
    mpimg.imsave('stack_gray',combined_stack_4)
    mpimg.imsave('stack_dir',combined_stack_5)
    mpimg.imsave('stack_color',combined_stack_6)
    """

    """
    f, ((ax1, ax2, ax3),(ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20,10))

    ax1.set_title('Original')
    ax1.imshow(img)

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]
    S_img = np.dstack(( np.zeros_like(gradx), np.zeros_like(gradx), S))
    ax2.set_title('S color channel')
    ax2.imshow(S_img)

    ax3.set_title('Combined S channel, gradient and direction thresholds')
    ax3.imshow(combined, cmap='gray')

    ax4.set_title('Grad color (green) and x (blue) thresh')
    ax4.imshow(combined_stack_1)

    ax5.set_title('Grad color (green) and gray (blue) thresh')
    ax5.imshow(combined_stack_2)

    ax6.set_title('Grad magnitude (green) and direction (blue) thresh')
    ax6.imshow(combined_stack_3)
    """

    return combined

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image

def get_lines_from_scratch(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    #nwindows = 9   #Project and Challenge 1
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    #margin = 50   #Harder challenge
    # Set minimum number of pixels found to recenter window
    minpix = 50
    #minpix = 30   #Harder challenge
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    if len(leftx)==0 or len(lefty)==0 or len(rightx)==0 or len(righty)==0:
        return

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    out_img = np.uint8(out_img)
  
    """
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.savefig('sliding_windows.png')
    """
    
    update_lines(binary_warped, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty)

def get_lines_with_previous_fit(binary_warped):
    # Get previous lines
    left_fit = left_line.best_fit
    right_fit = right_line.best_fit
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    if len(leftx)==0 or len(lefty)==0 or len(rightx)==0 or len(righty)==0:
        return
        
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    """
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.savefig('sliding_windows.png')
    """

    update_lines(binary_warped, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty)

def update_lines(binary_warped, left_fit, right_fit, left_fitx, right_fitx, leftx, lefty, rightx, righty):
    ### Get curvature and offset
    left_curverad, right_curverad, offset, left_dist, right_dist = get_curvature_and_offset(binary_warped, left_fit, right_fit)

    ### Validate lines
    valid_lines = are_lines_valid(binary_warped, left_fit, right_fit, left_curverad, right_curverad)

    #### Update line parameters
    left_line.recent_xfitted.insert(0, left_fitx)
    if left_line.current_fit!=None:
        left_line.diffs = [left_fit[0]-left_line.current_fit[0],left_fit[1]-left_line.current_fit[1],left_fit[2]-left_line.current_fit[2]]
    left_line.current_fit = left_fit
    left_line.allx = leftx
    left_line.ally = lefty  

    right_line.recent_xfitted.insert(0, right_fitx)
    if right_line.current_fit!=None:
        right_line.diffs = [right_fit[0]-right_line.current_fit[0],right_fit[1]-right_line.current_fit[1],right_fit[2]-right_line.current_fit[2]]
    right_line.current_fit = right_fit
    right_line.allx = rightx
    right_line.ally = righty       
    
    global frames_wo_valid_lines
    global find_lines_from_scratch
    frames_wo_valid_lines = frames_wo_valid_lines + 1
    if valid_lines:
        left_line.detected = True
        right_line.detected = True
        find_lines_from_scratch=False
        frames_wo_valid_lines = 0
    else:
        left_line.detected = False
        right_line.detected = False
        find_lines_from_scratch=True

    # If lines are valid, then update de following attributes in the line   
    if valid_lines or frame_number==1:
        global vehicle_offset
        vehicle_offset = offset
        left_line.radius_of_curvature = left_curverad
        left_line.line_base_pos = left_dist
        # Smooth polynomial fit using exponential smoothing
        if left_line.best_fit==None:
            left_line.best_fit = left_fit
        else:
            left_line.best_fit = left_line.best_fit*(1-exponential_smooth_lambda)+left_fit*exponential_smooth_lambda
        # Smooth x values using exponential smoothing
        if left_line.bestx==None:
            left_line.bestx = left_fitx
        else:
            left_line.bestx = left_line.bestx*(1-exponential_smooth_lambda)+left_fitx*exponential_smooth_lambda

        right_line.radius_of_curvature = right_curverad
        right_line.line_base_pos = right_dist
        # Smooth polynomial fit using exponential smoothing
        if right_line.best_fit==None:
            right_line.best_fit = right_fit
        else:
            right_line.best_fit = right_line.best_fit*(1-exponential_smooth_lambda)+right_fit*exponential_smooth_lambda
        # Smooth x values using exponential smoothing
        if right_line.bestx==None:
            right_line.bestx = right_fitx
        else:
            right_line.bestx = right_line.bestx*(1-exponential_smooth_lambda)+right_fitx*exponential_smooth_lambda

def get_pixel_curvature(binary_warped, left_fit, right_fit):
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])

    return left_curverad, right_curverad

def get_curvature_and_offset(binary_warped, left_fit, right_fit):
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculate the new radius of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Get offset from line center
    bottom_left_fitx = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    bottom_right_fitx = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    lane_center = (bottom_left_fitx + bottom_right_fitx)/2
    pixel_offset = lane_center - binary_warped.shape[1]/2
    offset = round(pixel_offset * xm_per_pix,2)
    center_left_line_dist = abs(bottom_left_fitx-lane_center) * xm_per_pix
    center_right_line_dist = abs(bottom_right_fitx-lane_center) * xm_per_pix

    return left_curverad, right_curverad, offset, center_left_line_dist, center_right_line_dist

def are_lines_valid(binary_warped, left_fit, right_fit, left_curverad, right_curverad):
    # Validate lines using (i) curvature difference, (ii) lane size at the bottom and top of the warped image 
    #and (iii) the lane size both at the bottom and top of the image are similar to the last valid lane size estimation
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] ) 
    y_bottom = np.max(ploty)
    y_top = np.min(ploty)
    bottom_left_fitx = left_fit[0]*y_bottom**2 + left_fit[1]*y_bottom + left_fit[2]
    bottom_right_fitx = right_fit[0]*y_bottom**2 + right_fit[1]*y_bottom + right_fit[2]
    top_left_fitx = left_fit[0]*y_top**2 + left_fit[1]*y_top + left_fit[2]
    top_right_fitx = right_fit[0]*y_top**2 + right_fit[1]*y_top + right_fit[2]
    lane_size_bottom = (bottom_right_fitx-bottom_left_fitx)*xm_per_pix
    lane_size_top = (top_right_fitx-top_left_fitx)*xm_per_pix    
    delta_curverad_percentage = np.absolute((right_curverad - left_curverad)/right_curverad)
    delta_curverad = np.absolute((right_curverad - left_curverad))
    print(delta_curverad, delta_curverad_percentage, lane_size_bottom, lane_size_top)

    # Use bigger threshold when curvatures are bigger (i.e lines are straight)
    
    # Harder challenge
    """
    delta_curverad_percentage_threshold = 0.9
    delta_curverad_threshold = 1200
    if left_curverad>3000 and right_curverad>3000:
        delta_curverad_percentage_threshold = 1.2
    """
 
    # Challenge
    """
    delta_curverad_percentage_threshold = 0.8
    delta_curverad_threshold = 1000
    if left_curverad>3000 and right_curverad>3000:
        delta_curverad_percentage_threshold = 0.9
    """

    # Project
    
    delta_curverad_percentage_threshold = 0.4
    delta_curverad_threshold = 500
    if left_curverad>3000 and right_curverad>3000:
        delta_curverad_percentage_threshold = 0.5
    

    global lane_size
    #if abs(lane_size_bottom-lane_size_top)<1.4 and ((delta_curverad < delta_curverad_threshold) | (delta_curverad_percentage < delta_curverad_percentage_threshold)) and abs(lane_size_bottom-lane_size)<1.4 and abs(lane_size_top-lane_size)<1.4:
    #if abs(lane_size_bottom-lane_size_top)<0.7 and ((delta_curverad < delta_curverad_threshold) | (delta_curverad_percentage < delta_curverad_percentage_threshold)) and abs(lane_size_bottom-lane_size)<0.7 and abs(lane_size_top-lane_size)<0.7:
    if abs(lane_size_bottom-lane_size_top)<0.5 and ((delta_curverad < delta_curverad_threshold) | (delta_curverad_percentage < delta_curverad_percentage_threshold)) and abs(lane_size_bottom-lane_size)<0.6 and abs(lane_size_top-lane_size)<0.6:
        lane_size = (lane_size_bottom+lane_size_top)/2
        global bottom_lane_size
        bottom_lane_size = lane_size_bottom
        global top_lane_size
        top_lane_size = lane_size_top
        return True
    else:
        return False

def draw_lines_on_original_image(undist, Minv, binary_warped, left_fit, right_fit):
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.fillPoly(color_warp, np.int_([pts_left]), (255,0, 0))
    #cv2.polylines(color_warp,np.int32([pts_left]), True, (255,0,0), 3) 
    cv2.fillPoly(color_warp, np.int_([pts_right]), (0,0, 255))
    #cv2.polylines(color_warp,np.int32([pts_right]), True, (0,0,255), 3) 

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (undist.shape[1], undist.shape[0])) 

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    #plt.imshow(result)

    return result

def process_frame_lanes(img):    

    ### Undistort image
    undist = undistort_image(img, mtx, dist)
    
    ### Get binary image combining magnitude, direction and color gradient thresholds
    binary = get_binary_image(undist)
    #plt.imshow(binary, cmap='gray')
    #mpimg.imsave('binary',binary, cmap='gray')
    

    ### Harder challenge: Apply region mask to black out pixels outside of the region
    #vertices = np.array([[(150,720),(300, 550), (1050, 550), (1200,720)]], dtype=np.int32)
    #binary = region_of_interest(binary,vertices)
    #plt.imshow(binary, cmap='gray')
    #mpimg.imsave('binary_masked',binary, cmap='gray')
    

    #### Define source and destination points to warp image 
    #pts_offset = 100 # offset for dst points
    #img_size = (img.shape[1], img.shape[0])
    src = np.float32([[580, 460], [198, 720], [1127, 720], [705, 460]])  #Project 
    #src = np.float32([[580, 500], [198, 720], [1127, 720], [800, 500]])  #Challenge 1
    #src = np.float32([[280, 550], [153, 720], [1127, 720], [1000, 550]])  #Challenge 2
    dst = np.float32([[320, 0], [320, 720], [960,720], [960, 0]])  
    #dst = np.float32([[pts_offset, pts_offset], [img_size[0]-pts_offset, pts_offset],[img_size[0]-pts_offset, img_size[1]-pts_offset],[pts_offset, img_size[1]-pts_offset]])

    ### Warp image to from above perspective
    binary_warped, M, Minv = warper(binary, binary, src, dst)
    
    """
    img_warped, M, Minv = warper(img, undist, src, dst)
    plt.imshow(img_warped)
    mpimg.imsave('img_warped',img_warped) 
    
    plt.imshow(binary_warped, cmap='gray')
    mpimg.imsave('binary_warped',binary_warped)
    
    ### Draw points in original and warped images
    img_src = img
    cv2.polylines(img_src,np.int32([src]), True, (255,0,0), 2) 
    mpimg.imsave('img_src',img_src)

    img_warped_dst = img_warped
    cv2.polylines(img_warped_dst,np.int32([dst]), True, (255,0,0), 2) 
    mpimg.imsave('img_warped_dst',img_warped_dst)

    binary_src = np.dstack((binary, binary, binary))*255
    cv2.polylines(binary_src,np.int32([src]), True, (255,0,0), 2) 
    binary_src = np.uint8(binary_src)
    mpimg.imsave('binary_src',binary_src)

    binary_warped_dst = np.dstack((binary_warped, binary_warped, binary_warped))*255
    cv2.polylines(binary_warped_dst,np.int32([dst]), True, (255,0,0), 2) 
    binary_warped_dst = np.uint8(binary_warped_dst)
    mpimg.imsave('binary_warped_dst',binary_warped_dst)
    """
      
    ### Find lines
    global frames_wo_valid_lines
    if find_lines_from_scratch:
        # If frames without valid lines exceeds threshold, then use again 
        if frames_wo_valid_lines > frames_wo_valid_lines_threshold:
            # Find line lanes from scratch using sliding windows
            print("find lines from scratch")
            get_lines_from_scratch(binary_warped)
        else:
            print("keeping lines from previous iterations")
            frames_wo_valid_lines = frames_wo_valid_lines+1
    else:
        # Find lines using previous fit
        print("previous fits used")
        get_lines_with_previous_fit(binary_warped)

    ### Draw lines in original image
    result = draw_lines_on_original_image(undist, Minv, binary_warped, left_line.best_fit, right_line.best_fit)

    ### Write text on frame
    text1 = "Vehicle is {}m right of center".format(vehicle_offset)
    if vehicle_offset<0:
        text1 = "Vehicle is {}m left of center".format(-vehicle_offset)

    text2 = "Radius of curvature: {}(m)".format(round((left_line.radius_of_curvature+right_line.radius_of_curvature)/2,0))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result, text1, (200, 100), font, 2, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(result, text2, (200, 200), font, 2, (255,255,255), 2, cv2.LINE_AA) 
    
      
    global frame_number
    print ("Frame: {}".format(frame_number))
    print ("Left Line Detected: {}".format(left_line.detected))
    print ("Left Line Radius of Curvature: {} m".format(left_line.radius_of_curvature))
    print ("Left Line Coefficients: {}".format(left_line.best_fit))
    print ("Right Line Detected: {}".format(right_line.detected))
    print ("Right Line Radius of Curvature: {} m".format(right_line.radius_of_curvature))
    print ("Right Line Coefficients: {}".format(right_line.best_fit))
    print ("Vehicle offset: {} m".format(vehicle_offset))
    print ("Lane size: {} m".format(lane_size))
    print ("Bottom lane size: {} m".format(bottom_lane_size))
    print ("Top lane size: {} m".format(top_lane_size))
    print ("Frames without lines: {}".format(frames_wo_valid_lines))
    frame_number = frame_number+1

    #mpimg.imsave('result',result)
    return result


########### P5 ##################

def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)


# Function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features


# Function to compute color histogram features  
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Function to extract color and hog features from a list of images
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


# Function to extract features from a single image window
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)


# Function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                                  visualise=True, feature_vector=False)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=False, 
                       visualise=False, feature_vector=feature_vec)
        return features


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


# Function you will pass an image and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows


# Function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    draw_img = np.copy(img)
    window_list = []
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell)-1
    nyblocks = (ch1.shape[0] // pix_per_cell)-1 
    nfeat_per_block = orient*cell_per_block**2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell)-1 
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                window_list.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
                cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,0,255),6) 
                
    #return draw_img
    return window_list


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 6)
    # Return the image
    return img

############ Combined P4 y P5 #############

def process_frame(image):

    ### Detect lane lines
    lane_image = np.copy(image)
    lane_image = process_frame_lanes(lane_image)

    ### Run sliding windows to detect cars using the classifier
    #jpg_image = np.copy(image)
    image = image.astype(np.float32)/255

    """
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None], xy_window=(128, 128), xy_overlap=(0.5, 0.5))
                           
    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, spatial_size=spatial_size, hist_bins=hist_bins, orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel, spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat)                       

    window_img = draw_boxes(image, hot_windows, color=(0, 0, 255), thick=6)   
    """

    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Get windows with multiple scale search
    windows_list=[]
    for scale in scales:
    	scale_windows_list = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
    	windows_list.extend(scale_windows_list)

    # Combine windows from previous frames to remove false positives and smooth classifier output
    previous_windows_list.append(windows_list)
    if(len(previous_windows_list) > 10):
        previous_windows_list.pop(0)

    new_windows_list=[]
    for w_list in previous_windows_list:
        new_windows_list.extend(w_list)

    # Add heat maps to windows
    heat = add_heat(heat,new_windows_list)
        
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,8)

    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)

    # Draw labels
    result = draw_labeled_bboxes(lane_image, labels)

    """
    plt.imshow(result)
    mpimg.imsave('result', result)
    """

    return result


if __name__ == "__main__":

    plt.ion()

    ### Define global variables: left and right lane lines
    global left_line 
    left_line = Line()
    global right_line 
    right_line = Line()
    global find_lines_from_scratch
    find_lines_from_scratch = True
    global vehicle_offset 
    vehicle_offset = 0
    global exponential_smooth_lambda 
    #exponential_smooth_lambda = 0.7
    exponential_smooth_lambda = 0.3
    global frames_wo_valid_lines 
    frames_wo_valid_lines = 3
    global frames_wo_valid_lines_threshold 
    frames_wo_valid_lines_threshold = 2
    global frame_number
    frame_number = 1
    global lane_size
    lane_size = 3.0  #Project and challenge
    #lane_size = 2.7 #Harder challenge
    global ym_per_pix
    ym_per_pix = 30/720
    global xm_per_pix
    xm_per_pix = 3.7/700
    global bottom_lane_size
    bottom_lane_size = 3.0
    global top_lane_size
    top_lane_size = 3.0
    global mtx
    global dist

    ### Find and plot cheeseboard corners
    nx = 9
    ny = 6
    cal_images, objpoints, imgpoints = find_chessboard_corners(nx, ny)

    ### Calibrate camera
    mtx, dist = calibrate_camera(cal_images[0], objpoints, imgpoints)

    ### Read training data
    car_path_1 = 'training_data/vehicles/GTI_Far'
    car_path_2 = 'training_data/vehicles/GTI_Left'
    car_path_3 = 'training_data/vehicles/GTI_MiddleClose'
    car_path_4 = 'training_data/vehicles/GTI_Right'
    car_path_5 = 'training_data/vehicles/KITTI_extracted'
    car_paths = [car_path_1, car_path_2, car_path_3, car_path_4, car_path_5]
    not_car_path_1 = 'training_data/non-vehicles/Extras'
    not_car_path_2 = 'training_data/non-vehicles/GTI'
    not_car_paths = [not_car_path_1, not_car_path_2]
    cars = []
    notcars = []
    for path in car_paths:
        car_images = glob.glob(os.path.join(path, '*.png'))
        for image in car_images:
                cars.append(image)
    for path in not_car_paths:
        not_car_images = glob.glob(os.path.join(path, '*.png'))
        for image in not_car_images:
                notcars.append(image)

    print(len(cars))
    print(len(notcars))

    ### Extract color and HOG features for training car detection classifier
    print('Extracting features for training')
    global color_space
    global orient
    global pix_per_cell
    global cell_per_block
    global hog_channel
    global spatial_size
    global hist_bins
    global spatial_feat
    global hist_feat
    global hog_feat
    global y_start_stop
    global ystart
    global ystop
    global scales
    global svc
    global previous_windows_list

    color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 16    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    y_start_stop = [None, None] # Min and max in y to search in slide_window()
    ystart = 380
    ystop = 680
    scales = [1.0, 1.5, 2.0]  # Scale parameter for multiple-scaled search windows
    previous_windows_list = []

    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

    ### Scale features

    # Create an array stack of feature vectors
    print('Scaling features for training')
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    """
    # Plot an example of raw, scaled and hog features
    car_ind = np.random.randint(0, len(cars))
    fig = plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.imshow(mpimg.imread(cars[car_ind]))
    plt.title('Original Image')
    plt.subplot(132)
    plt.plot(X[car_ind])
    plt.title('Raw Features')
    plt.subplot(133)
    plt.plot(scaled_X[car_ind])
    plt.title('Normalized Features')
    fig.tight_layout()
    """

    ### Train car detection classifier

    # Define the labels vector
    print('Training classifier')
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using spatial binning of:',spatial_size,
    'and', hist_bins,'histogram bins')
    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    ### Process test frame images
    
    """
    print('Processing test image')
    img_path = 'test_images/test3.jpg'
    image = mpimg.imread(img_path)
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    process_frame(image)
    """

    ### Process test videos
    

    print('Processing video')
    output = 'output.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    output_clip = clip1.fl_image( process_frame )
    output_clip.write_videofile(output, audio=False)
    
    

    #input()

    plt.ioff()

