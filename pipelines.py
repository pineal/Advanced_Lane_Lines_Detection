from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import pickle
import glob

# Load calibration pickles
dist_pickle = pickle.load (open("./camera_cal/calibration_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

def abs_sobel_thresh(img, orient='x', thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

def color_threshold(img, sthresh=(0,255), vthresh=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel >= vthresh[0]) & (v_channel <= vthresh[1])] = 1
    
    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary == 1)] = 1
    return output

def window_mask(width, height, img_ref, center,level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
    return output

def preprocess_image(img):

    result = np.zeros_like(img[:,:,0])
    
    gradx = abs_sobel_thresh(img, orient='x', thresh=(12, 255))
    grady = abs_sobel_thresh(img, orient='y', thresh=(25, 255))

    combined_binary = color_threshold(img, sthresh=(100,255), vthresh=(50,255))
    result[((gradx == 1) & (grady == 1) | (combined_binary == 1))] = 255
    return result

def find_window_centroids(image):
 
    window_centroids = [] # Store the (left,right) window centroid positions per level
    window = np.ones(window_width) # Create our window template that we will use for convolutions   
    
    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template 
    
    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
    r_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+int(image.shape[1]/2)
    
    # Add what we found for the first layer
    window_centroids.append((l_center,r_center))  
    
    # Go through each layer looking for max pixel locations
    for level in range(1,(int)(image.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset      
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin,image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center,r_center))
        
    return window_centroids 

def warp_image(img):
    img_size = (img.shape[1], img.shape[0])

    h = img.shape[0]
    w = img.shape[1]
    
    # source points        
    src = np.float32([[[ 610,    450]], 
                      [[ 720,    450]], 
                      [[ w-300,  720]],
                      [[ 380,    720]]])

    # offset
    offset = h*.33

    # destination points        
    dst = np.float32([[offset,      0], 
                      [w-offset,    0], 
                      [w-offset,    h], 
                      [offset,      h]])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Reversed transform matrix
    Minv = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)
    return warped, M, Minv

# Load testing images
images = glob.glob('./test_images/test*.jpg')
# window settings
window_width = 50 
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching
index = 0

ploty = np.linspace(0, 719, num=9)

def image_processing(img):
    img_size = (img.shape[1], img.shape[0])
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
    write_name = './output_images/undistort' + str(index) + '.jpg'
    cv2.imwrite(write_name, undistorted_img)
    preprocessed_img = preprocess_image(img)
    write_name = './output_images/preprocessed' + str(index) + '.jpg'
    cv2.imwrite(write_name, preprocessed_img)
    warped, perspective_M, perspective_Minv = warp_image(preprocessed_img)
    write_name = './output_images/warpped' + str(index) + '.jpg'
    cv2.imwrite(write_name, warped)
    window_centroids = find_window_centroids(warped)
    fitted, leftx, rightx = fit_curve(warped, window_centroids)
    write_name = './output_images/fitted' + str(index) + '.jpg'
    cv2.imwrite(write_name, fitted)
    leftx, rightx, left_fitx, right_fitx, left_curverad, right_curverad = measuring_curvature(leftx, rightx)
    result = draw_image(img, warped, perspective_M, left_fitx, right_fitx, left_curverad, right_curverad)
    return result    




def fit_curve(warped, window_centroids):
    leftx = []
    rightx = []

    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    # Go through each level and draw the windows 	
    for level in range(0,len(window_centroids)):

        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])
        # Window_mask is a function to draw window areas
        l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
        r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
        # Add graphic points from window mask here to total pixels found 
        l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
        r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

    # Draw the results
    template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
    zero_channel = np.zeros_like(template) # create a zero color channel
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
    out_img = np.dstack((warped, warped, warped))*255
    warpage = np.array(out_img,np.uint8) # making the original road pixels 3 color channels
    fitted = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results
    return fitted, leftx, rightx

    yvals = range(0, warped.shape[0])
    res_yvals = np.arange(warped.shape[0] - (window_height/2), 0, -window_height)

def img_process(img):
    img_size = (img.shape[1], img.shape[0])
    undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)
    write_name = './output_images/undistort' + str(index) + '.jpg'
    cv2.imwrite(write_name, undistorted_img)
    preprocessed_img = preprocess_image(img)
    write_name = './output_images/preprocessed' + str(index) + '.jpg'
    cv2.imwrite(write_name, preprocessed_img)
    warped, perspective_M, perspective_Minv = warp_image(preprocessed_img)

    write_name = './output_images/warpped' + str(index) + '.jpg'
    cv2.imwrite(write_name, warped)
    window_centroids = find_window_centroids(warped)
    
    leftx = []
    rightx = []

    l_points = np.zeros_like(warped)
    r_points = np.zeros_like(warped)

    # Go through each level and draw the windows 	
    for level in range(0,len(window_centroids)):

        leftx.append(window_centroids[level][0])
        rightx.append(window_centroids[level][1])
        # Window_mask is a function to draw window areas
        l_mask = window_mask(window_width,window_height,warped,window_centroids[level][0],level)
        r_mask = window_mask(window_width,window_height,warped,window_centroids[level][1],level)
        # Add graphic points from window mask here to total pixels found 
        l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
        r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

    # Draw the results
    template = np.array(r_points+l_points,np.uint8) # add both left and right window pixels together
    zero_channel = np.zeros_like(template) # create a zero color channel
    template = np.array(cv2.merge((zero_channel,template,zero_channel)),np.uint8) # make window pixels green
    out_img = np.dstack((warped, warped, warped))*255
    warpage = np.array(out_img,np.uint8) # making the original road pixels 3 color channels
    fitted = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

    yvals = range(0, warped.shape[0])
    res_yvals = np.arange(warped.shape[0] - (window_height/2), 0, -window_height)

    leftx = np.asarray(leftx[::-1])  # Reverse to match top-to-bottom in y
    rightx = np.asarray(rightx[::-1])  # Reverse to match top-to-bottom in y

    left_fit = np.polyfit(ploty, leftx, 2)      
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit = np.polyfit(ploty, rightx, 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    left_lane = np.array(list(zip(np.concatenate((left_fitx-window_width/2, left_fitx[::-1]+window_width/2), axis=0), np.concatenate((ploty, ploty[::-1]), axis=0))), np.int32)
    right_lane = np.array(list(zip(np.concatenate((right_fitx-window_width/2, right_fitx[::-1]+window_width/2), axis=0), np.concatenate((ploty, ploty[::-1]), axis=0))), np.int32)
    inner_lane = np.array(list(zip(np.concatenate((left_fitx+window_width/2, right_fitx[::-1]-window_width/2), axis=0), np.concatenate((ploty, ploty[::-1]), axis=0))), np.int32)

    road = np.zeros_like(img)
    road_bkg = np.zeros_like(img) 
    cv2.fillPoly(road, [left_lane], color= [255,0,0])
    cv2.fillPoly(road, [right_lane], color = [0,0,255])
    cv2.fillPoly(road, [inner_lane], color = [0,255,0])
    cv2.fillPoly(road_bkg, [left_lane], color = [255, 255, 255])
    cv2.fillPoly(road_bkg, [right_lane], color = [255, 255, 255])
    Minv = np.linalg.inv(perspective_M)

    road_warped = cv2.warpPerspective(road, Minv, img_size, flags = cv2.INTER_LINEAR)
    road_warped_bkg = cv2.warpPerspective(road_bkg, Minv, img_size, flags = cv2.INTER_LINEAR)
    base =   cv2.addWeighted(undistorted_img, 1.0, road_warped_bkg, -1.0, 0.0)
    
    result = cv2.addWeighted(base, 1.0, road_warped, 0.7, 0.0)    
    
    ym_per_pix = 10/720 # meters per pixel in y dimension
    xm_per_pix = 4/384 # meters per pixel in x dimension

    
    center = (left_fitx[-1] + right_fitx[-1])/2
    center_diff = (warped.shape[1]/2 - center)*xm_per_pix
    # Define conversions in x and y from pixels space to meters
    y_eval = np.max(ploty)
    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2) 
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    dir_str = 'right'
    if center_diff < 0:
        dir_str = 'left'
    cv2.putText(result, 'Vehicle is ' + str(abs(round(center_diff, 3))) + ' m ' + dir_str + ' of center', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, 'Radius of curvature = ' + str(round((left_curverad + right_curverad)/2, 3)) + ' (m)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return result


for idx, fname in enumerate (images):
    # read in image
    index = idx
    img = cv2.imread(fname)
    #result = image_processing(img)
    result = img_process(img)
    write_name = './output_images/tracked' + str(index) + '.jpg'
    cv2.imwrite(write_name, result)

Output_video = 'project_video_output.mp4'
#Output_video = 'challenge_video_output.mp4'
#Output_video = 'harder_challenge_video_output.mp4'
Input_video = 'project_video.mp4'
#Input_video = 'challenge_video.mp4'
#Input_video = 'harder_challenge_video.mp4'
clip1 = VideoFileClip(Input_video).subclip(41, 43)
video_clip = clip1.fl_image(img_process)
video_clip.write_videofile(Output_video, audio=False)
