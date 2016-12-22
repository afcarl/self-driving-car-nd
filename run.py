#importing some useful packages
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)
    
def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

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

def to_point(line):
    x1,y1,x2,y2 = line
    if x2-x1 == 0:
        return 0, x1
    if y2-y1 == 0:
        return np.pi/2, y1
    slope = (y2-y1)/(x2-x1)
    theta = math.atan(-1/slope)
    rho = x1*math.cos(theta) + y1*math.sin(theta)
    return theta,rho

def find_candidate(lines, theta_diff, rho_diff):
    vote = {}
    for i in range(len(lines)):
        theta_i,rho_i = to_point(lines[i])
		#print(lines[i], "theta_i", theta_i, "rho_i", rho_i)
        if (theta_i > (-np.pi/2 + np.pi/9) and theta_i < (np.pi/2 - np.pi/9)): #filter out horizontal segment
            for key in vote.keys():
                theta, rho = key
                if abs(theta-theta_i) < theta_diff and abs(rho-rho_i) < rho_diff:
                    vote[key] += 1
            if (theta_i,rho_i) not in vote:
                vote[(theta_i,rho_i)] = 1
    
    if not vote:
        return vote 
    
    theta_candidate, rho_candidate = max(vote, key=lambda i: vote[i])
    return theta_candidate, rho_candidate

def filter(lines, theta_maj, rho_maj, theta_diff, rho_diff):
    lane_lines = []
    if theta_maj == None or rho_maj == None:
        return lane_lines
    for line in lines:
        theta_i,rho_i = to_point(line)
        if abs(theta_maj-theta_i) < theta_diff and abs(rho_maj-rho_i) < rho_diff:
            lane_lines.append(line)            
			#print("lane_lines", line, "theta_i", theta_i, "rho_i", rho_i) 
    return lane_lines

def bottom_x(bottom_y, line):
    x1,y1,x2,y2 = line
    theta,rho = to_point(line)
    if theta == np.pi/2: # horizontal line
        assert(False), "horizontal line has no intersection with x axis"
        return 0 
    if theta == 0: 
        return x1
    else:
        k = -1/math.tan(theta)
    c = y1 - k*x1
    return int(round((bottom_y - c)/k))

def extend_lines(lane_lines, img_height):
    if lane_lines == None:
        return
    average_lines = []
    for i in range(len(lane_lines)-1):
        x1,y1,x2,y2 = lane_lines[i]
        x_1,y_1,x_2,y_2 = lane_lines[i+1]
        average_lines.append(((x1+x2)//2,(y1+y2)//2,(x_1+x_2)//2,(y_1+y_2)//2))
    lane_lines += average_lines	
    
    point_y_list = []
    for line in lane_lines:
        x1,y1,x2,y2 = line
        point_y_list.extend([y1,y2])
    point_y_list.sort()
    
    if len(point_y_list) <= 1:
        return
    y_max = point_y_list[-1]
    for line in lane_lines:
        x1,y1,x2,y2 = line
        if y_max == y1 or y_max == y2:
            bottom_line = line        
            break;
    bx = bottom_x(img_height,bottom_line)
    lane_lines.append((bx,img_height,bottom_line[0],bottom_line[1]))

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
   
    # divide into lef and right group 
    lane_seperator = img.shape[1]//2
    left_lines = []
    right_lines = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if (x1+x2)/2 < lane_seperator:
                left_lines.append((x1,y1,x2,y2))
            else:
                right_lines.append((x1,y1,x2,y2))
    
	# process left lines
    theta_maj,rho_maj = find_candidate(left_lines, 0.2, 100) or (None, None)
        
    lane_lines = filter(left_lines, theta_maj, rho_maj, 0.1, 10)
    
    extend_lines(lane_lines, img.shape[0]) 

    for line in lane_lines:
        x1,y1,x2,y2 = line
        cv2.line(img, (x1,y1), (x2,y2), color, thickness)    
 
    # process right lines
    theta_maj,rho_maj = find_candidate(right_lines, 0.1, 20) or (None, None)
    
    lane_lines = filter(right_lines, theta_maj, rho_maj, 0.1, 20)
    
    extend_lines(lane_lines, img.shape[0])
    
    for line in lane_lines:
        x1,y1,x2,y2 = line
        cv2.line(img, (x1,y1), (x2,y2), color, thickness)    

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def process_image(image):
    kernel_size = 5
    gray_image = grayscale(image)
    blur_gray = gaussian_blur(gray_image, kernel_size)

    edge_image = canny(blur_gray, 70, 200)

    imshape = image.shape
    vertices = np.array([[(0,imshape[0]), (imshape[1]//2 - 50,
                int(0.6*imshape[0])), (imshape[1]//2 + 50, int(0.6*imshape[0])), (imshape[1],imshape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edge_image, vertices)

    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 10     # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 5 #minimum number of pixels making up a line
    max_line_gap = 10    # maximum gap in pixels between connectable line segmen
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)
    
    result = weighted_img(line_image, image)
    return result

def test_images():
    test_images = os.listdir("test_images/")
    for test_image in test_images:
        #reading in an image
        image = mpimg.imread('test_images/'+test_image)
        print('This image is:', type(image), 'with dimesions:', image.shape)
        result = process_image(image)
        plt.imshow(result, cmap='gray')  #call as plt.iMSHOW(GRAY, CMAP='GRay') to show a grayscaled image
        plt.show()

def test_video():
    white_output = 'white.mp4'
    clip1 = VideoFileClip("solidWhiteRight.mp4")
    white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
    white_clip.preview()
#white_clip.write_videofile(white_output, audio=False)  
    yellow_output = 'yellow.mp4'
    clip2 = VideoFileClip("solidYellowLeft.mp4")
    yellow_clip = clip2.fl_image(process_image) 
    yellow_clip.preview()
#yellow_clip.write_videofile(yellow_output, audio=False)
    challenge_output = 'challenge_out.mp4'
    clip3 = VideoFileClip("challenge.mp4")
    challenge_clip = clip3.fl_image(process_image) 
    challenge_clip.preview()
	#
test_images()
test_video()
#white_clip.write_videofile(white_output, audio=False)
