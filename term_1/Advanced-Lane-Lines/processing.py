import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def get_perspective_points():
    src = np.float32([
        [275, 680],
        [1045, 680],
        [734, 480],
        [554, 480]
    ])
    # Destination points
    dst = np.float32([
        [220, 700],
        [1020, 700],
        [1020, 100],
        [220, 100]
    ])
    return src, dst

def draw_perspective_points(image):
    src, dst = get_perspective_points()
    cv2.polylines(image,[src.astype(int)],True,(0,0,255)) #blue
    cv2.polylines(image,[dst.astype(int)],True,(255,0,0)) #red

def get_perspective_matrix(inverse=False):
    src, dst = get_perspective_points()
    if inverse:
        M = cv2.getPerspectiveTransform(dst, src)
    else:
        M = cv2.getPerspectiveTransform(src, dst)
    return M

def warp(image, inverse=False):
    """Applies perspective transform and returns transformed image.
    """
    M = get_perspective_matrix(inverse)
    warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
#    warped =cv2.resize(warped,(360, 480), interpolation = cv2.INTER_CUBIC)
    return warped

def yellow_white(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([80,100,100])
    upper_yellow = np.array([120,255,255])
    yellow_only = cv2.inRange(hsv, lower_yellow, upper_yellow)
    yellow_only[(yellow_only == 255)] = 1

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    white_only = np.zeros_like(l_channel)
    white_only[(l_channel >= 180) & (l_channel <= 255)] = 1

    res = np.zeros_like(white_only)
    res[(yellow_only == 1) | (white_only == 1)] = 1
    return res

def sobel_threshold(img, sx_thresh=(20, 100)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    return sxbinary

def s_channel_threshold(img, s_thresh=(170, 255)):
    # Convert to HLS color space and separate the L/S channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    s_channel = hls[:,:,2]
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    return s_binary

def gradient_saturation(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    sx_binary = sobel_threshold(img)
    s_binary = s_channel_threshold(img)
    combined_binary = np.zeros_like(sx_binary)
    combined_binary[((s_binary == 1) | (sx_binary == 1))] = 1
    return combined_binary

def gradient_color(image):
    yw = yellow_white(image)
    sx = sobel_threshold(image)
    combined_binary = np.zeros_like(yw)
    combined_binary[((yw == 1) & (sx == 1))] = 1
    return combined_binary
