import cv2
import os
import utils
import numpy as np
import calibration
import detector
import processing as proc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

def plot_warped(image, func, title):
    image = calibration.undistort_image(image, mtx, dist)
    warped = proc.warp(image)
    binary_warped = func(warped)
    fig = plt.figure()
    fig.suptitle(title, fontsize=16)
    fig.add_subplot(1,2,1)
    plt.imshow(warped)
    fig.add_subplot(1,2,2)
    plt.imshow(binary_warped,cmap='gray')

def pipeline(image):
    image = calibration.undistort_image(image, mtx, dist)
    warped = proc.warp(image)
    detector.detect_lane(warped, debug=False)
    result = detector.draw_final_lane(image)
    return result

def plot_processing(name):
    image = mpimg.imread(name)
    plot_warped(image, proc.yellow_white, 'yellow_white')
    plot_warped(image, proc.gradient_color, 'gradient_color')
    plot_warped(image, proc.gradient_saturation, 'gradient_saturation')
    plt.show()

def video_preview(source, pipeline):
    clip = VideoFileClip(source)
    clip = clip.fl_time(lambda t:t+39)
    output = clip.fl_image(pipeline)
    output.duration = 5
    output.preview()

def video_output(source, pipeline):
    clip = VideoFileClip(source)
    dot = '.'
    name, extension = source.split(dot)
    output_name = name + '_out' + dot + extension
    output = clip.fl_image(pipeline)
    output.write_videofile(output_name, audio=False)

def image_preview(file_name):
    image = mpimg.imread(file_name)
    warped = proc.warp(image)
    detector.lane_width = 800 # bypass lane width estimation stage
    detector.detect_lane(warped, debug=True)
    image = detector.draw_final_lane(image)
    plt.figure()
    plt.imshow(image)
    plt.show()

if __name__ == '__main__':
    # calibrate the camera if not already done so
    data_file = './calibration_data.p'
    if not os.path.exists(data_file):
        mtx, dist = calibration.calibrate_camera('./camera_cal', 9, 6)
        calibration.save_data(data_file, mtx, dist)

    # load calibration data
    mtx, dist = calibration.load_data(data_file)
    #utils.plot_all('./test_images', calibration.undistort_image, mtx, dist)
    detector = detector.Detector()

    image_preview('./test_images/test4.jpg')
    #video_preview('project_video.mp4', pipeline)
    #video_output('challenge_video.mp4', pipeline)
