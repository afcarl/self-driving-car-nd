import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils import *
import classifier as clf
import detector

def pipeline(image):
    image = detector.detect(image)
    return image

if __name__ == '__main__':
    model = clf.load_model('model.pkl')
    detector = detector.Detector(model, clf.settings)
    #utils.image_preview('./test_images/test1.jpg')
    #video_preview('test_video.mp4', pipeline)
    video_output('project_video.mp4', pipeline)
