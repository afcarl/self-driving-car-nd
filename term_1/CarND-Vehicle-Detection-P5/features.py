import cv2
import glob
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog
import classifier as clf
from utils import *

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):

    return hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                      cells_per_block=(cell_per_block, cell_per_block),
                      transform_sqrt=True, visualise=vis, feature_vector=feature_vec)

def hog_features_for_channel(img, settings):
    """ return hog features for any channel of image or all channel
    """
    orient = settings['orient']
    pix_per_cell = settings['pix_per_cell']
    cell_per_block = settings['cell_per_block']
    channel_value = settings['hog_channel']
    channels = [0,1,2] if  channel_value == 'ALL' else [channel_value]
    hog_features = []
    for channel in channels:
        hog_features.extend(get_hog_features(img[:,:,channel], orient, pix_per_cell, cell_per_block))

    return np.ravel(hog_features)

def bin_spatial(img, size=(32, 32)):
    features = cv2.resize(img, size).ravel()
    return features

def color_hist(img, nbins=32, bins_range=(0, 1)):
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features

def convert_color(image, color_space):
    color_dict = { 'HSV' : cv2.COLOR_RGB2HSV, 'LUV' : cv2.COLOR_RGB2LUV, \
                   'HLS' : cv2.COLOR_RGB2HLS, 'YUV' : cv2.COLOR_RGB2YUV, \
                   'YCrCb': cv2.COLOR_RGB2YCrCb }

    value = color_dict.get(color_space)
    if value != None:
        return cv2.cvtColor(image, value)
    else:
        return np.copy(image)

def single_features(image, settings):
    """ extract features from a single image, e.g., a search window.
    """
    img_features = []
    feature_image = convert_color(image, settings['color_space'])
    if settings['spatial_feat'] == True:
        spatial_features = bin_spatial(feature_image, size=settings['spatial_size'])
        img_features.append(spatial_features)
    if settings['hist_feat'] == True:
        hist_features = color_hist(feature_image, nbins=settings['hist_bins'])
        img_features.append(hist_features)
    if settings['hog_feat'] == True:
        hog_features = hog_features_for_channel(feature_image, settings)
        img_features.append(hog_features)

    return np.concatenate(img_features)

def extract_features(imgs, settings):
    """ extract features from a list of images
    """
    features = []
    for file in imgs:
        file_features = []
        image = mpimg.imread(file)
        image_features = single_features(image, settings)
        features.append(image_features)
    return features

if __name__ == '__main__':
    cars = glob.glob('./vehicles/*/*.png')
    notcars = glob.glob("./non-vehicles/*/*.png")
    print(len(cars), len(notcars))
    index = np.random.randint(0, len(cars))
    car_image = mpimg.imread(cars[index])
    notcar_image = mpimg.imread(notcars[index])
    cov_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2YCrCb)
    settings = clf.settings
    fig, ((ax1, ax2, ax3, ax4), ((ax5, ax6, ax7, ax8))) = plt.subplots(nrows=2, ncols=4)
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.98)

    example_plot(ax1, car_image, 'Car Image')
    features, hog_image1 = get_hog_features(cov_image[:,:,0], settings['orient'], settings['pix_per_cell'], settings['cell_per_block'], vis=True, feature_vec=False)
    example_plot(ax2, hog_image1, 'Y channel')
    features, hog_image2 = get_hog_features(cov_image[:,:,1], settings['orient'], settings['pix_per_cell'], settings['cell_per_block'], vis=True, feature_vec=False)
    example_plot(ax3, hog_image2, 'Cr channel')
    features, hog_image3 = get_hog_features(cov_image[:,:,2], settings['orient'], settings['pix_per_cell'], settings['cell_per_block'], vis=True, feature_vec=False)
    example_plot(ax4, hog_image3, 'Cb channel')

    cov_image = cv2.cvtColor(notcar_image, cv2.COLOR_RGB2YCrCb)
    example_plot(ax5, notcar_image, 'Not Car Image')
    features, hog_image1 = get_hog_features(cov_image[:,:,0], settings['orient'], settings['pix_per_cell'], settings['cell_per_block'], vis=True, feature_vec=False)
    example_plot(ax6, hog_image1, 'Y channel')
    features, hog_image2 = get_hog_features(cov_image[:,:,0], settings['orient'], settings['pix_per_cell'], settings['cell_per_block'], vis=True, feature_vec=False)
    example_plot(ax7, hog_image2, 'Cr channel')
    features, hog_image3 = get_hog_features(cov_image[:,:,0], settings['orient'], settings['pix_per_cell'], settings['cell_per_block'], vis=True, feature_vec=False)
    example_plot(ax8, hog_image3, 'Cb channel')
    plt.show()
