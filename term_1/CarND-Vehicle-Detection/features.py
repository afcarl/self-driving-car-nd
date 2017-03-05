import cv2
import glob
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.feature import hog
import classifier as clf

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

# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 1)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
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
        hog_features = hog_features_for_channel(image, settings)
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
    image = mpimg.imread(cars[index])
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    settings = clf.settings
    features, hog_image = get_hog_features(gray, settings['orient'], settings['pix_per_cell'], settings['cell_per_block'], vis=True, feature_vec=False)

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(hog_image, cmap='gray')
    plt.title('HOG Visualization')
    plt.show()
