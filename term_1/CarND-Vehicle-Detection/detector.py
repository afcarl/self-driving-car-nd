import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import utils
import classifier as clf
from scipy.ndimage.measurements import label
from features import *

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
    img = np.copy(img)
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
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def get_heatmap(image, box_list):
    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,box_list)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)
    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    return heatmap

class Detector():
    def __init__(self):
        pass

    # Define a function you will pass an image
    # and the list of windows to be searched (output of slide_windows())
    def search_windows(self, img, windows, model):
        # scale to (0,1) as classifer is trained on png(0,1) but test image is jpeg (0,255)
        img = img.astype(np.float32)/255
        #1) Create an empty list to receive positive detection windows
        on_windows = []
        #2) Iterate over all windows in the list
        for window in windows:
            #3) Extract the test window from original image
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            #4) Extract features for that window using single_img_features()
            features = single_features(test_img, clf.settings)
            #5) Scale extracted features to be fed to classifier
            test_features = model['scaler'].transform(np.array(features).reshape(1, -1))
            #6) Predict using your classifier
            prediction = model['clf'].predict(test_features)
            #7) If positive (prediction == 1) then save the window
            if prediction == 1:
                on_windows.append(window)
        #8) Return windows for positive detections
        return on_windows

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, img, y_start_stop, scale, svc, X_scaler, color_space, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
        img = img.astype(np.float32)/255
        ystart = y_start_stop[0]
        ystop = y_start_stop[1]
        img_tosearch = img[ystart:ystop,:,:]
        ctrans_tosearch = convert_color(img_tosearch, color_space)
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

        car_windows = []
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
                test_prediction = svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    car_window = ((xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart))
                    car_windows.append(car_window)

        return car_windows

    def detect(self, model, settings, image):
        # svc = model['clf']
        # X_scaler = model['scaler']
        # y_start_stop = clf.settings['y_start_stop']
        # scale = clf.settings['scale']
        # orient = clf.settings['orient']
        # pix_per_cell = clf.settings['pix_per_cell']
        # cell_per_block = clf.settings['cell_per_block']
        # spatial_size = clf.settings['spatial_size']
        # hist_bins = clf.settings['hist_bins']
        # color_space = clf.settings['color_space']
        # hot_windows = detector.find_cars(image, y_start_stop, scale, svc, X_scaler, color_space, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        # window_img = utils.draw_boxes(image, hot_windows, color=(0, 0, 255), thick=6)

        windows = utils.slide_window(image, x_start_stop=[None, None], y_start_stop=settings['y_start_stop'],
                            xy_window=(64, 64), xy_overlap=(0.75, 0.75))
        hot_windows = detector.search_windows(image, windows, model)
        window_img = utils.draw_boxes(image, hot_windows, color=(0, 0, 255), thick=6)
        return hot_windows, window_img

    def draw_test_images(self, images_path):
        model = clf.load_model('model.pkl')
        fig = plt.figure(figsize=(9,9))
        row = len(os.listdir(images_path))
        col = 3
        index = 0
        for filename in os.listdir(images_path):
            image = mpimg.imread(images_path + filename)

            index += 1
            plt.subplot(row, col, index)
            hot_windows, window_img = self.detect(model, clf.settings, image)
            plt.imshow(window_img)
            plt.title('detected windows')

            index += 1
            plt.subplot(row, col, index)
            heatmap = get_heatmap(image, hot_windows)
            plt.imshow(heatmap, cmap='hot')
            plt.title('Heat Map')

            index += 1
            plt.subplot(row, col, index)
            final_img = draw_labeled_bboxes(image, label(heatmap))
            plt.imshow(final_img)
            plt.title('Car position')

        plt.show()

if __name__ == '__main__':
    detector = Detector()
    detector.draw_test_images('./test_images/')
