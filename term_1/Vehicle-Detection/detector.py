import cv2
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import utils
from features import *

class Detector():
    def __init__(self, model, settings):
        self.settings = settings
        self.X_scaler = model['scaler']
        self.clf = model['clf']
        self.y_start_stop = settings['y_start_stop']
        self.scale = settings['scale']
        self.orient = settings['orient']
        self.pix_per_cell = settings['pix_per_cell']
        self.cell_per_block = settings['cell_per_block']
        self.spatial_size = settings['spatial_size']
        self.hist_bins = settings['hist_bins']
        self.color_space = settings['color_space']
        self.smoother = utils.moving_average(10)
        self.ypos_scales = [([400,480], 0.7), ([400,550], 1.0), ([400,600], 1.5), ([400,660], 2), ([400,660], 2.5)]
        self.vehicles = []

    def search_windows(self, img, windows):
        img = img.astype(np.float32)/255
        on_windows = []
        for window in windows:
            test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
            features = single_features(test_img, clf.settings)
            test_features = self.scaler.transform(np.array(features).reshape(1, -1))
            prediction = self.clf.predict(test_features)
            if prediction == 1:
                on_windows.append(window)

        return on_windows

    def _find_cars(self, img, y_start_stop, scale, clf, X_scaler, color_space, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, debug=False):
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

        count = 0
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
                combined_features = np.hstack((spatial_features, hist_features, hog_features))
                test_features = X_scaler.transform(combined_features.reshape(1,-1))
                test_prediction = clf.predict(test_features)

                if test_prediction == 1 or debug:
                    xbox_left = np.int(xleft*scale)
                    ytop_draw = np.int(ytop*scale)
                    win_draw = np.int(window*scale)
                    car_window = ((xbox_left, ytop_draw+ystart), (xbox_left+win_draw, ytop_draw+win_draw+ystart))
                    car_windows.append(car_window)

        return car_windows

    def find_cars(self, image, y_start_stop, scale, debug=False):
        return self._find_cars(image, y_start_stop, scale,
                self.clf, self.X_scaler, self.color_space, self.orient,
                self.pix_per_cell, self.cell_per_block, self.spatial_size, self.hist_bins, debug)

    def draw_debug_windows(self, image):
        window_img = np.copy(image)
        color_option_iter = iter([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)])
        for ypos, scale in self.ypos_scales:
            scale_windows = self.find_cars(image,y_start_stop=ypos,scale=scale,debug=True)
            # one_window = [scale_windows[0]]
            window_img = utils.draw_boxes(window_img, scale_windows, color=next(color_option_iter), thick=4)
        return window_img

    def find_multiscale(self, image):
        out_windows=[]
        for ypos, scale in self.ypos_scales:
            scale_windows = self.find_cars(image,y_start_stop=ypos,scale=scale)
            out_windows.extend(scale_windows)
        return out_windows

    def detect(self, image, do_hog_once=True):
        out_windows = []
        if do_hog_once:
            out_windows = self.find_multiscale(image)
            heatmap = utils.get_heatmap(image, out_windows)
            heatmap = self.smoother(heatmap)
            heatmap = utils.apply_threshold(heatmap,3)
            boxes = utils.get_labeled_boxes(heatmap)
        else:
            windows = utils.slide_window(image, x_start_stop=[None, None], y_start_stop=self.y_start_stop,
                        xy_window=(64, 64), xy_overlap=(0.85, 0.85))
            boxes = self.search_windows(image, windows)

        final_image = utils.draw_boxes(image, boxes, color=(0, 0, 255), thick=6)
        return final_image


if __name__ == '__main__':
    from classifier import load_model, settings
    model = load_model('model.pkl')
    detector = Detector(model, settings)

    images_path = './test_images/'
    fig = plt.figure(figsize=(10,10))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.98)
    row = len(os.listdir(images_path))
    col = 3
    index = 0
    for filename in os.listdir(images_path):
        image = mpimg.imread(images_path + filename)

        index += 1
        plt.subplot(row, col, index)
        out_windows = detector.find_multiscale(image)
        window_img = utils.draw_boxes(image, out_windows, color=(0, 0, 255), thick=6)
        plt.imshow(window_img)
        plt.title('detected windows')

        index += 1
        plt.subplot(row, col, index)
        heatmap = utils.get_heatmap(image, out_windows)
        # Apply threshold to help remove false positives
        heatmap = apply_threshold(heatmap,3)
        plt.imshow(heatmap, cmap='hot')
        plt.title('Heat Map')

        index += 1
        plt.subplot(row, col, index)
        boxes = utils.get_labeled_boxes(heatmap)
        final_img = utils.draw_boxes(image, boxes, color=(0, 0, 255),thick=6)
        plt.imshow(final_img)
        plt.title('Car position')
    plt.show()

    # image = mpimg.imread(images_path + 'test1.jpg')
    # # search_windows = detector.draw_debug_windows(image)
    # final_image = detector.detect(image)
    # plt.imshow(final_image)
    # plt.show()
