import glob
import time
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from features import *
import pickle

settings = \
{
    'color_space'    : 'YCrCb', # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    'orient'         : 9,  # HOG orientations
    'pix_per_cell'   : 8, # HOG pixels per cell
    'cell_per_block' : 2, # HOG cells per block
    'hog_channel'    : 'ALL', # Can be 0, 1, 2, or "ALL"
    'spatial_size'   : (32, 32), # Spatial binning dimensions
    'hist_bins'      : 32,    # Number of histogram bins
    'spatial_feat'   : True, # Spatial features on or off
    'hist_feat'      : True, # Histogram features on or off
    'hog_feat'       : True, # HOG features on or off
    'scale'          : 1.5, # used in find_car
    'y_start_stop'   : [400, 656]
}

def load_model(filename):
    model = pickle.load(open(filename, 'rb'))
    return model

def save_model(filename, classifier, scaler):
    model = {'clf': classifier, 'scaler': scaler}
    pickle.dump(model, open(filename, 'wb'))

def train_model():
    cars = glob.glob('./vehicles/*/*.png')
    notcars = glob.glob("./non-vehicles/*/*.png")
    car_features = extract_features(cars, settings)
    notcar_features = extract_features(notcars, settings)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Feature vector length:', len(X_train[0]))
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
    save_model('model.pkl', svc, scaler)

def test_prediction():
    model = load_model('model.pkl')
    image = mpimg.imread('./test_images/test1.jpg')
    image = image.astype(np.float32)/255
    print(image)
    window = ((813, 412), (940, 494))
    test_img = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))

    features = single_features(test_img, settings)
    test_features = model['scaler'].transform(np.array(features).reshape(1, -1))
    prediction = model['clf'].predict(test_features)
    print('prediction: ', prediction)

if __name__ == '__main__':
    train_model()
    #test_prediction()
    # fig = plt.figure()
    # cars = glob.glob('./vehicles/*/*.png')
    # index = np.random.randint(0, len(cars))
    # image = mpimg.imread(cars[index])
    # plt.imshow(image)
    # plt.show()
