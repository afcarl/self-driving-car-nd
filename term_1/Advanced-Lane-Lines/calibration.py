import numpy as np
import cv2
import pickle

def calibrate_camera(images_path, cols, rows):
    objp = np.zeros((rows*cols,3), np.float32)
    objp[:,:2] = np.mgrid[0:cols, 0:rows].T.reshape(-1,2)
    objpoints = []
    imgpoints = []
    # Step through the list and search for chessboard corners
    for filename in os.listdir(images_path):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    # calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    return mtx, dist

def undistort_image(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist

def save_data(filename, mtx, dist):
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open(filename, 'wb'))

def load_data(filename):
    dist_pickle = pickle.load(open(filename, 'rb'))
    mtx = dist_pickle["mtx"]
    dist = dist_pickle["dist"]
    return mtx, dist
