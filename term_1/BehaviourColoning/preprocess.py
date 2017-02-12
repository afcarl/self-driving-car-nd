import pandas as pd
import numpy as np
import random
from scipy.misc import imread, imresize
import matplotlib.pyplot as plt
import cv2

def rgbToGray(rgb_image):
    result = np.zeros((32, 32, 1))
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    result = gray_image.reshape(32,32,1)
    return result

def crop_image(image):
    cropped_image = image[60:130, 0:image.shape[1]]
    return cropped_image

def read_image(path):
    path = path.replace(' ', '')
    image = imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def random_V(image, angle):
    HSV_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_v = 0.25 + np.random.uniform()
    HSV_image[:,:,2] = HSV_image[:,:,2]*random_v
    image = cv2.cvtColor(HSV_image, cv2.COLOR_HSV2RGB)
    return image, angle

def random_H(image, angle):
    HSV_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_h = 0.2 + np.random.uniform()
    HSV_image[:,:,0] = HSV_image[:,:,0]*random_h
    image = cv2.cvtColor(HSV_image, cv2.COLOR_HSV2RGB)
    return image, angle

def trans_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range * 0.4
    tr_y = 10*np.random.uniform()-10/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    rows,cols = image.shape[0:2]
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))    
    return tr_x, image_tr,steer_ang
 
def random_flip(image, angle):
    if np.random.random() > 0.4:
        image = cv2.flip(image, 1)
        angle = angle*(-1.0)
    return image, angle
       
def process_image(row, shape=(32, 32), angle_offset = 0.27):
    angle = row['steering']
    camera = np.random.choice(['center', 'left', 'right'])

    if camera == 'right':
        angle -= angle_offset
    elif camera == 'left':
        angle += angle_offset

    image = read_image(row[camera])
    image, angle = random_V(image, angle)
    image, angle = random_H(image, angle)
    
    image, angle = random_flip(image, angle)
    #image, angle = trans_image(image, angle, 100)    
    image = crop_image(image)
    cols, rows = shape
    image = cv2.resize(image, (cols, rows))

    image = image.astype(np.float32)
    return image, angle

def remove_zero_angle(driving_log, number):
    count = 0
    new_log = driving_log
    while count < number:
        index = random.randint(0, len(new_log)-1)
        angles = new_log['steering']
        if angles.iloc[index] == 0:
            new_log.drop(new_log.index[[index]], inplace=True)
            count += 1
    return new_log

def trans_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 10*np.random.uniform()-10/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    rows,cols = image.shape[0:2]
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    return tr_x, image_tr,steer_ang


def regen_data(driving_log):
    new_log = remove_zero_angle(driving_log, 3000)
    image_paths = pd.concat([new_log['center'], driving_log['left'], driving_log['right']])
    images = [read_image(path) for path in image_paths]
    mirror_paths = new_log['center']
    mirror_images = [np.fliplr(read_image(path)) for path in mirror_paths]
    images.extend(mirror_images)
    # should both side use the same adjustment? only if car drives in the center of the road
    angles = pd.concat([new_log['steering'], driving_log['steering'] + 0.27, driving_log['steering'] - 0.27, -new_log['steering']])
    angles = angles.tolist()
    return images, angles

def translate_images(images,angles):
    new_images = []
    new_angles = []
    for image, angle in zip(images,angles):
        new_image, new_angle = trans_image(image,angle,100)
        new_images.append(new_image)
        new_angles.append(new_angle)
    return new_images, new_angles 

if __name__ == '__main__':
    driving_log = pd.read_csv('driving_log.csv')
    image_paths = driving_log['center']
    new_log = remove_zero_angle(driving_log, 4000)
    
    angles = pd.concat([new_log['steering'], driving_log['steering'] + 0.27, driving_log['steering'] - 0.27, -new_log['steering']])
    angles = angles.tolist()
    plt.hist(angles,bins=1000)

    index = random.randint(0, len(image_paths) - 1)
    image = read_image(image_paths[index])
    image_new, angle = process_image(new_log.iloc[index])
    
    print("angle ", angle)
    tx, image_new, angle_new = trans_image(image, angle, 100)
    print("after ", angle_new, "tx " ,tx)
    plt.figure(figsize=(1,1))
#plt.imshow(image)
#    plt.imshow(image_new)
#    plt.show()

