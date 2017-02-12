import cv2
import pandas as pd
import numpy as np
import json
from keras.layers.core import Dense, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

rows, cols, ch = 32, 32, 3
batch_size = 100
split_size = 0.1
samples_per_epoch = 20000
angle_offset = 0.27
validation_samples = 2000
epoch_count = 4

def remove_zero_angle(driving_log, number):
    count = 0
    new_log = driving_log
    while count < number:
        index = np.random.randint(0, len(new_log)-1)
        angles = new_log['steering']
        if angles.iloc[index] == 0:
            new_log.drop(new_log.index[[index]], inplace=True)
            count += 1
    return new_log

# For learning roads with different brightness
def random_V(image, angle):
    HSV_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_v = 0.25 + np.random.uniform()
    HSV_image[:,:,2] = HSV_image[:,:,2]*random_v
    image = cv2.cvtColor(HSV_image, cv2.COLOR_HSV2RGB)
    return image, angle

# For learning roads with different main color
def random_H(image, angle):
    HSV_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_h = 0.2 + np.random.uniform()
    HSV_image[:,:,0] = HSV_image[:,:,0]*random_h
    image = cv2.cvtColor(HSV_image, cv2.COLOR_HSV2RGB)
    return image, angle

def angle_jitter(image, angle):
    angle = angle + 0.05*(np.random.uniform() - 0.5)
    return image, angle

# To generate more data
def random_flip(image, angle):
    if np.random.random() > 0.4:
        image = cv2.flip(image, 1)
        angle = angle*(-1.0)
    return image, angle

def trans_image(image,steer,trans_range):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 10*np.random.uniform()-10/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    rows,cols = image.shape[0:2]
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    return image_tr,steer_ang

def crop_image(image):
    cropped_image = image[55:135, 0:image.shape[1]]
    return cropped_image

def augment_and_process(row):
    angle = row['steering']
    camera = np.random.choice(['center', 'left', 'right'])

    if camera == 'right':
        angle -= angle_offset
    elif camera == 'left':
        angle += angle_offset

    path = row[camera]
    datapath = path.replace(" ", "")

    image = cv2.imread(datapath)
    assert(image != None)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image, angle = random_V(image, angle)
    image, angle = random_H(image, angle)
    image, angle = angle_jitter(image, angle)
    image, angle = random_flip(image, angle)

    image = crop_image(image)
    image = cv2.resize(image, (cols, rows))
    image = image.astype(np.float32)
    return image, angle

def batch_generator(data):
    batch_count = data.shape[0] // batch_size
    i = 0
    while 1:
        batch_features = np.zeros((batch_size, rows, cols, ch), dtype=np.float32)
        batch_labels = np.zeros((batch_size,), dtype=np.float32)

        j = 0
        for _, row in data.loc[i*batch_size: (i+1)*batch_size - 1].iterrows():
            batch_features[j], batch_labels[j] = augment_and_process(row)
            j += 1

        i += 1
        if i == batch_count - 1:
            i = 0
        yield batch_features, batch_labels

def the_model():
    model = Sequential()

    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(rows, cols, ch)))
    model.add(Convolution2D(32, 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2)))

    model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(ELU())
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dropout(0.2))

    model.add(Dense(512, name='Dense0'))
    model.add(Dropout(0.4))
    model.add(ELU())

    model.add(Dense(128, name='Dense1'))
    model.add(ELU())

    model.add(Dense(1, name='Out'))
    return model

def save_parameters(m):
    json_file = open('model.json', mode='w')
    json.dump(m.to_json(), json_file)

# load and remove zero steering data
log = pd.read_csv('driving_log.csv')
log = remove_zero_angle(log, 4000)
log = log.sample(frac=1).reset_index(drop=True)

# split training validation set
training_data = log.loc[0:(log.shape[0]*(1.0-split_size)) - 1]
validation_data = log.loc[log.shape[0]*(1.0-split_size):]

print("training len ", len(training_data))
# model and training
model = the_model()
model.summary()
model.compile(optimizer=Adam(lr=0.0001), loss='mse')
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True)
#early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1)
callbacks = [checkpoint]

model.fit_generator(batch_generator(training_data),
                    samples_per_epoch=samples_per_epoch,
                    nb_epoch=epoch_count,
                    verbose=1,
                    validation_data=batch_generator(validation_data),
                    nb_val_samples=validation_samples,
                    callbacks=callbacks)

save_parameters(model)
