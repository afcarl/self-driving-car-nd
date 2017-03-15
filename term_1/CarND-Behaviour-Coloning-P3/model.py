import cv2
import pandas as pd
import numpy as np
import random
import json
import csv
from keras.layers.core import Dense, Flatten, Dropout, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

rows, cols, ch = 32, 32, 3
batch_size = 100
test_size = 0.1
angle_offset = 0.27
epoch_count = 24
samples_to_remove = 4000
samples_per_epoch = 20000
validation_sample_count = 2000

def remove_zero_angle(samples, entry_count):
    count = 0
    result = samples
    while count < entry_count:
        index = np.random.randint(0, len(samples)-1)
        entry = samples[index]
        angles = float(entry[3])
        if angles == 0:
            result.remove(entry)
            count += 1
    return result

def plot_steering_distribution(samples):
    angles = [float(entry[3]) for entry in samples]
    plt.hist(angles,bins=1000)

def plot_image(samples):
    index = random.randint(0, len(samples) - 1)
    filename = samples[index][0].replace(" ", "")
    image = cv2.imread(filename)
    new_image, angle = augment_and_process(image, 0)
    fig = plt.figure()
    fig.add_subplot(1,2,1)
    plt.imshow(image)
    fig.add_subplot(1,2,2)
    plt.imshow(new_image)
    plt.show()

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

def random_flip(image, angle):
    if np.random.random() > 0.4:
        image = cv2.flip(image, 1)
        angle = angle*(-1.0)
    return image, angle

def crop_image(image):
    cropped_image = image[55:135, 0:image.shape[1]]
    return cropped_image

def augment_and_process(image, angle, shape=(32,32)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image, angle = random_V(image, angle)
    image, angle = random_H(image, angle)
    image, angle = angle_jitter(image, angle)
    image, angle = random_flip(image, angle)

    image = crop_image(image)
    image = cv2.resize(image, shape)
    image = image.astype(np.float32)
    return image, angle

def get_samples(filename):
    samples = []
    with open(filename) as csvfile:
        catergory = next(csvfile)
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    samples = remove_zero_angle(samples, samples_to_remove)
    return samples

def save_parameters(m):
    json_file = open('model.json', mode='w')
    json.dump(m.to_json(), json_file)

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

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                cameraIndex = random.randint(0, 2)
                filename = batch_sample[cameraIndex].replace(" ", "")
                image = cv2.imread(filename)
                angle = float(batch_sample[3])
                if cameraIndex == 2: #right
                    angle -= angle_offset
                elif cameraIndex == 1: #left
                    angle += angle_offset
                image, angle = augment_and_process(image, angle, shape=(cols, rows))
                images.append(image)
                angles.append(angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

samples = get_samples('./driving_log.csv')

# split training validation set
train_samples, validation_samples = train_test_split(samples, test_size=test_size)
print("train_samples ", len(train_samples), "validation_samples ", len(validation_samples))

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

model = the_model()
model.summary()
model.compile(optimizer=Adam(lr=0.0001), loss='mse')

checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1)
callbacks = [checkpoint, early_stop]

model.fit_generator(train_generator, samples_per_epoch=samples_per_epoch, validation_data=validation_generator,
                    nb_val_samples=validation_sample_count, nb_epoch=epoch_count, callbacks=callbacks)

save_parameters(model)
