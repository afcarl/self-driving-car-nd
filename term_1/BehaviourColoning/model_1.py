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
from preprocess import remove_zero_angle, process_image


rows, cols, ch = 32, 32, 3
batch_size = 100
split_size = 0.1
samples_per_epoch = 20000
validation_samples = 2000
epoch_count = 4

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
    
    model.compile(optimizer="adam", loss="mse")
    return model

def batch_generator(data):
    batch_count = data.shape[0] // batch_size
    i = 0
    while 1:
        batch_features = np.zeros((batch_size, rows, cols, ch), dtype=np.float32)
        batch_labels = np.zeros((batch_size,), dtype=np.float32)
        
        j = 0
        for _, row in data.loc[i*batch_size: (i+1)*batch_size - 1].iterrows():
            batch_features[j], batch_labels[j] = process_image(row)
            j += 1
        
        i += 1
        if i == batch_count - 1:
            i = 0
        yield batch_features, batch_labels


def save_parameters(m):
    m.save_weights('model.h5')
    json_file = open('model.json', mode='w')
    json.dump(m.to_json(), json_file)

model = the_model()
model.summary()

# removing zero angle rows from original log 
log = pd.read_csv('driving_log.csv')
log = remove_zero_angle(log, 3000)

# split training and validation dataset
log = log.sample(frac=1).reset_index(drop=True)
training_data = log.loc[0:(log.shape[0]*(1.0-split_size)) - 1]
validation_data = log.loc[log.shape[0]*(1.0-split_size):]

# training the model
model.fit_generator(batch_generator(training_data), 
                    samples_per_epoch= samples_per_epoch,
                    nb_epoch=epoch_count,
                    verbose=1,
                    validation_data=batch_generator(validation_data),
                    nb_val_samples=validation_samples)

save_parameters(model)

