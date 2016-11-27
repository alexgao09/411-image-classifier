from __future__ import print_function

from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.layers import Activation, BatchNormalization, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

batch_size = 32
nb_classes = 8
nb_epoch = 1
data_augmentation = True

# input image dimensions
img_rows, img_cols = 128, 128
# the CIFAR10 images are RGB
img_channels = 1


import numpy as np
import csv
import os
import cv2

N = 7000 # Number of training examples
M = 970 # Number of test cases
cwd = os.getcwd()


def get_training_data():
    '''
    Return training and validation data.
    '''

    path = os.path.join(cwd, 'train')
    image_list=[]
    training_data = np.zeros((N,128,128,1))

    for file in os.listdir(path):
        if file.endswith(".jpg"):
            image_list.append(file)

    index = 0
    for i in image_list: # get all the training data - we can split this after
        training_data[index,:,:,0] = cv2.imread(path+"/"+i,0)
        index += 1
        
    training_data = training_data.astype("float32")
    training_data /= 255 ######################################################## WHY DOES IT WORK WHEN WE DIVIDE ALL OF A SUDDEN?

    full_labels=[]
    with open('train.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            full_labels.append(row[0].split(',')[1])
            
    full_labels=full_labels[1:]

    labels = full_labels[0:N]
    labels = map(int,labels)
    print(labels[0:5])
    print(labels[6995:7000])
    labels = np.array(labels,dtype=np.uint8).reshape(-1,1) # The dtype is VERY important
    labels -= 1 # THIS IS VERY IMPORTANT


    X_train = training_data[0:6000,:,:,:]
    Y_train = labels[0:6000,:]

    X_valid = training_data[6000:7000,:,:,:] # For some reason, we need to do up to 6998 - 7000 doesn't work
    Y_valid = labels[6000:7000,:]

    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_valid = np_utils.to_categorical(Y_valid, nb_classes)

    return X_train, Y_train, X_valid, Y_valid

def get_testing_data():

    path = os.path.join(cwd, 'val')
    image_list=[]
    testing_data = np.zeros((M,128,128,1))

    for file in os.listdir(path):
        if file.endswith(".jpg"):
            image_list.append(file)

    index = 0
    for i in image_list:
        testing_data[index,:,:,0] = cv2.imread(i,0)
        index += 1
        
    testing_data = testing_data.astype("float32")
    testing_data /= 255 ######################################################## WHY DOES IT WORK WHEN WE DIVIDE ALL OF A SUDDEN?

    X_test = testing_data[0:970,:,:,:]

    return X_test

print("Done data pre-processing")

def make_predictions(model):
    
    # predict on test set
    model.load_weights('net.hdf5')
    X_test = get_testing_data()
    Y_test = model.predict(X_test, verbose=1)

    predictions = np.argmax(Y_test, axis=1)

    # Write to file
    with open('output.csv', 'wb') as f:
        f.write('Id,Prediction\n')
        
        for i in range(1, len(Y_test) + 1):
            f.write('%d,%d\n' % (i, predictions[i - 1]))

def alexnet(learning_rate):
    #inputs = Input((1, 128, 128))
    model = Sequential()

    model.add(Convolution2D(96, 11, 11, border_mode='same', input_shape=(128,128,1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    
    model.add(Convolution2D(256, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    
    model.add(Convolution2D(384, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(384, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))   
     
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])       

    return model


if __name__ == '__main__':

    model = alexnet(0.01) # Build alexnet w/ learning rate 0.01
    X_train, Y_train, X_valid, Y_valid = get_training_data()
    print("Printing X_train for sanity check")
    print(X_train)
    
    print("Printing Y_train for sanity check")
    print(Y_train)

    print("Printing X_valid for sanity check")
    print(X_valid)

    print("Printing Y_valid for sanity check")
    print(Y_valid)

    # Save best weights
    checkpoint = ModelCheckpoint('net.hdf5', monitor='loss', save_best_only=True)

    # Train Model
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=25, 
        validation_data=(X_valid, Y_valid),
        shuffle=True, callbacks=[checkpoint])

    # Make predictions
    make_predictions(model)
