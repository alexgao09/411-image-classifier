from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils

from keras.models import Model, Sequential
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, AveragePooling2D, BatchNormalization, UpSampling2D, Dropout
from keras.optimizers import Adam, SGD

from keras.layers.normalization import BatchNormalization

batch_size = 32
nb_classes = 8
nb_epoch = 200
data_augmentation = True

# input image dimensions
img_rows, img_cols = 128, 128
# the CIFAR10 images are RGB
img_channels = 1


import numpy as np
import csv
import os
import cv2

N=7000 # The number of jpg images in our current working directory
training_data=np.zeros((N,128,128,1)) # The second element is the number of channels


cwd = os.getcwd()
image_list=[]

for file in os.listdir(cwd):
    if file.endswith(".jpg"):
        image_list.append(file)

index=0
for i in image_list: # get all the training data - we can split this after
    training_data[index,:,:,0]=cv2.imread(i,0)
    index=index+1
    
training_data=training_data.astype("float32")
training_data/=255 ######################################################## WHY DOES IT WORK WHEN WE DIVIDE ALL OF A SUDDEN?

full_labels=[]
with open('train.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in reader:
        full_labels.append(row[0].split(',')[1])
        
full_labels=full_labels[2:]

labels=full_labels[0:N]
labels=map(int,labels)
labels=np.array(labels,dtype=np.uint8).reshape(-1,1) # The dtype is VERY important
labels=labels-1 # THIS IS VERY IMPORTANT


X_train=training_data[0:6000,:,:,:]
y_train=labels[0:6000,:]

X_test=training_data[6000:6998,:,:,:] # For some reason, we need to do up to 6998 - 7000 doesn't work
y_test=labels[6000:6998,:]

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print("Done data pre-processing")


def alexnet(learning_rate):
    #inputs = Input((1, 128, 128))
    model = Sequential()

    model.add(Convolution2D(96, 11, 11, border_mode='same', input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    
    model.add(Convolution2D(256, 5, 5, border_mode='same', input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    
    model.add(Convolution2D(384, 3, 3, border_mode='same', input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(384, 3, 3, border_mode='same', input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))

    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))   
     
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=X_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])       

    return model

M=alexnet(0.01) # Build alexnet w/ learning rate 0.01


M.fit(X_train, Y_train, batch_size=batch_size,
    nb_epoch=nb_epoch, validation_data=(X_test, Y_test),
            shuffle=True)
