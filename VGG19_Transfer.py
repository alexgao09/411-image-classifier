from __future__ import print_function

from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
import numpy as np

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K




from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.layers import Activation, BatchNormalization, Convolution2D, Dense, Dropout, Flatten, MaxPooling2D

from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

from keras.applications.vgg19 import VGG19
from keras.layers import Input




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

img_channels = 3


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
    training_data = np.zeros((N,128,128,img_channels))

    for file in os.listdir(path):
        if file.endswith(".jpg"):
            image_list.append(file)

    index = 0
    #matching_list=[]
    for i in image_list: # get all the training data - we can split this after
        training_data[index,:,:,:] = cv2.imread(path+"/"+i,1)
        #matching_list.append(i)
        index += 1
        
    training_data = training_data.astype("float32")
    training_data /= 255 ######################################################## WHY DOES IT WORK WHEN WE DIVIDE ALL OF A SUDDEN?

    full_labels=[]
    matching_list=[]
    with open('train.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            #print(row)
            full_labels.append(row[0].split(',')[1])
            matching_list.append(row[0].split(',')[0])
                   
    full_labels=full_labels[1:]
    matching_list=matching_list[1:] # contains the image number from train.csv. This list contains the correct order
    
    converted=[] 
    for k in image_list: # ordering of image_list comes from folder of pictures
        converted.append(int(k.split('.')[0]))
        
    # Now align folder with csv order (folder is unordered)
    A=map(float,converted) # unordered
    B=map(float,matching_list) # ordered
    full_labels_ordered=[] # the labels we're going to use in training
    
    print("Is the csv image ordering the same as the train folder ordering?")
    print(A==B)
    print("Error in ordering:")
    print(sum(abs(np.array(converted).astype("float32")-np.array(matching_list).astype("float32"))))
    
    for j in A: # for every element in the unordered list
        temp_index=B.index(j)
        full_labels_ordered.append(full_labels[temp_index])
        
    print("Done ordering.")

    labels = full_labels_ordered[0:N]
    labels = map(int,labels)
    # First 100 pictures with their labels:
    print("Picture label:")
    print(map(int,labels[0:100]))
    print("Picture:")
    print(map(int,A[0:100]))
    np.savetxt("check_order_labels.csv", np.concatenate((np.array(A).reshape(-1,1), np.array(labels).reshape(-1,1)),axis=1),delimiter=',')
    labels = np.array(labels,dtype=np.uint8).reshape(-1,1) # The dtype is VERY important
    labels -= 1 # THIS IS VERY IMPORTANT


    X_train = training_data[0:6000,:,:,:]
    Y_train = labels[0:6000,:]

    X_valid = training_data[6000:7000,:,:,:] # For some reason, we need to do up to 6998 - 7000 doesn't work
    Y_valid = labels[6000:7000,:]

    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_valid = np_utils.to_categorical(Y_valid, nb_classes)
    print("Done data pre-processing")

    return X_train, Y_train , X_valid, Y_valid

def get_testing_data():

    # First 970 public test data

    path = os.path.join(cwd, 'val')
    image_list=[]
    testing_data = np.zeros((M,128,128,img_channels))

    for file in os.listdir(path):
        if file.endswith(".jpg"):
            image_list.append(file)

    index = 0
    ubuntu_ordering_test=[] # this list will contain the ordered list of the 970 test images on the ubuntu server
    for i in image_list:
        testing_data[index,:,:,:] = cv2.imread(path+"/"+i,1)
        ubuntu_ordering_test.append(int(i.split('.')[0]))
        index += 1
        
    testing_data = testing_data.astype("float32")
    testing_data /= 255 ######################################################## WHY DOES IT WORK WHEN WE DIVIDE ALL OF A SUDDEN?

    X_test = testing_data[0:970,:,:,:]

    print("Test Images Ordering:")
    print(ubuntu_ordering_test)

    # Last 2000 public test data ###########################################################
    path_final = os.path.join(cwd, 'test_128')
    image_list_final=[]
    testing_data_final = np.zeros((2000,128,128,img_channels))

    for file in os.listdir(path_final):
        if file.endswith(".jpg"):
            image_list_final.append(file)

    index_final = 0
    ubuntu_ordering_test_final=[] # this list will contain the ordered list of the 970 test images on the ubuntu server
    for i in image_list_final:
        testing_data_final[index_final,:,:,:] = cv2.imread(path_final+"/"+i,1)
        ubuntu_ordering_test_final.append(int(i.split('.')[0]))
        index_final += 1

    testing_data_final = testing_data_final.astype("float32")
    testing_data_final /= 255 ######################################################## WHY DOES IT WORK WHEN WE DIVIDE ALL OF A SUDDEN?

    X_test_final = testing_data_final[0:2000,:,:,:]

    print("Test Images Ordering of last 2000:")
    print(ubuntu_ordering_test_final)


    return X_test, ubuntu_ordering_test, X_test_final, ubuntu_ordering_test_final


def make_predictions(model):
    
    # predict on test set
    model.load_weights('vgg19net.hdf5')
    X_test, X_test_ordering, X_test_final, X_test_ordering_final  = get_testing_data()
    # print(X_test)
    Y_test = model.predict(X_test, verbose=1) # first 970
    Y_test_final=model.predict(X_test_final, verbose=1) # last 2000
    
    #for i in range(Y_test.shape[0]):
    #    print(X_test_ordering[i], Y_test[i])

    predictions = np.argmax(Y_test, axis=1)
    predictions_final=np.argmax(Y_test_final, axis=1)

    # Write to file in the order of the images 1-970
    with open('vgg19_output.csv', 'wb') as f:
        f.write('Id,Prediction\n')
        
        for i in range(1, len(Y_test) + 1): # for image number i
            #f.write('%d,%d\n' % (X_test_ordering[i-1], predictions[i - 1]))
            temp_index=X_test_ordering.index(i)
            f.write('%d,%d\n' % (i, predictions[temp_index]))

        for i in range(971, len(Y_test_final) + 971): # for image number i
            #f.write('%d,%d\n' % (X_test_ordering[i-1], predictions[i - 1]))
            temp_index_final=X_test_ordering_final.index(i-971)
            f.write('%d,%d\n' % (i, predictions_final[temp_index_final]))        





base_model = VGG19(include_top=False, weights='imagenet', input_tensor=Input((128,128,3)))
for layer in base_model.layers:
    layer.trainable = False
    
x = base_model.output
x = Flatten(name='flatten')(x)
x = Dense(256, activation='relu', name='fc1')(x)
x = Dense(8, activation='softmax', name='predictions')(x)

#x.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
model = Model(input=base_model.input, output=x)
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])
#model.load_weights('vgg19net.hdf5') # load previous weights


X_train, Y_train, X_valid, Y_valid = get_training_data()


datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(X_train)

checkpoint = ModelCheckpoint('vgg19net.hdf5', monitor='loss', save_best_only=True)

model.fit_generator(datagen.flow(X_train, Y_train, batch_size=32), samples_per_epoch=len(X_train), callbacks=[checkpoint] ,nb_epoch=150, validation_data=(X_valid, Y_valid))


    # Make predictions
make_predictions(model)


