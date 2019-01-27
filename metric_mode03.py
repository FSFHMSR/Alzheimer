import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from keras.callbacks import TensorBoard,ModelCheckpoint
from time import time
import cv2

#For collecting the data on my pc
img_width, img_height = 150, 150

train_data_dir = 'data_03/train'
validation_data_dir = 'data_03/validation'

# used to rescale the pixel values from [0, 255] to [0, 1] interval
train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.1,horizontal_flip=True)
		
validation_datagen = ImageDataGenerator(rescale=1./255)		

# automagically retrieve images and their classes for train and validation sets
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=16,
        class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')


"""
This is the simple keras CNN model, CNN models often don't need more than 3 layers when working with small datasets. The focus here is to set alot of 
filters on the layers, so the model have the possibility too find alot of patterns for the diffrent kinds of dogs and cats.
"""
model = Sequential()
model.add(Conv2D(36, 3, 3, input_shape=(img_width, img_height,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(48, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dense(36, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
#print(model.summary())
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.load_weights('model_03.h5');

#tensorboard = TensorBoard(log_dir="logs_00/{}".format(time()))

#filepath = 'model_00.h5'
#checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,save_best_only=True,save_weights_only=False, mode='min',period=1)
#callbacks_list = [checkpoint,tensorboard]

# This is for quickly being able to change the values of the diffrent training parameters
nb_epoch = 50
nb_train_samples = 5120
nb_validation_samples = 1280

#model.fit_generator(
#        train_generator,
#        samples_per_epoch=nb_train_samples,
#        nb_epoch=nb_epoch,
#        callbacks = callbacks_list,
#        validation_data=validation_generator,
#        nb_val_samples=nb_validation_samples)

alzfiles = [validation_data_dir+'/alzheimers/'+f for f in os.listdir(validation_data_dir+'/alzheimers') if f[-4:]=='.jpg']
nonalzfiles = [validation_data_dir+'/nonalzheimers/'+f for f in os.listdir(validation_data_dir+'/nonalzheimers') if f[-4:]=='.jpg']
testfiles = alzfiles + nonalzfiles
#print(testfiles)
testlabels = [0]*len(alzfiles) + [1]*len(nonalzfiles)
#print(testlabels)
pred = []
for i in range(len(testfiles)):
    img = cv2.imread(testfiles[i])
    img = img / 255.0
    img = cv2.resize(img, (150,150))
    x = np.expand_dims(img, axis=0)
    pred.append(model.predict(x, verbose=1)[0][0])
      
#print('pred',pred)
tpred = [int(round(min(1,max(p,0)))) for p in pred]
tp = sum([a[0]*a[1] for a in zip(tpred,testlabels)]) #1,1
tn = sum([(1-a[0])*(1-a[1]) for a in zip(tpred,testlabels)]) #0,0
fp = sum([a[0]*(1-a[1]) for a in zip(tpred,testlabels)]) #1,0
fn = sum([(1-a[0])*a[1] for a in zip(tpred,testlabels)]) #0,1
precision = tp/(tp+fp)
print('precision',precision)
recall = tp/(tp+fn)
print('recall',recall)
specificity = tn/(tn+fp)
print('specificity',specificity)