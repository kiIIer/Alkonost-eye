import cv2

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Dense

input_shape=(450, 450, 3)
image_gen = ImageDataGenerator()
image_gen.flow_from_directory('./alkonost-eye-dataset/train')

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3,3), input_shape=input_shape, activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters = 64, kernel_size = (3,3), input_shape=input_shape, activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters = 64, kernel_size = (3,3), input_shape=input_shape, activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics = ['accuracy'])

batch_size = 16
train_image_gen = image_gen.flow_from_directory('./alkonost-eye-dataset/train/', 
                                                target_size = input_shape[:2],
                                                batch_size = batch_size,
                                               class_mode = 'binary')

test_image_gen = image_gen.flow_from_directory('./alkonost-eye-dataset/test/', 
                                                target_size = input_shape[:2],
                                               batch_size = batch_size,
                                               class_mode = 'binary')

results = model.fit_generator(train_image_gen, epochs = 5, steps_per_epoch = 24,
                             validation_data=test_image_gen, validation_steps = 12)
