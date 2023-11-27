import cv2

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Dense
from keras.applications import Xception

input_shape=(640, 640, 3)
reshaped_shape = (1, 640, 640, 3)

base_model = Xception(
    include_top=False,
    weights='imagenet',
    input_shape=input_shape
)

for layer in base_model.layers:
    layer.trainable = False

model = Sequential()
model.add(base_model)
print(base_model.output_shape)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics = ['accuracy'])

batch_size = 16
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    './dataset-with-blur/train',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary',
    # color_mode='grayscale'
)

test_generator = test_datagen.flow_from_directory(
    './dataset-with-blur/test',
    target_size = input_shape[:2],
    batch_size = batch_size,
    class_mode='binary',
    # color_mode='grayscale'
)

results = model.fit(
    train_generator,
    steps_per_epoch=10,
    epochs=5,
    validation_data=test_generator,
    validation_steps=1
)