from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, BatchNormalization
from keras.optimizers import SGD
from keras.applications.vgg16 import VGG16
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler
from keras.regularizers import l2
from keras.optimizers import Adam

learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy',
                                            patience=4,
                                            factor=0.5,
                                            min_lr = 0.001,
                                            verbose = 1)

early_stoping = EarlyStopping(monitor='val_loss',patience= 3,restore_best_weights=True,verbose=0)

input_shape=(320, 320, 1)
reshaped_shape = (1, 320, 320, 1)
l2_rate = 0.01


def schedule(epoch, lr):
    if epoch <= 10:
        return 0.001
    else:
        return 0.0001

lr_schedule = LearningRateScheduler(schedule)
adam = Adam(learning_rate=0.01)


model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(l2_rate), input_shape=input_shape))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(l2_rate)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))


model.add(Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(l2_rate)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

# model.add(Conv2D(512, (3, 3), activation='relu', kernel_regularizer=l2(l2_rate)))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

batch_size = 16
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    './Dataset4/train',
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary',
    color_mode='grayscale'
)

test_generator = test_datagen.flow_from_directory(
    './Dataset4/test',
    target_size = input_shape[:2],
    batch_size = batch_size,
    class_mode='binary',
    color_mode='grayscale'
)

results = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator,
    callbacks = [lr_schedule, early_stoping, learning_rate_reduction]
)

model_save_path = './alkonost-eye-v0.5.3.keras'
model.save(model_save_path)