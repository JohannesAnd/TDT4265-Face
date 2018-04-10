import keras
import numpy as np
from keras.applications import VGG16, VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense


base_model = keras.applications.vgg16.VGG16(
    include_top=False, weights='imagenet', input_shape=(150, 150, 3))
#base_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(150, 150,3))
#base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(150, 150,3))

for layer in base_model.layers:
    layer.trainable = False
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
model.summary()

# Example on how to read an image
#test_image = plt.imread('./dataset/test600/cat/cat.1401.jpg')
# print('Shape: {}, max value: {}, min value: {}'.format(test_image.shape,
#                                                      np.amax(test_image),
#                                                      np.amin(test_image)))

datagen = ImageDataGenerator(
    rescale=1./255)


train_generator = datagen.flow_from_directory(
    './dataset/train1000',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary')

validation_generator = datagen.flow_from_directory(
    './dataset/validation400',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary')

test_generator = datagen.flow_from_directory(
    './dataset/test600',
    target_size=(150, 150),
    batch_size=16,
    class_mode='binary')


model.fit_generator(
    train_generator,
    steps_per_epoch=2000 // 16,
    epochs=10,
    validation_data=test_generator,
    validation_steps=400 // 16)

score = model.evaluate_generator(test_generator, 600//16)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
