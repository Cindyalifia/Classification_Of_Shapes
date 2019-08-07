from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Initialising the CNN
classifier = Sequential()
classifier.add(Convolution2D(
    16, 3, 3, input_shape=(28, 28, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(56, activation='relu', kernel_initializer='uniform'))
classifier.add(Dense(3, activation='softmax', kernel_initializer='uniform'))

# Compiling the CNN
classifier.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Image Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('training_set',
                                                 target_size=(28, 28),
                                                 batch_size=1,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('test_set',
                                            target_size=(28, 28),
                                            batch_size=1,
                                            class_mode='categorical')

history = classifier.fit_generator(training_set,
                                   steps_per_epoch=len(training_set.filenames),
                                   epochs=20,
                                   validation_data=test_set,
                                   validation_steps=len(test_set.filenames))
classifier.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
print(training_set.class_indices)

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

