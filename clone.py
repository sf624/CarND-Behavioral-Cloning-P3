import csv
import cv2
import numpy as np

lines = []
with open("./data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

print("There are " + str(len(lines)) + " data points.")

images = []
measurements = []
for line in lines:
    # [0] ... path to center_image
    # [1] ... path to left_image
    # [2] ... path to right_image
    # [3] ... steering angle [-1,1]
    # [4] ... throttle [0,1]
    # [5] ... break (0)
    # [6] ... speed [0,30]

    # load images to list
    for i in range(3):
        source_path = line[i]
        image = cv2.imread(source_path)
        images.append(image)

    # load measurement to list
    correction = 0.2
    measurement = float(line[3])
    measurements.extend([measurement, measurement + correction, measurement - correction])

# augment images and steering
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append((-1.0)*measurement)

# convert to numpy array to be able to appply to Keras
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

print("There are " + str(len(X_train)) + " training data. (befor split)")

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Construct regression DNN (note: not classification network! )
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100,activation="tanh"))
model.add(Dense(50,activation="tanh"))
model.add(Dense(10,activation="tanh"))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")
history_object = model.fit(X_train,y_train,validation_split=0.2,shuffle=True,epochs=2,verbose=1)

model.save("model.h5")

'''
# Visualize history of mse loss
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch

import matplotlib.pyplot as plt
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
'''

exit()
