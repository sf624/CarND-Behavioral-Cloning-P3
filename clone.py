import csv
import cv2
import numpy as np

lines = []
with open("./data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

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

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

# Construct regression DNN (note: not classification network! )
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((50,25),(0,0))))
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5,activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,epochs=2)

model.save("model.h5")
exit()
