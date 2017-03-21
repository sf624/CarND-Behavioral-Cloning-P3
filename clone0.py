import csv
import cv2
import numpy as np

lines = []
with open("./sample_training_data/data/driving_log.csv") as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []

print(len(lines[1:]))

for line in lines[1:]:
    # [0] ... path to center_image
    # [1] ... path to left_image
    # [2] ... path to right_image
    # [3] ... steering angle [-1,1]
    # [4] ... throttle [0,1]
    # [5] ... break (0)
    # [6] ... speed [0,30]

    # load images to list
    source_path = "./sample_training_data/data/" + line[0]
    image = cv2.imread(source_path)
    images.append(image)

    # load measurement to list
    measurement = float(line[3])
    measurements.append(measurement)

# convert to numpy array to be able to appply to Keras
X_train = np.array(images)
y_train = np.array(measurements)

print(X_train.shape)
print(y_train.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense

# Construct regression DNN (note: not classification network! )
model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")
model.fit(X_train,y_train,validation_split=0.2,shuffle=True,nb_epoch=5)

model.save("model.h5")
