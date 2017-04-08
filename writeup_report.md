# **Behavioral Cloning** 

## Writeup Template

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)


[image_center]: ./center_2017_03_21_18_27_47_248.jpg "Center Image"
[image_left]: ./left_2017_03_21_18_27_47_248.jpg "Left Image"
[image_right]: ./right_2017_03_21_18_27_47_248.jpg "Right Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* clone.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* run1.mp4 for visualizing the result of autonomous driving using trained network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```
$ python drive.py model.h5 
```

#### 3. Submission code is usable and readable

The clone.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of 5 convolution neural networks with 5x5 and 3x3 filter sizes and depths between 24 and 64 (clone.py lines 55-68), followed by 4 fully connected layer to produce steering angle, which is single float number. 

The model includes ReLU and tanh layers to introduce nonlinearity (code line 58-62 and 65-67), and the data is normalized in the model using a Keras lambda layer (code line 18). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 64). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 71). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 70). The mean squared error of steering angle was defined as a cost function to be minimize.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a dataset of center lane driving. In addition to center camera imges, left and right camera images were also used as a dataset to help the car recover to the center lane. Those steering angle were increased by 0.2 degrees from the original data.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the one proposed by M. Bojraski, et al. The model was claimed to have controled the car automatically in real environment, so it seemed to be a good starting point for this project.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. To combat the overfitting, I have tuned the number of epochs and implemeted dropout layer in the Keras architecture.

The final step was to run the simulator to see how well the car was driving around track one.The vehicle was able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 52-64) consisted of a convolution neural network with the following layers and layer sizes 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 320x160x3 rgb image   							| 
| Normalization | |
| Cropping | output: 320x65x3 normalized rgb image |
| Convolution 5x5     	| 2x2 stride, valid padding, 24 features |
| ReLU					|		(Activation)										|
| Convolution 5x5	    | 2x2 stride, valid padding, 36 features									|
| ReLU					|		(Activation)										|
| Convolution 5x5	    | 2x2 stride, valid padding, 48 features     									|
| ReLU					|		(Activation)										|
| Convolution 3x3	    | 1x1 stride, valid padding, 64 features  									|
| ReLU					|		(Activation)										|
| Convolution 3x3	    | 1x1 stride, valid padding, 64 features     									|
| ReLU					|		(Activation)										|
| Flatten		|        									|
| Dropout | Keep probability: 0.5 |
| Fully connected		| output 100        									|
| tanh					|		(Activation)										|
| Fully connected		| output 50        									|
| tanh					|		(Activation)										|
| Fully connected		| output 10        									|
| tanh					|		(Activation)										|
| Fully connected		| output 1 = steering angle       									|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image_center]

Also the left and right camera images were captured in order to train the network to recover the car to the center lane. The corresponding steering angles were artificially created by modifying the angle of original steering angle. Following are the example image of left and right mounted camera images.

![alt text][image_left]
![alt text][image_right]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would increased the number of training data and helps the network to generalize better.

After the collection process, I had initially 5308 number of data points (combination of center, left and right camera images and steering angle). I associated the center camera images and original steering angle. As for the left and right camera images, the related steering angle was generated by modifying the original steering angle by adding up 0.2 degrees. Also the flipped images and steering angles were added to increase training size. As a result, there were 31848 data points which could be used as training and validation.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was around 1 ~ 3 as evidenced by visualizing the mse error in each epochs. So 2 epochs was used for the actual coding. I used an adam optimizer so that manually training the learning rate wasn't necessary.

