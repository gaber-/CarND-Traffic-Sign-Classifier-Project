# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
       
[image4]: ./images/extra/resized_right_of.png "Traffic Sign 1"
[image5]: ./images/extra/resized_stop1.png "Traffic Sign 2"
[image6]: ./images/extra/resized_work.png "Traffic Sign 3"
[image7]: ./images/extra/resized_roundabout.png "Traffic Sign 4"
[image8]: ./images/extra/resized_stop2.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  


---
### Writeup / README

Here is a link to my [project code](https://github.com/gaber-/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32, 32, 3
* The number of unique classes/labels in the data set is 43

Here is an exploratory visualization of the data set. It shows how the data is not balanced, and some classes are overrepresented.

![alt text][image1]

### Design and Test a Model Architecture

I created copies of the images of classes with a lower number of samples, this lead to better results even without using any agumentation techniques.

I tried adding random noise, shading the images or flipping them vertically, but that did not improve the results so I didn't use them.

#### 2. Model

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 7x7     	| 1x1 stride, valid padding, outputs 26x26x6 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 10x10x12 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride,  outputs 8x8x24 			    	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride,  outputs 3x3x48   				|
| RELU					|												|
| Fully connected		| 432, 250   									|
| Fully connected		| 250, 150    									|
| Fully connected		| 150, 43      									|
 
the softmax is applied to the output afterwards


#### 3. Training

To train the model, I used an adam optimizer with cross-entropy as loss function.

I set the number of epochs to 50, as the various iterations of the model seemed to yeld stable results at around this number.

I set the batch size to 126, as higher numbers seemed to lead to worse results, while lower numbers made the process really slow.

I tried changing the learning rate but 0.001 ended up giving the best results (substantially increasing the learning rate lead to ineffective training),
while lowering the rate lead to lower accuracy on the test and accuracy sets, probably due to overfitting.

#### 4. Method

My final model results were:
* training set accuracy of 1
* validation set accuracy of 0.949
* test set accuracy of 0.932

I took LeNet as a base model because it is a light network that solves a similar image recognition task.
I replaced the pooling layers with convolutional layers, which seemed to perform better.
I then tried adding some layers, without seeing much improvement.
The longest part was finding the right kernel sizes and depth:

* I started with changing the initial kernel sizes, an setting these of the new layers (replacing the pooling layers)
* then I set the depth
* I adjusted again the kernels
* and finally I changed the depth again, checking the feature map and results of the tests, trying to obtain the best result with the lowest depth

The final results show that the model does overfit (the training accuracy is 100%, while the validation accuracy is around 95%), but the performance is still acceptable.

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

I thought some of the images would have been difficult to classify because the sign is seen at an angle, but the model classified them correctly.

I picked 5 random traffic sign images from pixbay. Some had to be cropped into a square pcture.

#### Accuracy and confidence

The model was able to correctly guess 5 out of 5 traffic signs, which gives an accuracy of 100% which is consistent with the validation results(95% accuracy,
there is probably going to be no error on 5 random images).

The model is way too secure of the results, this may be due to overfitting (acheiving a training accuracy of 100%, with a validation accuracy only around 95%)

#### Visualization

The network seems to pick up the shapes based on the highest color contrasts, for example it does not care if it is black on white or red on white.
As it is rather susceptible to lighting differences.

The first layer is a map of the color difference, the second layer tends to show the shapes, the third an fourth layers are harder to interpret.
