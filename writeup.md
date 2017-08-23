#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./images/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
       
[image4]: ./images/extra/resized_right_of.png "Traffic Sign 1"
[image5]: ./images/resized_stop1.png "Traffic Sign 2"
[image6]: ./images/resized_work.png "Traffic Sign 3"
[image7]: ./images/resized_roundabout.png "Traffic Sign 4"
[image8]: ./images/resized_stop2.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  


---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

Here is a link to my [project code](https://github.com/gaber-/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32, 32, 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It shows how the data is not balanced, and some classes are overrepresented.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I created copies of the images of classes with a lower number of samples, this lead to better results even without using any agumentation techniques.

I tried adding random noise, shading the images or flipping them vertically, but that did not improve the results so I didn't use them.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

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


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an adam optimizer with cross-entropy as loss function.

I set the number of epochs to 50, as the various iterations of the model seemed to yeld stable results at around this number.

I set the batch size to 126, as higher numbers seemed to lead to worse results, while lower numbers made the process really slow.

I tried changing the learning rate but 0.001 ended up giving the best results (substantially increasing the learning rate lead to ineffective training),
while lowering the rate lead to lower accuracy on the test and accuracy sets, probably due to overfitting.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

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

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

I thought some of the images would have been difficult to classify because the sign is seen at an angle, but the model classified them correctly.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

I picked 5 random traffic sign images from pixbay. Some had to be cropped into a square pcture.

The model was able to correctly guess 5 out of 5 traffic signs, which gives an accuracy of 100% which is consistent with the validation results(95% accuracy,
there is probably going to be no error on 5 random images).

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The model is way too secure of the results, this may be due to overfitting (acheiving a training accuracy of 100%, with a validation accuracy only around 95%)

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

The network seems to pick up the shapes based on the highest color contrasts, for example it does not care if it is black on white or red on white.
As it is rather susceptible to lighting differences.

The first layer is a map of the color difference, the second layer tends to show the shapes, the third an fourth layers are harder to interpret.
