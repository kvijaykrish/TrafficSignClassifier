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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/img1.png "Traffic Sign 1"
[image5]: ./examples/img2.png "Traffic Sign 2"
[image6]: ./examples/img3.png "Traffic Sign 3"
[image7]: ./examples/img4.png "Traffic Sign 4"
[image8]: ./examples/img5.png "Traffic Sign 5"
[image9]: ./examples/img6.png "Traffic Sign 6"
[image10]: ./examples/img7.png "Traffic Sign 7"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/kvijaykrish/TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of original training set is 34799
* The size of validation set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because of two reasons:
1. To reduce the number of channels 32X32x3 to 32x32x1 so that the number of paramters to be trained are less 
2. The colour of the image is not significant feature that could be used for classification.

Here is an example of a traffic sign image after grayscaling.

![alt text][image2]

-> As a last step, I normalized the image data because the optimizer used in training can converge faster on a normalized data.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by ...
-> My final training set had 4 times 34000 number of images + 34799 original preprocessed dataset.
-> My validation set had 12630 number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because:
1. The inital train data set has more images in one class and few images in another class
2. Hence the training could be biased to the classes with more images.

To add more data to the the data set, I used the following techniques: 
1. I added more images in the class which had few original images. 
2. This is done by adding modified images by appling affine transformation (rotation, shear, translation) on exisitng images 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following:
1. The inital training data set had 34799 images of size 32x32x3
2. The agumented data set consists of 4 sets of 34000 images of size 32x32x1 + 1 set of preprocessed original image set of 34799 images of size 32x32x1
3. Each of the augumeted images set has 1/4th of the preprocessed original image set + remaing images are augumeted images

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 preprocessed grayscale normalized image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| 3 Fully connected layer output 43 logits        									|
| Softmax				| Softmax and Cross Entropy is applied        									|
|	Loss Optimize					|	Adam optimizer											|



####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used:
1. LeNet Architecture
2. Impelemneted a slow learning rate of 0.0005
3. Adam optimizer
4. Batch size of 128
5. 6 times 8 Epoch of 4 sets of augumented dataset
6. Finally 1 times 8 Epoch of 1 sets of original preprocessed dataset

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of 0.937
* test set accuracy of 0.911

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
-> Initially the LeNet architecture was used with original data set of color images.
* What were some problems with the initial architecture?
-> This had a very low validation accuracy and it did not increase
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
-> The architecute was slightly modified for the channels. 3 channels were reduced to 1 channel to decrease the number of parameters that has to be trained and also for normalized grayscale values
* Which parameters were tuned? How were they adjusted and why?
-> The learaning rate was decreased. 
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
-> The same LeNet Architecure was choosen
* Why did you believe it would be relevant to the traffic sign application?
-> The architecure was good for chanracter recognition, since traffic sign are similar, I thought it would be relevant
-> The accuracy was low initialy, but with more training data it improved well.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
-> With additonal agumetation of image, the test accuracy reached alost 100%. The validation accuracy slowly increased and reached more than 93% The test accuracy was 91.1%. 
-> Some possible improvemnts: Training accuracy saturated and hence the validation accuracy could not improve more. This could be due to overfitting. There is scope to add some dropouts to improve validation accuracy.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]![alt text][image9]
![alt text][image10]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (70km/h)      		| Speed limit (70km/h)   									| 
| No entry     			| No entry 										|
| Stop					| Stop											|
| Ahead only      		| Ahead only					 				|
| Yield			| Yield      							|
| Speed limit (60km/h)			| Speed limit (60km/h)      							|
| Turn left ahead			| Turn left ahead      							|


The model was able to correctly guess 7 of the 7 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Speed limit (70km/h) sign (probability of 0.99), and the image does contain a Speed limit (70km/h) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.99987721e-01         			| Speed limit (70km/h)   									| 
| 1.21809599e-05     				| Speed limit (30km/h) 										|
| 9.96598715e-08					| Speed limit (20km/h)											|
| 1.98718625e-10	      			| Speed limit (120km/h)					 				|
| 3.85846910e-11				    | Speed limit (50km/h)      							|

