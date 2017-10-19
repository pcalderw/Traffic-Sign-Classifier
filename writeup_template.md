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

[image1]: ./images/class_averages.png "Average image per class"
[image2]: ./images/class_bars.png "Number of examples/class"
[image3]: ./images/extras_classified.png "Test Data, green/red for correctly/incorrectly classified"
[image4]: ./images/extras_color.png "Test Data Unmodified"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration
There are 34799 training examples spread out over 43 classes. There is a large spread for the class sizes. We should home to see about 810 per class however, the spread is quite poor for some of them.

![alt text][image2]

See for example "Dangerous Curve to the left" which has the least at 180 examples, or "Speed limit (50km/h)" which has the most at 2010 examples. This means we will probably need to augment these smaller sets to generate more training data. During testing it would also be ideal to test it against either more examples or more complicated examples of these less seen signs.

The statistics were calculated using simple numpy functions.

In order to get a better understanding of each classes approximate image I calculated the average image for each class. Remarkable, they appear to be quite similar to each other. Each sign is near perfectly centered and shown face on. The averages were done after normalizing and grayscale for simplicity of calculation.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Besides nromalization, the easiest augmentation to make was to use grayscale images. As can be seen in the above "Averages" image. The data is quite distinguishable regardless of the loss of color. This cuts the dimensional possibilities quite a lot without losing prceious information. Ideally the structure of the signs is learned instead of the color because it's less susceptible to shadow and illumination problems.

I also increased the training set size by adding small per pixel noise to classes with low number of examples. This beefed up the training set to at least 500/class, or a total of 55017 examples.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x12 					|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x24 	|
| Max pooling	    	| 2x2 stride, outputs 5x5x24 					|
| Flatten				| outputs 600x1									|
| Dropout				| 50% dropout									|
| Fully connected		| outputs 400x1									|
| Dropout				| 50% dropout									|
| Fully connected		| outputs 120x1									|
| Fully connected		| outputs 84x1									|
| Fully connected		| outputs 43x1									|
| Softmax				| 												|
|						|												|
 Each layer adds a Relu as it's activation unless it's not an actual change to the structure just shape. Like Flatten and the dropouts


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used the 'tf.train.AdamOptimizer' with a bath size of 128. This was run off a Google Cloud instance with a single Tesla GPU. Training time wasn't signifigantly improved toying with batch size so I felt confident it is close to optimal.

The single largest variable hyperparamters were the two kernel sizes, which I ended up settling somewhere in the midddle of my first choice 6, 16 (from LeNet) and 24, 36 as my largest. The larger ones often failed to converge and require a large learning rate to start which often eventually caused training to diverge and get nowhere near as optimal as the LeNet architecture.

The other parameter with a drastic impact was learning rate. The final value of 0.08 is experimentally chosen. It nearly always resulted in a good convergence and didn't diverge near the end of training.

The final model was chosen as the one with the maximum seen validation accuracy, which tended to be around 30-40 epochs. By around 50-60 it would result in a dramatic loss of performance on both training and validation sets.

## TODO
####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of: 97.8%
* validation set accuracy of: 94.7%
* test set accuracy of: 91.8%

The first model was a generalized version of LeNet which seemed like a good fit as the data is quite similar to hand digits. The grayscale seem to have much similar internal class structure than digits do so I was certain a network of it's complexity would be acceptable. LeNet' achieved at its peak with a training accuacy of 98%, which is about as accurate as the most overfitted final model architecture's results. However this is only after playing with dataset augmentation. The very first was LeNet with 3 color channels as input and incorrectly computed normalization, which was using int division instead of the appropriate float which was difficult to detect until the more complicated models. After applying grayscale and achieving 90% on training I increased the channel sizes of the conv layers to see if it could generalize better. This increase in complexity resulted in the model only outputting 0, obviously something was wrong.

There were two issues made apparent by this.
1. The normalization was fixed.
2. The learning rate was too low for the complexity and it struggled to actually improve. I addressed this by fiddling with this parameter more frequently when a model seemed to be stuck with poor performance. 

Once these issues were resolved the larger channelled LeNet architecture hit 90% on validation. From there I added some the two dropout layers as regularization. Two dropouts ended up working better than just one. This met the minimum of 93% on validation. The dataset augmentation for small classes was just to try to boost it further but I suspect the number of examples was also seen in the training set and it did not add much in terms of performance.

The extra dense layer was added only to allow the last layer not have a dropout just before the logits.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I pulled these images off Google map's street view in the city of Frankfurt. 

![alt text][[image4]

I provided these instead of the typical search results because they provided better examples of real world data.
They have additional obstacles to overcome when compared to the training data.

1. They are not strictly centered.
2. They are skewed by nature of being taken from a moving car instead of face on.
3. They show extra information.
	* "No entry" is poorly cropped and cuts off the top
	* "Vehicles over 3.5 ..." has a second sign below it.

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


