# Facial Expression to Emoji


Facial Expression to emoji which detects the expression of the person in the video being captured by the web cam. The relevant emoji to the expression of the person in the video is shown on the screen which changes with the change in the expressions. Facial expressions are important in human communication and interactions. The purpose was to develop an intelligent system for facial based expression classification using CNN algorithm and Haar classifier. The emojis used in this project include Happiness, sadness, fear, anger, disgust, neutral, surprise.

Dataset

Proper and large dataset is required for identification and conversion of facial expressions to emoji’s during the training and the testing phase. The dataset for the experiment is FER+ 2013 which is downloaded from the Kaggle which contains around 33,000 images.It  has total 33,000 images. The Train images are 28,709 images and the Test images are 3,589 images. For every emotion in the train images has around 5000 images and for every emotion in the test images has around 1000 images.. Here the emotions included are Happiness, Sadness, Angry, Disgust, Fearful, Surprise, Neutral.

The proposed CNN model

CNN architectures vary with the type of the problem at hand. The proposed model consists of three convolutional layers each followed by a maxpooling layer. The ﬁnal layer is fully connected MLP. ReLu activation function is applied to the output of every convolutional layer and fully connected layer. The ﬁrst convolutional layer ﬁlters the input image with 32 kernels of size 3x3. After max pooling is applied, the output is given as an input for the second convolutional layer with 64 kernels of size 4x4. The last convolutional layer has 128 kernels of size 1x1 followed by a fully connected layer of 512 neurons. The output of this layer is given to softmax function which produces a probability distribution of the four output class. The model is trained using adaptive moment estimation (Adam) with batch size of 100 for 1000 epochs.



![image](https://user-images.githubusercontent.com/52529370/197345203-d38f7fd7-ff9b-4037-89d0-201e5bbfca8c.png)





















Convolution Neural Network

There are four CNN algorithm steps,
Convolution: The term convolution refers to the mathematical combination of two functions toproduce a third function. It merges two sets of information. In the case of a CNN, the convolutionis performed on the input data with the use of a filter or kernel to then produce a feature map.
Here are the three elements that enter into the convolution operation:
Input image
Feature detector
Feature map



![image](https://user-images.githubusercontent.com/52529370/197345230-5d550afc-0b80-4864-a44a-a6a1793e34f5.png)
























Convolution in CNN

Max pooling: Max pooling is a sample-based discretization process. The objective is to down-sample an input representation (image, hidden-layer output matrix, etc.), reducing its dimensionality and allowing for assumptions to bemade about features contained in the sub- regions binned.


![image](https://user-images.githubusercontent.com/52529370/197345266-2cb5ad05-bdab-4f0a-b4b0-71ba928b01a9.png)
























Max Pooling in CNN

Flattening: Flattening is the process of converting all the resultant 2 dimensional arrays into asingle long continuous linear vector.



![image](https://user-images.githubusercontent.com/52529370/197345296-5000311d-b590-4ac8-8309-16957106cf85.png)

























Flattening in CNN

Full Connection: At the end of a CNN, the output of the last Pooling Layer acts as a input to theso called Fully Connected Layer. There can be one or more of these layers (“fully connected”means that every node in the first layer is  connected to  every  node  in  the  second layer).
As you see from the image below, we have three layers in the full connection step:
Input layer
Fully-connected layer
Output layer



![image](https://user-images.githubusercontent.com/52529370/197345318-fb857160-e62f-4d05-a23d-426a68c82b98.png)
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          
                                          

Full Connection in CNN

Haar Cascade Classifier
A Haar classifier, or a Haar cascade classifier, is a machine learning object detection program that identifies objects in an image and video.
The algorithm can be explained in four stages:
1.Calculating Haar Features
2.Creating Integral Images
3.Using Adaboost
4.Implementing Cascading Classifiers
This algorithm requires a lot of positive images of faces and negative images of non-faces to train the classifier, similar to other machine learning models.

Calculating Haar Features

The first step is to collect the Haar features. A Haar feature is essentially calculations that are performed on adjacent rectangular regions at a specific location in a detection window. The calculation involvessumming the pixel intensities in each region and calculatingthe differences between the sums. Here are some examples of Haar features below.



![image](https://user-images.githubusercontent.com/52529370/197345407-1294fa1d-25aa-4efa-9188-2384768b7a83.png)























Calculating Haar features

Creating Integral Images
Without going into too much of the mathematics behind it (check out the paper if you’re interested in that), integral images essentially speed up the calculation of these Haar features. Instead of computing at every pixel, it instead creates sub-rectangles and creates arrayreferences foreach of those sub-rectangles.These are then used to compute the Haar features.
Adaboost Training
Adaboost essentially chooses the best features and trains the classifiers to use them. It uses a combination of “weak classifiers” to create a “strong classifier” that the algorithm can use to detect objects.
Implementing Cascading Classifiers
The cascade classifier is made up of a series of stages, where each stage is a collection ofweak learners. Weak learners are trained using boosting, which allows for a highly accurate classifier from the mean prediction of all weak learners.The cascade classifier is made up of a series of stages, where each stage is a collection of weak learners. Weak learners are trained using boosting, which allows for a highly accurate classifier from the mean prediction of all weak learners.


![image](https://user-images.githubusercontent.com/52529370/197345420-947944cd-ad75-4383-a19f-a731dfcd3681.png)
































Object detection using Haar cascade classifier

Libraries Used

Pandas

Numpy

Tensorflow

Open cv

Matplotlib

sckit-learn

Keras

Python

Jyupiter Notebook

Anaconda


IMPLEMENTATION

Data Preparation:
In this phase initially the dataset is read which is in csv file format. The dataset contains labelled and unlabelled images. Removed all the unlabelled images.Next reshaped all the existing labelled images into same height and same width.

Model Creation:
This is the second phase in this project. In this phase sequential convolutional model is created.

Model Building:
It is the third phase in this project. Model is trained in this phase. In neural network algorithm the model is trained in a layered format, In each layer the understanding of the model increases which results in betterment of the accuracy.

Image Capturing:
In this phase the image is captured from the live video and features of the face like eyes, mouth, eye brows are detected. This is achieved using Haar Cascade Classifier.

Facial Expression to Emoji:
It is the last phase in this project. In this phase, the detected facial expression is converted to the corresponding emoji.


OUTPUT

![image](https://user-images.githubusercontent.com/52529370/197346096-903da66b-9165-4c10-8aeb-72cf8e1ea1ff.png)


























![image](https://user-images.githubusercontent.com/52529370/197346058-3a6b476d-df8e-41e9-9e30-6a217b11b721.png)































![image](https://user-images.githubusercontent.com/52529370/197346029-c85eea83-4045-4f2b-9deb-f3f1021a0664.png)






























