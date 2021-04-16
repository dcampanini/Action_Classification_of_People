# Action Classification of People

This is the project of my second internship, at the Computer Vision Laboratory of the University of Chile. In this work, I wrote the code to use the system of action classification developed by Georgia Gkioxari et al. in the paper "R-CNNs for Pose Estimation and Action Detection"

# Description of the system for action detection 

The system of detection is a convolutional neural network (CNN) of 8 layers. The first 5 layers are a group of convolutional layers, then layers 6,7 and 8 are fully connected (FC). After the last layer, there is a softmax function to transform the output in probabilities. 

All the convolutional layers have a ReLU activation function, and the FC6-7 consider dropout. The next image shows the architecture of the implemented system for action detection.

![red_cnn](https://user-images.githubusercontent.com/19544865/71310342-da9c7a80-23f1-11ea-809f-84f2370b4787.png)


# Dataset
The system was training and testing in  PASCAL VOC 2012 action dataset. There are in total 10 different actions: jumping, phoning, playing instruments, reading, riding bike, riding horse, running, taking photos, using the computer and walking. The next images are a sample of the cited dataset

![clases](https://user-images.githubusercontent.com/19544865/71310242-8ba21580-23f0-11ea-97d8-df4b23d3316f.png)
