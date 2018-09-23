# Deep Neural Network for Logistic Regression from Scratch

A complete cycle of deep learning method is developed without involvement of any framework. The project is done for the learning purpose while doing the course: https://www.coursera.org/learn/neural-networks-deep-learning. Many of the idea is taken from their assignments.

Prerequisite:
* Python 2.7 or Python 3.6
* numpy
* scikit-learn(for dataset)

There are total 7 numbers of file are present. Each name of the file is self explanatory about describing the purpose of each module.

# Tensorflow Implementation 

Name of the file: Deep_neural_tensorFlow_classification.py
source: https://github.com/khanhafizurrahman80/Deep-Neural-Network-Classification/blob/master/Deep_neural_tensorFlow_classification.py

# Keras Implementation 

Name of the file: Deep_neural_keras_classification.py
source: https://github.com/khanhafizurrahman80/Deep-Neural-Network-Classification/blob/master/Deep_neural_keras_classification.py

In both tensorflow and Keras following are the description of hyperparameters:


No.of layer: 3

weight initialization: Xavier

Activation Function: Except output layer, in all layers the activation function is RELU where softmax is used in the final layer

Cost Function: Cross entropy

Optimization Algorithm: Adam

## LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
