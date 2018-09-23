
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import pandas as pd
import io
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.python.framework import ops
from tensorflow import keras
from tensorflow.examples.tutorials.mnist import input_data


def loadTheDatasetfromlocal_repo(dataset, targetVariable):
    """

    File contain dataset from the local file repository system
    Arg : 
    dataset -- pandas dataframe
    targetVariable - name of the target variabel
    Return : a tuple containing 
        a. Input Data set (X) -- numpy ndarray
        b. Target Data set (Y) -- numpy ndarray
        c. no of samples (m)
        d. no of features (n)
    """
    X= dataset.drop(targetVariable, axis=1)
    Y= dataset[targetVariable]
    m = X.shape[0]
    n = X.shape[1]
    return (X.values, Y.values, m, n)

input_df = pd.read_csv('../Datasets/bank_datasets/bank-full.csv')
input_df.head()

input_data, input_label_data, no_of_samples, no_of_features = loadTheDatasetfromlocal_repo(input_df, "y")

def convertcategorytoNumericalData(df):
    '''
    Convert the categorical dataframe into numerical dataframe
    
    Argument:
    df -- name of the dataframe
    
    Return:
    imp_df -- a dataframe replacing the category values into numerical values
    '''
    imp_df = df

    for f in imp_df.columns:
        if imp_df[f].dtype== 'object':
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(imp_df[f].values))
            imp_df[f] = lbl.transform(list(imp_df[f].values))
            
    return imp_df

def normalizeTheDataset(inputData):
    """
        # 1. Find mean of the whole dataset
        # 2. Subtract each sample from the mean of dataset
        # 3. divide result of 2 by standard deviation
        
    :return: normalizing data
    """
    meanOfInputData = np.mean(inputData)
    subEachSampleFromMean = inputData - meanOfInputData
    divideByStdDev = np.divide(subEachSampleFromMean, np.std(inputData))
    return divideByStdDev

def splitDataset(inputData, inputLabelData):
    X_train, X_test, Y_train, Y_val = train_test_split(inputData, inputLabelData, test_size=0.2)
    return {"X_train": X_train, "X_test": X_test, "Y_train": Y_train, "Y_test": Y_val}

input_df = convertcategorytoNumericalData(df= input_df)
input_data, input_label_data, no_of_samples, no_of_features = loadTheDatasetfromlocal_repo(input_df, "y")
input_data = normalizeTheDataset(input_data)
splitData = splitDataset(input_data, input_label_data)
train_input_data = splitData["X_train"]
train_label_data = splitData["Y_train"]
test_input_data  = splitData["X_test"]
test_label_data  = splitData["Y_test"]

# Cross check of training set
print("Train input set (data) shape: {shape}".format(shape=train_input_data.shape))
print("Train label set (label) shape: {shape}".format(shape=train_label_data.shape))

# Shapes of test set
print("Test input set (data) shape: {shape}".format(shape=test_input_data.shape))
print("Test input set (label) shape: {shape}".format(shape=test_label_data.shape))

def create_one_hot_vector_array(input_array):
  output_array = np.zeros((input_array.shape[0], 2))
  output_array[np.arange(input_array.shape[0]), input_array] = 1
  return output_array

train_label_data = create_one_hot_vector_array(train_label_data)
test_label_data = create_one_hot_vector_array(test_label_data)

# Cross check of training set
print("Train input set (data) shape: {shape}".format(shape=train_input_data.shape))
print("Train label set (label) shape: {shape}".format(shape=train_label_data.shape))

# Shapes of test set
print("Test input set (data) shape: {shape}".format(shape=test_input_data.shape))
print("Test input set (label) shape: {shape}".format(shape=test_label_data.shape))

n_hidden_1 = 128
n_hidden_2 = 128
n_input = train_input_data.shape[1]
n_classes = 2
n_samples = train_input_data.shape[0]

def create_placeholders(n_x, n_y):
  '''
  Creates the placeholders for the tensorflow session.
  
  Arguments:
  n_x -- scalar, size of an image vector (28*28= 784)
  n_y -- scalar, number of classes (10)
  
  
  Returns:
  X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
  Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
  '''
  
  X = tf.placeholder(tf.float32, [n_x, None], name="X")
  Y = tf.placeholder(tf.float32, [n_y, None], name="Y")
  
  return X, Y

def initialize_parameters():
  '''
  Initialize parameters to build a neural network with tensorflow. The shapes are:
  
    W1: [n_hidden_1, n_input]
    b1: [n_hidden_1, 1]
    W2: [n_hidden_2, n_hidden_1]
    b2: [n_hidden_2, 1]
    W3: [n_classes, n_hidden_2]
    b3: [n_classes, 1]
    
  Returns:
  parameters -- a dictionalry of tensors containing W1, b1, W2, b2, W3, b3
  '''
  
  # Set random seed for reproducibility
  tf.set_random_seed(42)
  
  # Initialize weights and biases for each layer
  # First hidden layer
  
  W1 = tf.get_variable("W1", [n_hidden_1, n_input], initializer= tf.contrib.layers.xavier_initializer(seed=42))
  b1 = tf.get_variable("b1", [n_hidden_1, 1], initializer= tf.zeros_initializer())
  
  # Second hidden layer
  
  W2 = tf.get_variable("W2", [n_hidden_2, n_hidden_1], initializer= tf.contrib.layers.xavier_initializer(seed=42))
  b2 = tf.get_variable("b2", [n_hidden_2, 1], initializer= tf.zeros_initializer())
  
  # third hidden layer
  
  W3 = tf.get_variable("W3", [n_classes, n_hidden_2], initializer= tf.contrib.layers.xavier_initializer(seed=42))
  b3 = tf.get_variable("b3", [n_classes, 1], initializer= tf.zeros_initializer())
  
  parameters = {
      "W1": W1,
      "b1": b1,
      "W2": W2,
      "b2": b2,
      "W3": W3,
      "b3": b3
  }
  
  return parameters

def forward_propagation(X, parameters):
  '''
  Implements the forward propagation for the model:
  LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
  
  Arguments:
  
  X -- Input dataset placeholder, of shape (input_size, number of examples)
  parameters -- python dictionalry containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                the shapes are given in intialize_parameters
                
  Returns:
  Z3 -- the output of the last linear unit
  '''
  print("X shape: {shape}".format(shape = X.shape))
  ### Retrieve parameters from dictionary
  W1 = parameters['W1']
  b1 = parameters['b1']
  W2 = parameters['W2']
  b2 = parameters['b2']
  W3 = parameters['W3']
  b3 = parameters['b3']
  
  ### Carry out forward propagation
  Z1 = tf.add(tf.matmul(W1, X), b1)
  A1 = tf.nn.relu(Z1)
  Z2 = tf.add(tf.matmul(W2,A1), b2)
  A2 = tf.nn.relu(Z2)
  Z3 = tf.add(tf.matmul(W3,A2), b3)
  
  return Z3

def compute_cost(Z3, Y):
  '''
  Computes the cost
  
  Arguments:
  Z3 -- output of forward propagation (output of the last Linear Unit), of shape (10, number_of_examples)
  Y -- "true" labels vector placeholder, same shape as Z3
  
  Returns:
  cost - Tensor of the cost funciton
  
  # Get logits (predictions) and labels
  '''
  
  logits = tf.transpose(Z3)
  labels = tf.transpose(Y)
  
  # Compute cost
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = labels))
  
  return cost

def model(train_data, train_label_data, test_data, test_label_data, learning_rate=0.0001, num_epochs=16, minibatch_size=32, print_cost = True, graph_filename='costs'):
  
  print(train_data.shape)
  '''
  Implements a three-layer tensorflow neural network: LINEAR-> RELU -> LINEAR->RELU -> LINEAR-> SOFTMAX
  
  Arguments:
  train_image -- training images set 
  train_label -- training labels
  test -- test images set 
  test_label -- test labels
  learning_rate -- learning rate of the optimization
  num_epochs -- number of epochs of the optimization loop
  minibatch_size -- size of a minibatch
  print_cost -- True to print the cost every epoch
  
  Returns:
  paramters -- parameters learnt by the model. They can then be used to predict.
  '''
  
  # Ensure that model can be rerun without overwritting tf variables
  ops.reset_default_graph()
  # For reproducibility
  tf.set_random_seed(42)
  seed = 42
  # Get input and output shapes
  (n_x, m) = train_data.T.shape # 16, 36168
  n_y = train_label_data.T.shape[0] # 2
  
  
  costs = []
  
  # Create placeholders of shape
  X, Y = create_placeholders(n_x, n_y)
  # Initialize parameters
  parameters = initialize_parameters()
  
  # Forward propagation
  Z3 = forward_propagation(X,parameters)
  
  # Cost Funciton
  cost = compute_cost(Z3, Y)
  
  # Backpropagation (using Adam Optimizer)
  optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(cost)
  
  # Initialize variables
  init = tf.global_variables_initializer()
  
  # Start session to compute Tensorflow graph
  with tf.Session() as sess:
    
    # Run initializer
    sess.run(init)
    
    # Training loop
    for epoch in range(num_epochs):
      
      epoch_cost = 0.
      num_minibatches = int(m / minibatch_size)
      seed = seed + 1
      
      for i in range(num_minibatches):
        
        # Get next batch of training data and labels
        offset = (i * minibatch_size) % (train_data.shape[0] - minibatch_size)
        minibatch_X = train_data[offset:(offset + minibatch_size), :]
        minibatch_Y = train_label_data[offset:(offset + minibatch_size), :]
        #minibatch_X, minibatch_Y = train_images.next_batch(minibatch_size)
        
        # Execute optimizer and cost Function
        _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X.T, Y: minibatch_Y.T})
        
        # Update epoch cost
        epoch_cost += minibatch_cost /num_minibatches
        
      
      # Print the cost every epoch
      if print_cost == True:
        print("cost after epoch {epoch_num}: {cost}".format(epoch_num = epoch, cost=epoch_cost))
        costs.append(epoch_cost)
        
      # Plot costs
    plt.figure(figsize=(16,5))
    plt.plot(np.squeeze(costs), color="#2A688B")
    plt.xlim(0, num_epochs-1)
    plt.ylabel("cost")
    plt.xlabel("iterations")
    plt.title("learning rate = {rate}".format(rate=learning_rate))
    plt.show()

    # Save parameters
    parameters = sess.run(parameters)
    print("paramters have been trained!!!")

    correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

    # Calculate accuracy on test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print ("Train Accuracy:", accuracy.eval({X: train_data.T, Y: train_label_data.T}))
    print ("Test Accuracy:", accuracy.eval({X: test_data.T, Y: test_label_data.T}))

    return parameters

parameters = model(train_input_data, train_label_data, test_input_data, test_label_data, learning_rate=0.0005)
