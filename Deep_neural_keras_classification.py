
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tensorflow import keras
from google.colab import files
print(tf.__version__)


input_df = pd.read_csv('../Datasets/bank_datasets/bank-full.csv')
input_df.head()

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

# Dense layer means fully connected
model = keras.Sequential([
    keras.layers.Dense(train_input_data.shape[1], kernel_initializer= keras.initializers.glorot_normal(seed=42), activation= tf.nn.relu),
    keras.layers.Dense(128, kernel_initializer= keras.initializers.glorot_normal(seed=42), activation= tf.nn.relu), 
    keras.layers.Dense(128, kernel_initializer= keras.initializers.glorot_normal(seed=42), activation= tf.nn.relu),
    keras.layers.Dense(2, activation= tf.nn.softmax) # output layer
])

model.compile(optimizer=tf.train.AdamOptimizer(), loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(train_input_data, train_label_data, epochs=16)

test_loss, test_acc = model.evaluate(test_input_data, test_label_data)
print('Test accuracy:', test_acc)

predictions = model.predict(test_input_data)
