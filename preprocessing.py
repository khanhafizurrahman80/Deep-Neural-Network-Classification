import logging
import numpy as np
from sklearn.model_selection import train_test_split


logger = logging.getLogger('Deep_Logistic_Model')

def initial_parameters_Deep(list_of_no_of_unit_layer):
    params = {}
    np.random.seed(1)

    for l in range(1, len(list_of_no_of_unit_layer)):
        params['W' + str(l)] = np.random.randn(list_of_no_of_unit_layer[l], list_of_no_of_unit_layer[l - 1]) * 0.01
        params['b' + str(l)] = np.zeros(shape=(list_of_no_of_unit_layer[l], 1))
        logger.debug('parameters W%s shape is %s -- b%s shape is %s', str(l), params['W' + str(l)].shape, str(l), params['b' + str(l)].shape)
    return params


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

def preprocessingStep(inputData, inputLabelData):
    logger.debug('preprocessing step started where input data shape is %s and input label data shape is %s ', inputData.shape, inputLabelData.shape)
    logger.debug('first input before normalizing is \n %s', inputData[0,:])
    inputData = normalizeTheDataset(inputData)
    logger.debug('first input after normalizing is \n %s', inputData[0, :])
    splitData = splitDataset(inputData, inputLabelData)
    return splitData



def loadTheDataset(dataset):
    """
    Arg : Dataset to be modelled
    Return : a tuple containing 
        a. Input Data set (X)
        b. Target Data set (Y)
        c. no of samples (m)
        d. no of features (n)
    """
    logging.debug('load the database started!!!')
    X = dataset.data
    Y = dataset.target
    m = X.shape[0]
    n = X.shape[1]
    return (X, Y, m, n)