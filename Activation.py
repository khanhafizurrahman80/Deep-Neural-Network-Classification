import numpy as np
import logging

logger = logging.getLogger('Deep_Logistic_Model')

def sigmoid(data):
    afterApplyingSigmoidFunction = 1 / (1 + np.exp(-data))
    return afterApplyingSigmoidFunction

def derSigmoid(data):
    derivative = np.multiply(data, 1 - data)
    return derivative

def relu(data):
    return np.maximum(data, 0)

def derRelu(data):
    data[data <= 0] = 0
    data[data > 0] = 1
    return data

def derivativeOfOutputWithRespectToFinalLayerL(y, AL):
    logger.debug('started parameters:: \ny.shape = {} \t AL.shape={} \nY= {} \nAL= {}'.format(y.shape, AL.shape, y[0,:],AL[0,:]))
    dAL = np.divide(1 - y, 1 - AL) - np.divide(y, AL)
    logger.debug('end output:: \ndAL.shape={} \ndAL= {}'.format(dAL.shape, dAL[0, :]))
    return dAL

def derivativeOfOutputWithRespectToSigmoid( dAL, AL):
    logger.debug('started parameters:: \nshapes::: dAL.shape = {}, AL.shape = {} \nvalues::: dAL = {} \t AL = {}'.format(dAL.shape, AL.shape, dAL[0, :], AL[0, :]))
    dZ = np.multiply(dAL, np.multiply(AL, (1 - AL)))
    logger.debug('return output:: \nshapes::: dZ.shape = {} \nvalues::: dZ = {} '.format(dZ.shape, dZ[0, :]))
    return dZ

def derivativeOfOutputWithRespectToRelu( dAL, AL):
    logger.debug('started parameters:: \nshapes::: dAL.shape = {}, AL.shape = {} \nvalues::: dAL = {} \t AL = {}'.format(dAL.shape, AL.shape, dAL[0,:], AL[0,:]))
    derReluData = derRelu(AL)
    logger.debug('variable defined after implementing derRelu:: \nshapes::: derReluData.shape = {} \nvalues::: derReluData = {} '.format(derReluData.shape, derReluData[0, :]))
    dZ = np.multiply(dAL, derReluData)
    logger.debug('return output:: \nshapes::: dZ.shape = {} \nvalues::: dZ = {} '.format(dZ.shape, dZ[0, :]))
    return dZ