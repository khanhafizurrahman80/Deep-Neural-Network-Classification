import numpy as np
import Activation as act
import logging

logger = logging.getLogger('Deep_Logistic_Model')

def linear_backward(dZ, cache):
    """ Implement the linear portion of backward propagation for a single layer (layer 1)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    logger.debug('started parameters:: \nShapes::: dZ.shape = {} \t cache = {} \nValues::: dZ={}  '.format(dZ.shape, len(cache), dZ[0,:]))
    A_prev, W, b = cache
    m = A_prev.shape[1]
    logger.debug('variable defined:: \nShapes::: A_prev.shape = {} \t W = {} \t b = {} \nValues::: A_prev={} \t W={} \t b={}  \t m={}'.format
                 (A_prev.shape, W.shape, b.shape, A_prev[0,:], W[0,:], b[0,:], m))
    dW = np.divide(np.dot(dZ, A_prev.T), m)
    db = np.divide(np.sum(dZ, axis=1, keepdims=True), m)
    dA_prev = np.dot(W.T, dZ)
    logger.debug('end output:: \nShapes::: dA_prev.shape = {} \t dW.shape = {} \t db.shape = {} \nValues::: dA_prev={} \t dW={} \t db={} '.format
                 (dA_prev.shape, dW.shape, db.shape, dA_prev[0,:], dW[0,:], db[0,:]))
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    logger.debug('started parameters:: dA.shape = {} \t length of cache = {} \t activation = {} \n dA={}'.format(dA.shape, len(cache), activation, dA[0,:]))
    linear_cache, activation_cache = cache
    logger.debug('variable defined:: linear cache :: \nShapes::::A.shape ={} \tW.shape={} \tb.shape={} \nValues::::A={} \t W={} \t b={} \nactivation cache shape={} \t activation_cache={}'.
                 format(linear_cache[0].shape,linear_cache[1].shape,linear_cache[2].shape,linear_cache[0][0,:],linear_cache[1][0,:],linear_cache[2][0,:], activation_cache.shape,activation_cache))
    if activation == "relu":
        dZ = act.derivativeOfOutputWithRespectToRelu(dA, activation_cache)
    elif activation == "sigmoid":
        dZ = act.derivativeOfOutputWithRespectToSigmoid(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    logger.debug('ended output:: dA_prev.shape = {} \t dW.shape = {} \t db.shape = {}'.format(dA_prev.shape,dW.shape,db.shape))
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    logger.debug('started with parameters:: \nAL.shape = {}\t Y.shape ={}\t caches.length = {}'.format(AL.shape, Y.shape, len(caches)))
    grads = {}
    L = len(caches)
    m = Y.shape[1]
    Y = Y.reshape(AL.shape)
    logger.debug('variable defined:: \nL = {}\t m ={}\t Y.shape = {}'.format(L, m, Y.shape))

    dAL = act.derivativeOfOutputWithRespectToFinalLayerL(Y, AL)
    current_cache = caches[-1]
    dA_prev, dw, db = linear_activation_backward(dAL, current_cache, "sigmoid")
    grads["dA" + str(L)] = dA_prev
    grads["dW" + str(L)] = dw
    grads["db" + str(L)] = db
    logger.debug('variable defined::: \nShape:::: grads["dA" + {}] = {} \t grads["dW" + {}] = {} \t grads["db" + {}] = {}\nValues:::: grads["dA" + {}] = {} \t grads["dW" + {}] = {} \t grads["db" + {}] = {}'.format
                 (str(L),grads["dA" + str(L)].shape, str(L),grads["dW" + str(L)].shape, str(L),grads["db" + str(L)].shape,str(L),grads["dA" + str(L)][0,:], str(L),grads["dW" + str(L)][0,:], str(L),grads["db" + str(L)][0,:]))
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev, dw, db = linear_activation_backward(dA_prev, current_cache, "relu")
        dAL = dA_prev
        grads["dA" + str(l + 1)] = dA_prev
        grads["dW" + str(l + 1)] = dw
        grads["db" + str(l + 1)] = db
        logger.debug('variable defined::: \nShape:::: grads["dA" + {}] = {} \t grads["dW" + {}] = {} \t grads["db" + {}] = {}\nValues:::: grads["dA" + {}] = {} \t grads["dW" + {}] = {} \t grads["db" + {}] = {}'.format
                     (str(l + 1), grads["dA" + str(l + 1)].shape, str(l + 1), grads["dW" + str(l + 1)].shape, str(l + 1), grads["db" + str(l + 1)].shape,str(l + 1), grads["dA" + str(l + 1)][0, :], str(l + 1), grads["dW" + str(l + 1)][0, :], str(l + 1),grads["db" + str(l + 1)][0, :]))
    return grads