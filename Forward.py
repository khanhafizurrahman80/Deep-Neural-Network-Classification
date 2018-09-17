import  numpy as np
import Activation as act
import logging

logger = logging.getLogger('Deep_Logistic_Model')
def linear_Forward(A, W, b):
    logger.debug('start with parameters:: A.shape is {}, W.shape is {}, b.shape is {}'.format(A.shape, W.shape, b.shape))
    z = np.dot(W, A) + b
    cache = (A, W, b)
    logger.debug('end with output:: z.shape is {}, \ninside caches:: A.shape is {}, W.shape is {}, b.shape is {}'.format(z.shape, cache[0].shape, cache[1].shape, cache[2].shape))
    return z, cache

def linear_activation_function(A_prev, W, b, activation):
    logger.debug('start with parameters: A_prev.shape is {} , W.shape is {}, b.shape is {} & activation function is {}'.format(A_prev.shape, W.shape, b.shape, activation))

    if activation == "sigmoid":
        z, linear_cache = linear_Forward(A=A_prev, W=W, b=b)
        A = act.sigmoid(z)
        activation_cache = A

    elif activation == "relu":
        z, linear_cache = linear_Forward(A=A_prev, W=W, b=b)
        A = act.relu(z)
        activation_cache = A

    cache = (linear_cache, activation_cache)

    logger.debug('end with output: after apply active function A.shape is {} , \ninside caches(linear cache):: A.shape is {}, W.shape is {}, b.shape is {} \ninsidecaches(activation cache):: {}'.
          format(A.shape,cache[0][0].shape, cache[0][1].shape, cache[0][2].shape,cache[1].shape))
    return A, cache


def deep_model_linear_activaiton_function(inputs, parameters):
    logger.debug('start with parameters:: input shape %s parameters shape %s', inputs.shape, len(parameters))
    caches = []
    A = inputs
    input_shape = A.shape
    L = int(len(parameters) /2)
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_function(A_prev, parameters['W' + str(l)], parameters['b' + str(l)],
                                              activation='relu')
        caches.append(cache)

    AL, cache = linear_activation_function(A, parameters['W' + str(L)], parameters['b' + str(L)], activation='sigmoid')
    caches.append(cache)
    logger.debug('end with output:: AL.shape is {}, len of caches is {}'.format(AL.shape, len(caches)))
    return AL, caches
