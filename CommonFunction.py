import preprocessing as prepro
import Forward as forward
import Cost_Function as cost
import Backward as back
import logging
from matplotlib import pyplot as plt
import numpy as np

logger = logging.getLogger('Deep_Logistic_Model')
def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """

    L = int(len(parameters) /2 )
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters

def batchGradientDescent(inputData, labelData, parameters, num_iteration, batch_size, no_of_samples):
    logger.debug('batchGradientDescent is started with parameters:: \ninputData shape: %s \nlabelData: %s\nparameters length: %s\nnum_iteration %s\nbatch_size: %s\nno_of_samples: %s',
                 inputData.shape, labelData.shape, len(parameters), num_iteration, batch_size, no_of_samples)
    cost_store_based_on_iteration =[]
    for i in range(0, num_iteration):
        logger.debug('iteration no {}'.format(i))
        starting_counter = 0
        ending_counter = starting_counter + 1
        start_index = starting_counter * batch_size
        end_index = ending_counter * batch_size
        cost_store_based_on_batch = []
        while(start_index< end_index):
            logger.debug('starting_counter is %s, ending_counter is %s, start_index is %s, end_index is %s', starting_counter, ending_counter, start_index, end_index)
            # forward propagation
            a, caches = forward.deep_model_linear_activaiton_function(inputs=inputData[:,start_index:end_index],
                                                              parameters=parameters)
            # compute cost
            entropy_cost = cost.crossEntropyCost(totalNoOfSamples=batch_size, afterApplyingActivationFunction=a,
                                        inputLabelData=labelData[:,start_index:end_index])
            cost_store_based_on_batch.append(entropy_cost);
            # backward propagation
            logger.debug("========================= backward propagation started!!!!!!!!! ===================")
            grads = back.L_model_backward(AL = a, Y = labelData[:,start_index:end_index], caches= caches)
            # update parameters
            parameters = update_parameters(parameters= parameters, grads= grads, learning_rate= 0.05)
            starting_counter = starting_counter + 1
            ending_counter = ending_counter + 1
            start_index = starting_counter * batch_size
            if start_index >= no_of_samples:
                break
            end_index = ending_counter * batch_size
            if end_index >= no_of_samples:
                end_index = no_of_samples - 1
        cost_store_based_on_iteration.append(sum(cost_store_based_on_batch))

    for x in range(len(cost_store_based_on_iteration)):
        logger.debug('total cost after iteration {} is {}'.format(x,cost_store_based_on_iteration[x]))

    return parameters,cost_store_based_on_iteration

def plotCostFunction(costs, learning_rate):
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations no')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()