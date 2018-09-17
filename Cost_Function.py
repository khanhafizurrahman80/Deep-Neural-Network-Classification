import numpy as np
import logging

logger = logging.getLogger('Deep_Logistic_Model')

def crossEntropyCost(totalNoOfSamples, afterApplyingActivationFunction, inputLabelData):
    """Arguments:
        totalNoOfSamples : no of examples 
        afterApplyingActivationFunction : value of the output of final layer
        inputLabelData: labeled data for each example provided"""
    logger.debug("start with parameters:: totalNo of sample = {}, afterApplyingActivationFunction.shape = {}, inputLabelData.shape = {}".format(totalNoOfSamples, afterApplyingActivationFunction.shape, inputLabelData.shape))
    calcCost = (-1 / totalNoOfSamples) * np.sum(np.multiply(inputLabelData, np.log(afterApplyingActivationFunction)) + np.multiply(1 - inputLabelData,np.log(1 - afterApplyingActivationFunction)))
    logger.debug('end with output:: calc cost = {}'.format(calcCost))
    return calcCost