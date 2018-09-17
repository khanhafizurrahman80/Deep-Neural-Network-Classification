import logging
import numpy as np
import Forward as fwd

logger = logging.getLogger('Deep_Logistic_Model')
def predict(total_no_of_samples, parameters, test_data):
    prediction_result = np.zeros((1,total_no_of_samples))
    feed_forward_result = fwd.deep_model_linear_activaiton_function(inputs= test_data, parameters= parameters)
    feed_forward_prediction = feed_forward_result[0]
    for i in range(prediction_result.shape[1]):
        prediction_result[0,i] = 1 if feed_forward_prediction[0,i] >= 0.5 else 0
    return  prediction_result

def predict_accuracy(prediction_result, test_label_data):
    return (100 - np.mean(np.abs(prediction_result-test_label_data)) * 100)