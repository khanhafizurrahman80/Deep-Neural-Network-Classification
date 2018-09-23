import logging
import preprocessing as pre
import CommonFunction as com_F
from sklearn.datasets import load_breast_cancer
import Predict as prd
import pandas as pd


def main():
    logger.info('main method started!!!')
    input_df = pd.read_csv("../Datasets/bank_datasets/bank-full.csv")
    input_df = pre.convertcategorytoNumericalData(df= input_df)
    input_data, input_label_data, no_of_samples, no_of_features = pre.loadTheDatasetfromlocal_repo(dataset=input_df, targetVariable= "y")
    # input_data, input_label_data, no_of_samples, no_of_features = pre.loadTheDataset(dataset=load_breast_cancer())
    print (type(input_data))
    print (type(input_label_data))
    input_label_data = input_label_data.reshape(-1,1)
    print (input_data.shape)
    logger.info('input_data shape is %s ;\t input_label_data shape is %s; \t no of sample is %s; no of feature is %s', input_data.shape, input_label_data.shape, no_of_samples, no_of_features)
    print (input_label_data.shape)
    print(no_of_samples)
    print(no_of_features)
    splitData = pre.preprocessingStep(inputData= input_data, inputLabelData= input_label_data)
    training_input_data = splitData["X_train"].T
    training_label_data = splitData["Y_train"].T
    test_input_data = splitData["X_test"].T
    test_label_data = splitData["Y_test"].T
    no_of_samples = training_input_data.shape[1]
    no_of_test_samples = test_input_data.shape[1]
    logger.debug('training input data is %s training label data is %s test input data is %s test label data is %s no_of_sample is %s',
               training_input_data.shape, training_label_data.shape, test_input_data.shape, test_label_data.shape,no_of_samples)
    print("Train input set (data) shape: {shape}".format(shape=training_input_data.shape))
    print("Train label set (label) shape: {shape}".format(shape=training_label_data.shape))
    print("Test input set (data) shape: {shape}".format(shape=test_input_data.shape))
    print("Test label set (label) shape: {shape}".format(shape=test_label_data.shape))
    print(no_of_samples, no_of_test_samples)

    parameters = pre.initial_parameters_Deep(list_of_no_of_unit_layer=[no_of_features,5,1])
    logger.debug('length of parameters is %s', len(parameters))
    final_params, cost_in_per_batch = com_F.batchGradientDescent(inputData=training_input_data,labelData= training_label_data, parameters= parameters,  num_iteration= 16, batch_size=32,no_of_samples=no_of_samples)
    com_F.plotCostFunction(costs= cost_in_per_batch, learning_rate= 0.05)

    prediction_result = prd.predict(total_no_of_samples = no_of_test_samples, parameters = final_params, test_data = test_input_data)
    accuracy = prd.predict_accuracy(prediction_result = prediction_result, test_label_data = test_label_data)
    logger.info('prediction accuracy is {}'. format(accuracy))
    print('prediction accuracy is {accuracy}'. format(accuracy= accuracy))
    
if __name__ == '__main__':
    logger = logging.getLogger('Deep_Logistic_Model')
    logger.setLevel('DEBUG')

    filehandler_dbg = logging.FileHandler(logger.name + '-debug.log', mode='w')
    filehandler_dbg.setLevel('DEBUG')

    streamformatter = logging.Formatter(fmt='%(asctime)s-\t%(filename)s-\t%(funcName)s-\t\t%(message)s', datefmt='%H:%M:%S')

    filehandler_dbg.setFormatter(streamformatter)

    logger.addHandler(filehandler_dbg)
    main()