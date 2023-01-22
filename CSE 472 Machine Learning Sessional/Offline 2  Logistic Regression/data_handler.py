import pandas as pd
import numpy as np
import random


def load_dataset(csvFileName):
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    Last column of csv file is taken as the vector of class
    :return: (features, classes) 
    """
    data = pd.read_csv(csvFileName)
    features = data.iloc[:, :-1]
    classes = data.iloc[:, -1:]
    return features.to_numpy(), classes.to_numpy()


def split_dataset(X, y, test_size=0.3, shuffle=False):
    """
    function for spliting dataset into train and test
    :param X: 
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    numofData = X.shape[0]
    mask = np.zeros( (numofData,), dtype=bool)
    mask[-int(round(test_size*numofData)) : ] = True
    if shuffle:
        np.random.shuffle(mask)

    X_train, y_train, X_test, y_test = X[~mask], y[~mask], X[mask], y[mask]

    # appending a column of 1s before each matrices
    X_train = np.append(np.ones( (X_train.shape[0], 1) ), X_train, axis=1)
    X_test = np.append(np.ones( (X_test.shape[0], 1) ), X_test, axis=1)
    
    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    numOfData = X.shape[0]
    indices = np.arange(0, numOfData)
    chosenIndices = random.choices(indices, k=numOfData)
    X_sample, y_sample = X[chosenIndices], y[chosenIndices]

    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    return X_sample, y_sample
