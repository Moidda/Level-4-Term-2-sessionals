from data_handler import bagging_sampler
import copy
import numpy as np


class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator
        self.estimators = [None]*n_estimator


    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2

        for i in range(self.n_estimator):
            X_sample, y_sample = bagging_sampler(X, y)
            self.base_estimator.fit(X_sample, y_sample)
            self.estimators[i] = copy.deepcopy(self.base_estimator)


    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        y_pred = self.estimators[0].predict(X)
        predictions = y_pred
        for i in range(1, self.n_estimator):
            y_pred = self.estimators[i].predict(X)
            predictions = np.append(predictions, y_pred, axis=1)

        summaryPredictions = np.empty(y_pred.shape, dtype=int)
        for i in range(predictions.shape[0]):
            row = predictions[i]
            countOne = np.count_nonzero(row)
            countZero = row.shape[0] - countOne
            if countZero > countOne:
                row = np.array([0])
            else:
                row = np.array([1])

            summaryPredictions[i] = row

        return summaryPredictions
