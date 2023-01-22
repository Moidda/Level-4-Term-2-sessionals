import numpy as np


class LogisticRegression:
    def __init__(self, params):
        """
        :param params: alpha, iterations
        """
        self.alpha = params["alpha"]
        self.iterations = params["iterations"]
        self.theta = np.empty((0,0))

    # sigmoid
    def sigmoid(self, z):
        return 1.0/(1 + np.exp(-z))

    # hypothesis
    def h(self, theta, X):
        return self.sigmoid(np.dot(X, theta.T))


    def partialDerivativeOfJ(self, X, y, theta):
        return np.dot((self.h(theta, X) - y).T, X)


    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        self.theta = np.matrix(np.zeros(X.shape[1])) 
        
        for i in range(self.iterations):
            self.theta = self.theta - self.alpha * self.partialDerivativeOfJ(X, y, self.theta)

        

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        prob = self.h(self.theta, X)
        prob = np.where(prob>=0.5, 1, 0)
        return prob
