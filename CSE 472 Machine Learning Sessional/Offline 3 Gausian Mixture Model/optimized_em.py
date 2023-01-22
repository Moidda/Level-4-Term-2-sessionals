import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


class GMM:
    def __init__(self, X, k, iterations=50):
        self.X = X
        self.k = k
        self.iterations = iterations
        self.pi = np.random.rand(self.k)
        self.pi = self.pi / np.sum(self.pi)
        self.mu = np.random.rand(self.k, self.X.shape[1])
        self.sigma = np.zeros((self.k, self.X.shape[1], self.X.shape[1]))
        self.sigma[:] = np.identity(self.X.shape[1])


    def EStep(self):
        p_matrix = np.zeros((self.X.shape[0], self.k))
        for k in range(self.k):
            p_matrix[:, k] = self.pi[k] * multivariate_normal.pdf(self.X, mean=self.mu[k], cov=self.sigma[k])
        
        p_matrix = p_matrix + 0.00001
        p_matrix = p_matrix / p_matrix.sum(axis=1)[:, np.newaxis]
        return p_matrix


    def MStep(self, p_matrix):
        m = np.sum(p_matrix, axis=0)
        self.pi = m/self.X.shape[0]
        for k in range(self.k):
            self.mu[k] = (1/m[k]) * np.sum(p_matrix[:, k] * self.X.T, axis=1) 
            self.sigma[k] = (1/m[k]) * (p_matrix[:, k] * (self.X-self.mu[k]).T @ (self.X-self.mu[k]))

    
    def loglikelihood(self):
        p = np.zeros((self.X.shape[0], self.k))
        for k in range(self.k):
            p[:, k] = self.pi[k] * multivariate_normal.pdf(self.X, mean=self.mu[k], cov=self.sigma[k])
        
        return np.sum(np.log(np.sum(p, axis=1)))

    
    def EMAlgo(self):
        for i in range(self.iterations):
            p_matrix = self.EStep()
            self.MStep(p_matrix)
        
        likelihood = self.loglikelihood()
        return likelihood


    def __str__(self):
        return 'pi = \n' + str(self.pi) + '\nmu = \n' + str(self.mu) + '\nsigma=\n' + str(self.sigma)


def processFile(file_name):
    with open(file_name) as file:
        data = file.read().replace(' ', ',')
        file_name += '.csv'
        print(data, file=open(file_name, 'w'))

    return pd.read_csv(file_name).to_numpy()



# def normalDistribution(self):
    #     dim = self.X.shape[1]
    #     values = (2*np.pi)**(-dim/2) * np.linalg.det(self.sigma)**(-1/2) 
        
    #     X_mu = np.zeros((self.k, self.X.shape[0], self.X.shape[1]))
    #     X_mu_T = np.zeros((self.k, self.X.shape[1], self.X.shape[0]))
    #     for j in range(self.k):
    #         X_mu[j, :] = self.X  - self.mu[j, :]
    #         X_mu_T[j, :] = X_mu[j, :].T
 
    #     mat = X_mu @ np.linalg.inv(self.sigma) @ X_mu_T
    #     ret = np.zeros( (self.k, self.X.shape[0], ) )
    #     for j in range(self.k):
    #         ret[j, :] = np.diag(mat[j, :])

    #     return ret