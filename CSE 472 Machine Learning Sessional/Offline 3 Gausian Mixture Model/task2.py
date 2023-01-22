import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from optimized_em import processFile


plt.ion()


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

            x,y = np.meshgrid(np.sort(self.X[:,0]), np.sort(self.X[:,1]))
            self.XY = np.array([x.flatten(), y.flatten()]).T

            plt.clf()
            plt.scatter(self.X[:, 0], self.X[:, 1], s=1)
            plt.title('Iteration ' + str(i))

            for k in range(self.k):
                mvn = multivariate_normal.pdf(self.XY, mean=self.mu[k], cov=self.sigma[k])
                plt.contour(np.sort(X[:, 0]), np.sort(X[:, 1]), mvn.reshape(len(X), len(X)), colors = 'black', alpha = 0.3)
            
            plt.pause(0.1)
        
        plt.show()

        likelihood = self.loglikelihood()
        return likelihood


    def __str__(self):
        return 'pi = \n' + str(self.pi) + '\nmu = \n' + str(self.mu) + '\nsigma=\n' + str(self.sigma)



if __name__ == "__main__":
    file_name = 'data2D.txt' # TODO: take this as input
    # file_name = input("File Name: ")
    X = processFile(file_name)

    plot_x = X[:, 0]
    plot_y = X[:, 1]

    gmm = GMM(X, 3, 50)
    gmm.EMAlgo()