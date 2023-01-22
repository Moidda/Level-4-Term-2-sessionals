import numpy as np
import pandas as pd
import sys
np.set_printoptions(threshold=sys.maxsize)


def processFile(file_name):
    with open(file_name) as file:
        data = file.read().replace(' ', ',')
        file_name += '.csv'
        print(data, file=open(file_name, 'w'))

    return pd.read_csv(file_name)


def initParams(n_clusters, n_data, n_attributes):
    pi = np.random.rand(n_clusters)
    pi = pi / np.sum(pi)
    
    mu = np.random.rand(n_clusters, n_attributes)  
    
    sigma = np.zeros((n_clusters, n_attributes, n_attributes), dtype=float)
    for i in range(n_clusters):
        for j in range(n_attributes):
            sigma[i][j][j] = 1.0

    return pi, mu, sigma


def normalDist(X, mu, sigma, dim):
    X_new = np.atleast_2d(X)
    mu_new = np.atleast_2d(mu)
    sigma_new = np.atleast_2d(sigma)
    sigma_inv = np.linalg.inv(sigma_new)
    return (2*np.pi)**(-dim/2) * np.linalg.det(sigma_new)**(-1/2) * np.exp(-0.5 * (X_new-mu_new) @ sigma_inv @ (X_new-mu_new).T)


def reNormalize(vec):
    newvec = vec + 0.00001
    newvec = newvec / np.sum(newvec)
    return newvec


def logLikelihood(X, pi, mu, sigma, n_clusters):
    p = 0
    for i in range(X.shape[0]):
        sum = 0
        for k in range(n_clusters):
            sum += pi[k]*normalDist(X[i], mu[k], sigma[k], X.shape[1])
        p += np.log(sum)
    return p


def EStep(X, pi, mu, sigma, n_clusters):
    """
    Returns (n,k) matrix denoting the probability
    that ith datapoint belongs to jth cluster
    """
    p_matrix = np.zeros((X.shape[0], n_clusters))
    for i in range(X.shape[0]):
        for k in range(n_clusters):
            p_matrix[i][k] = pi[k]*normalDist(X[i], mu[k], sigma[k], X.shape[1])
        
        p_matrix[i] = reNormalize(p_matrix[i])

    return p_matrix


def MStep(X, p_matrix, n_clusters):
    n_attributes = X.shape[1]
    pi = np.zeros( (n_clusters,) )
    mu = np.zeros( (n_clusters, n_attributes) )
    sigma = np.zeros( (n_clusters, n_attributes, n_attributes) )
    m = np.sum(p_matrix, axis=0)

    pi = m/X.shape[0]

    for k in range(n_clusters):
        bostu = np.zeros( (n_attributes,) )
        for i in range(X.shape[0]):
            bostu += p_matrix[i][k]*X[i]

        mu[k] = bostu / m[k]

    for k in range(n_clusters):
        bostu = np.zeros( (n_attributes, n_attributes) )
        for i in range(X.shape[0]):
            bostu += p_matrix[i][k] * mult(X[i], mu[k])
        
        sigma[k] = bostu / m[k]
        assert not np.isclose(np.linalg.det(sigma[k]), 0), str(sigma) + "\n" + str(np.linalg.det(sigma[k]))

    return pi, mu, sigma


def mult(x, mu):
    x_new = np.atleast_2d(x)
    mu_new = np.atleast_2d(mu)
    return (x_new - mu_new).T @ (x_new - mu_new)


def printFile(ob):
    open("debug.txt", 'w').close()
    print(str(ob), file=open("debug.txt", 'a'))


def getSummary(pi, mu, sigma):
    msg = "pi = \n" + str(pi) + "\n"
    msg += "mu = \n" + str(mu) + "\n"
    msg += "sigma = \n" + str(sigma) + "\n"
    return msg


def GMM(X, n_clusters, iterations = 50):
    pi, mu, sigma = initParams(n_clusters, X.shape[0], n_attributes)
    likelihood = 0
    for i in range(iterations):
        p_matrix = EStep(X, pi, mu, sigma, n_clusters)
        pi, mu, sigma = MStep(X, p_matrix, n_clusters)
        likelihood = logLikelihood(X, pi, mu, sigma, n_clusters)

    return likelihood


if __name__ == "__main__":
    file_name = 'data2D.txt' # TODO: take this as input
    # file_name = input("File Name: ")

    df = processFile(file_name)
    n_data, n_attributes = len(df), len(df.columns)
    X = df.to_numpy()
    print("k = " + str(3) + ", likelihood = " + str(GMM(X, 3)))
    # for n_clusters in range(1, 11):
    #     print("k = " + str(n_clusters) + ", likelihood = " + str(GMM(X, n_clusters)))
