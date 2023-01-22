import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

from optimized_em import GMM, processFile


if __name__ == "__main__":
    file_name = 'data2D.txt' # TODO: take this as input
    # file_name = input("File Name: ")
    X = processFile(file_name)

    plot_x = []
    plot_y = []

    for k in range(1, 11):
        gmm = GMM(X, k)
        loglikelihood = gmm.EMAlgo()
        plot_x.append(k)
        plot_y.append(loglikelihood)
        print(str(k) + ': ' + str(loglikelihood))

    plt.plot(plot_x, plot_y)
    plt.xlabel('k - no of clusters')
    plt.ylabel('log likelihood')
    plt.title('log likelihood vs no of clusters')
    plt.show()
