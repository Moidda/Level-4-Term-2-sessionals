# Take n as input
# Create a random symmetric square matrix of integers of dimension n
# Perform eigen decomposition of the matrix
# Reconstruct the matrix from its eigen decomposition
# Check if reconstruction is correct


import numpy as np


n = input()
n = int(n)

# creating a diagonally dominant matrix
# a square matrix is said to be diagonally dominant 
# if, for every row of the matrix, the magnitude of the diagonal entry in a row 
# is larger than or equal to the sum of the magnitudes of all the other 
# (non-diagonal) entries in that row
# a diagonally dominant matrix is invertible
A = np.random.randint(1, 100, (n, n))
for i in range(n):
    for j in range(n):
        if i>=j:
            continue
        A[j][i] = A[i][j]
        
rowSums = np.sum(np.abs(A), axis=1)
np.fill_diagonal(A, rowSums)

det = np.linalg.det(A)
if det == 0:
    print("Non invertible")
    exit(0)

eigen_values, eigen_vectors = np.linalg.eig(A)

# A = V*diag(lambda)*V^{-1}
# V = matrix obtained by concatenating all the eigenvectors, 
#     one vector per column
# diag(lambda) = diagonal matrix made from the eigenvalues as the diagonal

D = np.diag(eigen_values)
V = np.array(eigen_vectors)
V_inv = np.linalg.inv(V)
A_calc =  np.matmul(V, np.matmul(D, V.T))
# A_calc =  np.matmul(V, np.matmul(D, V_inv))

ok = np.allclose(A, A_calc)

