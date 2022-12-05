import numpy as np


n = int(input())
m = int(input())

A = np.random.randint(1, 100, (n, m))

U, s, V_T = np.linalg.svd(A, full_matrices=False)
D = np.diag(s)

# this is to check that the singular value decomposition
# works correctly
# ok = np.allclose(A, np.dot(U, np.dot(D, V_t)))

D_pinv = np.reciprocal(D, where=np.isclose(D, 0)==False)
D_pinv = np.array(D_pinv).T

V = np.array(V_T).T
U_T = np.array(U).T

A_pinv_calc = np.dot(V, np.dot(D_pinv, U_T))

ok = np.allclose(np.linalg.pinv(A), A_pinv_calc)