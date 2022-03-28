
import numpy as np

A = np.array([[0.6, 0.9],
              [0.1, 0.6]])
x = np.array([3, 1])

answer = np.linalg.matrix_power(A, 10) @ x


V = np.array([[3, 3],
              [1, -1]])
V_inv = np.linalg.inv(V)

lamb = np.array([[0.9, 0],
                [0, 0.3]])

lam_sq = np.linalg.matrix_power(lamb, 10)

A = V @ lam_sq @ np.linalg.inv(V) @ x

ad = 5
