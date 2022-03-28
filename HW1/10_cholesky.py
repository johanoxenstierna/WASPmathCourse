import numpy as np

A = np.array([[1, 1, 3],
              [1, 20, 26],
              [3, 26, 70]])

answer = np.linalg.cholesky(A)
# [1.00000, 0.00000, 0.00000],
# [1.00000, 4.35890, 0.00000],
# [3.00000, 5.27656, 5.75829]

from fractions import Fraction
answer_exact = []
for i in range(3):
    inner = []
    for j in range(3):
        inner.append(Fraction(answer[i, j]))
    answer_exact.append(inner)
# aad = Fraction(aa[1, 1])

hh = 6
