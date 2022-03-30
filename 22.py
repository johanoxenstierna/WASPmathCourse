
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm

X0, X1 = np.meshgrid(np.linspace(-3, 3, 20), np.linspace(-3, 3, 20))
# b)
Y = np.zeros(X0.shape, dtype=float)
aa = Y.shape[0]
for r in range(Y.shape[0]):
    for c in range(Y.shape[1]):
        Y[r, c] = np.sqrt(1 + np.sqrt(X0[r, c] + X1[r, c])**2)

aa = 5
