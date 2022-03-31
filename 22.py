
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm

# Test in 2D
# X = np.linspace(-3, 3, 50)
# X_norm = np.sqrt(np.sum([x**2 for x in X]))
# Y = np.sqrt(1 + X_norm)
# aa = 6

X0, X1 = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(-3, 3, 30))  #  this gives 100 datapoints

# b)  https://math.stackexchange.com/questions/84331/does-this-derivation-on-differentiating-the-euclidean-norm-make-sense
# OBS there are 100 data points, each with x0 and x1
Y = np.zeros(X0.shape, dtype=float)
# dY_x0 = np.zeros(X0.shape, dtype=float)
# dY_x1 = np.zeros(X0.shape, dtype=float)
Y_norms = np.zeros(X0.shape, dtype=float)

for r in range(Y.shape[0]):
    for c in range(Y.shape[1]):
        norm_X = np.sqrt(X0[r, c]**2 + X1[r, c]**2)
        Y[r, c] = np.sqrt(1 + norm_X**2)
        dY_x0 = X0[r, c] / np.sqrt(1 + norm_X**2)
        dY_x1 = X1[r, c] / np.sqrt(1 + norm_X**2)
        Y_norms[r, c] = np.sqrt(dY_x0**2 + dY_x1**2) / np.sqrt(X0[r, c]**2 + X1[r, c]**2)

# L_max = np.max(Y_norms)  # upper bound
# min = np.min(Y_norms)
assert(np.min(Y_norms) >= 0 and np.max(Y_norms) <= np.inf)  # condition for L-smooth

fig = plt.figure(figsize=(12, 9))

# 3D =======================
ax0 = fig.add_subplot(1, 1, 1, projection='3d')
# ax1 = fig.add_subplot(1, 1, 1, projection='3d')
# ax2 = fig.add_subplot(1, 1, 1, projection='3d')
ax0.plot_surface(X0, X1, Y, cmap=cm.coolwarm, linewidth=0, alpha=0.7)
ax0.plot_surface(X0, X1, Y_norms, cmap=cm.jet, linewidth=0, alpha=0.3)
# ax0.plot_surface(X0, X1, dY_x1, cmap=cm.jet, linewidth=0, alpha=0.3)
# ax0.plot_surface(X0, X1, Y, cmap=cm.coolwarm, linewidth=0, alpha=0.7)
ax0.set_xlabel('X0', fontsize=10)
ax0.set_ylabel('X1', fontsize=10)
ax0.set_zlabel('sqrt(1 + ||x||^{2})', fontsize=10)

plt.show()
aa = 5
