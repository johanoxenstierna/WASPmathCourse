# https://matplotlib.org/3.1.0/gallery/mplot3d/surface3d.html

# This import registers the 3D projection, but is otherwise unused.
# from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('X', fontsize=20)
ax.set_ylabel('Y')

# Make data.
X_MIN, X_MAX = 0.9, 1.1   # 27
Y_MIN, Y_MAX = 0.9, 1.1

# X_MIN, X_MAX = -7, 7   # 29
# Y_MIN, Y_MAX = -7, 7

X = np.arange(X_MIN, X_MAX, 0.01)
Y = np.arange(Y_MIN, Y_MAX, 0.01)
X, Y = np.meshgrid(X, Y)
# R = np.sqrt(X**2 + Y**2)
# Z = np.sin(R)
Z = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):

        Z[i, j] = np.sin(np.pi * X[i, j] * Y[i, j]**2)  # 27
        # Z[i, j] = np.sin(X[i, j]**2 + Y[i, j]) + Y[i, j]
# Z = 1 / (X + Y) + X + Y


# Plot the surface.
cmap = cm.coolwarm
surf = ax.plot_surface(X, Y, Z, cmap=cmap,
                       linewidth=0, antialiased=False)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

