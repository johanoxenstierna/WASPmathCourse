

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


X,Y = np.mgrid[0:2:100j, 0:2:100j]

Z = np.exp(X + Y)
xx = np.linspace(0, 2, 100)
yy = np.exp(xx)
fig = plt.figure(figsize=(10, 8))
# for i,z in enumerate( Z_vals):
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z, cmap=cm.gnuplot)
# ax.set_title('z = %.2f'%Z_vals[0], fontsize=30)
# fig.savefig('contours.png', facecolor='grey', edgecolor='none')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.show()

fff = 7

