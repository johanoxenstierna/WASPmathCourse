#
#
# # import numpy as np
# # import matplotlib.pyplot as plt
# #
# # fig, ax = plt.subplots(figsize=(9, 6))
# #
# # x, y = np.linspace(0, 10), np.linspace(0, 10)
# #
# #
# # plt.show()

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def V(x,y,z):
     # return np.cos(10*x) + np.cos(10*y) + np.cos(10*z) + 2*(x**2 + y**2 + z**2)
     return 3*x**2 * y + z**2

X,Y = np.mgrid[0:2:100j, 0:2:100j]
Z_vals = [-0.5, 0, 0.9]
num_subplots = len(Z_vals)

fig = plt.figure(figsize=(10, 8))
# for i,z in enumerate( Z_vals):
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, V(X,Y,Z_vals[2]), cmap=cm.gnuplot)
# ax.set_title('z = %.2f'%Z_vals[0], fontsize=30)
# fig.savefig('contours.png', facecolor='grey', edgecolor='none')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_ylabel("Z")

plt.show()

# from mpl_toolkits import mplot3d
# import numpy as np
# import matplotlib.pyplot as plt
# x = np.outer(np.linspace(0, 2, 30), np.ones(30))
# y = x.copy().T # transpose
# z = np.cos(x ** 2 + y ** 2)
# # 3*x**2 * y + z**2
#
# fig = plt.figure()
# ax = plt.axes(projection='3d')
#
# ax.plot_surface(x, y, z,cmap='viridis', edgecolor='none')
# ax.set_title('Surface plot')
# plt.show()