
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm


''' 
c) All possible combinations of two points on the x and y axis are sought
But the final plot is 2D since fx is 2D!!! Although it might be possible to show it in 3D, its easier in this case
to just compute the minimum and maximum L_norms for each x and then plot those together with fx. It would be 
counterintuitive to plot something in 3D when the function is 2D. 
'''
X = np.linspace(-3, 3, 30)
fx = np.asarray([x**3 for x in X])  # the function values
L_norm_mins = np.zeros(fx.shape, dtype=float)
L_norm_maxs = np.zeros(fx.shape, dtype=float)

for i in range(len(X)):
    x0 = X[i]
    dfx0 = 3 * x0**2
    L_norm_min = np.inf
    L_norm_max = -np.inf

    for j in range(len(X)):
        x1 = X[j]
        if x0 == x1:
            continue
        dfx1 = 3 * x1**2

        numerator = np.linalg.norm([dfx0, dfx1])
        denominator = np.linalg.norm([x0, x1])
        # denominator = np.sqrt(x0**2 - x1**2)  # DOESN"T ALWAYS WORK!!!
        L_norm = numerator / denominator
        if L_norm < L_norm_min:
            L_norm_min = L_norm
        if L_norm > L_norm_max:
            L_norm_max = L_norm

    L_norm_mins[i] = L_norm_min
    L_norm_maxs[i] = L_norm_max

assert(np.min(L_norm_mins) >= 0 and np.max(L_norm_maxs) < np.inf)

'''
b)
The plot is 3D!!!
'''
# # b)  https://math.stackexchange.com/questions/84331/does-this-derivation-on-differentiating-the-euclidean-norm-make-sense
# X0, X1 = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(-3, 3, 30))
# Y = np.zeros(X0.shape, dtype=float)  # the function we have
# L_pointwise_norms = np.zeros(X0.shape, dtype=float)
#
# for r in range(Y.shape[0]):
#     for c in range(Y.shape[1]):
#         norm_X = np.sqrt(X0[r, c]**2 + X1[r, c]**2)
#         Y[r, c] = np.sqrt(1 + norm_X**2)  # the function we have
#         dY_x0 = X0[r, c] / np.sqrt(1 + norm_X**2)
#         dY_x1 = X1[r, c] / np.sqrt(1 + norm_X**2)
#         L_pointwise_norms[r, c] = np.sqrt(dY_x0**2 + dY_x1**2) / np.sqrt(X0[r, c]**2 + X1[r, c]**2)
#
# # L_max = np.max(Y_norms)  # upper bound
# # min = np.min(Y_norms)
# assert(np.min(L_pointwise_norms) >= 0 and np.max(L_pointwise_norms) <= np.inf)  # condition for L-smooth

fig = plt.figure(figsize=(12, 9))

# # 2D =======================
ax0 = fig.add_subplot(1, 1, 1)
ax0.plot(X, fx)
ax0.plot(X, L_norm_mins)
ax0.plot(X, L_norm_maxs)

# # 3D =======================
# ax0 = fig.add_subplot(1, 1, 1, projection='3d')
# # ax1 = fig.add_subplot(1, 1, 1, projection='3d')
# # ax2 = fig.add_subplot(1, 1, 1, projection='3d')
# ax0.plot_surface(X0, X1, Y, cmap=cm.coolwarm, linewidth=0, alpha=0.7)
# ax0.plot_surface(X0, X1, L_pointwise_norms, cmap=cm.jet, linewidth=0, alpha=0.3)
# # ax0.plot_surface(X0, X1, dY_x1, cmap=cm.jet, linewidth=0, alpha=0.3)
# # ax0.plot_surface(X0, X1, Y, cmap=cm.coolwarm, linewidth=0, alpha=0.7)
# ax0.set_xlabel('X0', fontsize=10)
# ax0.set_ylabel('X1', fontsize=10)
# ax0.set_zlabel('sqrt(1 + ||x||^{2})', fontsize=10)

plt.show()
aa = 5
