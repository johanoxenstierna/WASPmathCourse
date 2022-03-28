import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random


# x = np.linspace(0, 600, 1000)
# y = 1/(np.pi * (1 + x**2))
#
# expectation = np.mean(y)
# print(expectation)

# 15 =======================================

p = np.linspace(0.01, 0.99, num=20)  # rows
num_jumps = range(1, 30)  # cols
NUM_JUMPS, P = np.meshgrid(num_jumps, p)  # rows are p, cols are num_jumps

# result = np.zeros((len(p), len(num_jumps)), dtype=float)
result_expectation = np.zeros(P.shape, dtype=float)
result_variance = np.zeros(P.shape, dtype=float)
NUM_EXPERIMENTS = 100

for r in range(len(p)):  # meshgrid could be used instead of these
    for c in range(len(num_jumps)):
        _p = p[r]
        _num_jumps = num_jumps[c]
        result_experiments = []
        for _ in range(NUM_EXPERIMENTS):

            pos_cur = 0  # position in x
            for i in range(_num_jumps):
                rand = random.random()
                if rand <= _p:
                    pos_cur -= 1
                else:
                    pos_cur += 1
            result_experiments.append(pos_cur)
        pos_expectation = np.mean(result_experiments)
        pos_variance = np.var(result_experiments)

        result_expectation[r, c] = pos_expectation
        result_variance[r, c] = pos_variance


        aa = 5

fig = plt.figure()
# fig, (ax0, ax1) = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
ax0 = fig.add_subplot(1, 2, 1, projection='3d')
ax1 = fig.add_subplot(1, 2, 2, projection='3d')
# ax0 = fig.gca(projection='3d')
# ax1 = fig.gca(projection='3d')
cmap = cm.coolwarm

ax0.plot_surface(NUM_JUMPS, P, result_expectation, cmap=cmap,
                                   linewidth=0, antialiased=False)
ax1.plot_surface(NUM_JUMPS, P, result_variance, cmap=cmap,
                                   linewidth=0, antialiased=False)

ax0.set_xlabel('NUM_JUMPS', fontsize=10)
ax0.set_ylabel('P', fontsize=10)
ax0.set_zlabel('Expectation', fontsize=10)

ax1.set_xlabel('NUM_JUMPS', fontsize=10)
ax1.set_ylabel('P', fontsize=10)
ax1.set_zlabel('Variance', fontsize=10)

plt.show()