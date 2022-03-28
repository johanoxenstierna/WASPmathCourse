

import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 8))

# def f()

X = np.linspace(0, 2 * np.pi, 50)
Y0 = np.cos(X)
Y1 = np.sin(X)
Y2 = X
# Xagg = np.zeros((X.shape[0], 3))
# Xagg[:, 0] = Y0
# Xagg[:, 1] = Y1
# Xagg[:, 2] = Y2

# OBS FIX: REPLACE 0 with x. Could also rewrite LHS.
fjoo = np.zeros((X.shape[0], 3))
for i, x in enumerate(X):
    LHS = (np.cos(x) - np.cos(0), np.sin(x) - np.sin(0), x - 0)
    RHS = (-np.sin(x), np.cos(x), x)
    fjoo[i, 0] = LHS[0] - RHS[0]
    fjoo[i, 1] = LHS[1] - RHS[1]
    fjoo[i, 2] = LHS[2] - RHS[2]

aa = 5

# f1 = (np.cos(1), np.sin(1), 1)
# f0 = (np.cos(0), np.sin(0), 0)
fjo = (np.cos(1) - np.cos(0), np.sin(1) - np.sin(0), 1 - 0)
fder1 = (-np.sin(1), np.cos(1), 1)
# CONCLUSION: It's a 3 arg function and all 3 args in fjoo are never 0 at the same time

fx = np.zeros((X.shape[0]))

axY0, = ax.plot(X, Y0, '--', color='red')
axY1, = ax.plot(X, Y1, '--', color='blue')
axY2, = ax.plot(X, Y2, '--', color='green')
axf0, = ax.plot(X, fjoo[:, 0], '-', color='red')
axf1, = ax.plot(X, fjoo[:, 1], '-', color='blue')
axf2, = ax.plot(X, fjoo[:, 2], '-', color='green')

ax.legend([axY0, axY1, axY2, axf0, axf1, axf2], ["sin(X)", "cos(X)", "X", "diff_for_sin", "diff_for_cos", "diff_for_x"])
ax.set_xlabel("X")
ax.set_ylabel("Y")

plt.show()


