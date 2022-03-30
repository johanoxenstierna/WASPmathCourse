

import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm

def gen_data(num_rows, num_cols, type):
    """
    num_rows: m  (data samples)
    num_cols: n  (dimensions)
    """

    X = None
    if type == 'simple':
        # X = np.random.uniform(0, 5, (num_rows, num_cols))  # each row is a vector
        # x = np.linspace([0, 0], [5, 5], num=num_rows)  # each row is a vector
        b = 0  # bias
        X0, X1 = np.meshgrid(np.linspace(0, 5, num_rows), np.linspace(0, 5, num_rows))  # each row is a vector
        Y = np.zeros(X0.shape, dtype=float)
        for r in range(Y.shape[0]):
            for c in range(Y.shape[1]):
                b = random.random() * 2
                x0 = X0[r, c] + b
                x1 = X1[r, c] + b
                Y[r, c] = np.log(np.sum(np.exp([x0, x1])))

        return (X0, X1), Y
    # else:


def loss_gradient(W, X, Y):
    pointwise_losses = np.zeros(Y.shape, dtype=float)
    # for w in W:
    for r in range(Y.shape[0]):
        for c in range(Y.shape[1]):
            x0 = X[0][r, c]
            x1 = X[1][r, c]
            x = np.array([x0, x1])

            pointwise_losses[r, c] = W.dot(x) - Y[r, c]
            # pointwise_losses[r, c] = 2 * (w * x0 + w * x1 - y)
    dL = (1/len(pointwise_losses)) * np.sum(pointwise_losses)  # MSE
    # dL = np.sum(pointwise_losses)  # SE
    return pointwise_losses, dL


def fit_model(W, X):
    # w = 2
    Y_hat = np.zeros(X[0].shape, dtype=float)
    for r in range(Y.shape[0]):
        for c in range(Y.shape[1]):
            x0 = X[0][r, c]
            x1 = X[1][r, c]
            x = np.array([x0, x1])
            Y_hat[r, c] = W.dot(x)
    return Y_hat


def perform_gradient_descent(W, X, Y, eta, num_iters):
    dL_hist = np.zeros((num_iters,))
    for i in range(1, num_iters):
        pointwise_losses, dL = loss_gradient(W[i - 1, :], X, Y)
        dL_hist[i - 1] = dL
        W[i, :] = W[i - 1, :] - eta*dL
    return dL_hist


NUM_ROWS = 30
NUM_COLS = 2
NUM_ITERS = 160
ETA = 0.0005
W = np.zeros((NUM_ITERS, NUM_COLS))
W_init = np.full((NUM_COLS,), fill_value=-7.1)  # initial guess
W[0, :] = W_init

X, Y = gen_data(num_rows=NUM_ROWS, num_cols=NUM_COLS, type='simple')
# W = np.arange(0, 5, 0.5)  # initial guess for gradient

dL_hist = perform_gradient_descent(W, X, Y, eta=ETA, num_iters=NUM_ITERS)
Y_hat = fit_model(W[-1, :], X)
pointwise_losses, dL = loss_gradient(W[-1, :], X, Y)

fig = plt.figure(figsize=(12, 9))
ax0 = fig.add_subplot(1, 2, 1, projection='3d')
# ax1 = fig.add_subplot(1, 2, 2, projection='3d')
ax1 = fig.add_subplot(1, 2, 2)
# ax0.scatter(range(0, NUM_ROWS), Y)
ax0.plot_surface(X[0], X[1], Y, cmap=cm.coolwarm, linewidth=0, alpha=0.7)
# ax0.plot_surface(X[0], X[1], Y_hat, cmap=cm.jet, linewidth=0, alpha=0.4)
# ax1.plot_surface(X[0], X[1], pointwise_losses, cmap=cm.jet, linewidth=0, alpha=0.8)
ax1.plot(range(0, NUM_ITERS), dL_hist)
# ax1.set_zlabel('Pointwise losses', fontsize=10)
ax0.set_xlabel('X0', fontsize=10)
ax0.set_ylabel('X1', fontsize=10)
ax0.set_zlabel('logSumExp', fontsize=10)

plt.show()
