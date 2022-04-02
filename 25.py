
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm

# def loss_gradient(W, X, Y):
#     pointwise_losses = np.zeros(Y.shape, dtype=float)
#     # for w in W:
#     for r in range(Y.shape[0]):
#         for c in range(Y.shape[1]):
#             x0 = X[0][r, c]
#             x1 = X[1][r, c]
#             x = np.array([x0, x1])
#
#             pointwise_losses[r, c] = W.dot(x) - Y[r, c]
#             # pointwise_losses[r, c] = 2 * (w * x0 + w * x1 - y)
#     dL = (1/len(pointwise_losses)) * np.sum(pointwise_losses)  # MSE
#     # dL = np.sum(pointwise_losses)  # SE
#     return pointwise_losses, dL
#
#
# def perform_gradient_descent(W, X, Y, eta, num_iters):
#     dL_hist = np.zeros((num_iters,))
#     for i in range(1, num_iters):
#         pointwise_losses, dL = loss_gradient(W[i - 1, :], X, Y)
#         dL_hist[i - 1] = dL
#         W[i, :] = W[i - 1, :] - eta*dL
#     return dL_hist



def rosenbrock(X, a=1, b=1):
    x, y = X  # x is second value, y is first value
    return (a - x)**2 + b * (y - x**2)**2

def rosenbrock_grad(X, a=1, b=1):
    x, y = X  # x
    return np.array([
        2 * (x - a) - 4 * b * x * (y - x**2),
        2 * b * (y - x**2)
    ])


def rosenbrock_hess(X, a=1, b=100):
    x, y = X
    return np.matrix([
        [2 - 4 * b * (y - 3 * x**2), -4 * b * x],
        [-4 * b * x, 2 * b]
    ])


def gradient_descent(J_grad, x_init, alpha=0.01, epsilon=1e-10, max_iterations=1000):
    x = x_init
    for i in range(max_iterations):
        x = x - alpha * J_grad(x)
        if np.linalg.norm(J_grad(x)) < epsilon:
            return x, i + 1
    return x, max_iterations

# def rosenbrock(X):  # 2 matrices that combined give all combinations of two values in a range.
#
#     # # 3D ===================
#     # Y = np.zeros(X0.shape, dtype=float)  # 2D
#     #
#     # for r in range(Y.shape[0]):
#     #     for c in range(Y.shape[1]):
#     #         x0 = X0[r, c]
#     #         x1 = X0[r, c]
#     #         inner = (x1 - x0**2)**2 + (1 - x0)**2
#     #         fx = 100 * inner  # there is no sum because I only do it for 1 iteration (there are only 2 dimension)
#     #         Y[r, c] = fx
#
#     # # 2D  =========
#     sum = 0
#     Y = np.zeros(X.shape, dtype=float)
#     for i in range(1, len(Y)):
#
#         x_this = X[i]
#         x_prev = X[i - 1]
#         inner = (x_this - x_prev)**2 + (1 - x_this)**2
#         sum += inner
#         Y[i - 1] = sum
#     return Y

def gss(f, a, b, tol=1e-7):
    phi = (np.sqrt(5) + 1) / 2
    d = b - (b - a) / phi
    c = a + (b - a) / phi

    while abs(d - c) > tol:
        if f(d) < f(c):
            b = c
        else:
            a = d

        d = b - (b - a) / phi
        c = a + (b - a) / phi

    return (a + b) / 2


def gradient_descent_optimal(J, J_grad, x_init, epsilon=1e-10, max_iterations=1000):
    x = x_init
    for i in range(max_iterations):
        q = lambda alpha: J(x - alpha * J_grad(x))
        alpha = gss(q, 0, 1)
        x = x - alpha * J_grad(x)
        if np.linalg.norm(J_grad(x)) < epsilon:
            return x, i + 1
    return x, max_iterations


X = np.linspace(0.5, 1.5, num=25)
X0, X1 = np.meshgrid(X, X)
results = np.zeros(X0.shape, dtype=float)
# for i in range(NUM_TESTS):

for r in range(X0.shape[0]):
    print(r)
    for c in range(X0.shape[1]):

        # all possible combinations of two variables
        # x_init = np.zeros(2)
        x_init = [X0[r, c], X1[r, c]]
        # x_init = [1.16, 0.5]
        x_min, it = gradient_descent_optimal(rosenbrock, rosenbrock_grad, x_init, max_iterations=400)
        # x_min, it = gradient_descent(rosenbrock_grad, x_init, alpha=0.002, max_iterations=1000)

        # x_temp = np.array([[1., 1.], x_min])

        dist_to_optimal = np.linalg.norm([1., 1.] - x_min)
        results[r, c] = dist_to_optimal


print('x* =', x_min)
print('Rosenbrock(x*) =', rosenbrock(x_min))
print('Grad Rosenbrock(x*) =', rosenbrock_grad(x_min))
print('Iterations =', it)

#
# X = np.full((50, ), fill_value=1.0)
# Y = rosenbrock(X)
# aa = np.min(Y)
# X = np.linspace(-1, 4)
# TODO have to show several candidate X's here and that above is what really works.

# # 3D =================================
# X_this, X_prev = np.meshgrid(X, X)
# X0, X1 = np.meshgrid(X, X)  # all possible combinations of two variables
# Y = rosenbrock(X0, X1)
fig = plt.figure(figsize=(12, 9))
ax0 = fig.add_subplot(1, 1, 1, projection='3d')
# ax0.plot(X, Y)
ax0.plot_surface(X0, X1, results, cmap=cm.jet, linewidth=0, alpha=0.6)
ax0.set_xlabel('X0', fontsize=10)
ax0.set_ylabel('X1', fontsize=10)
plt.show()