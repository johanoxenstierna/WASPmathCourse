from scipy.optimize import minimize, rosen, rosen_der
import numpy as np
import matplotlib.pyplot as plt

#
# def my_fun(X):
#
#     val = (X[0] - 5)**2 + 5
#     return val

def my_fun(X):
    val = 1 / (X[0] * X[1]) + X[0] + X[1]
    return val

fig, ax = plt.subplots(figsize=(9, 6))

# x0 = [10000.3, 0.7]

# x = np.linspace(0, 50, 100)
# y = (x - 5)**2 + 5
# ax.plot(x, y)


res = minimize(my_fun, (0.1, 0.2), bounds=((0, None), (0, None)))  # 4, 22
print(res)

# ax.set_xlim(left=-10, right=10)
# ax.set_ylim([-10, 30])
# plt.show()
