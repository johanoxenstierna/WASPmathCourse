
import numpy as np
import matplotlib.pyplot as plt


def foo():
    return

X = np.linspace(0, 5, 20)  # obs can't use discrete values (granularity must be sub-zero)
sums = np.zeros((X.shape))
Y = np.zeros((X.shape))
Y_grad1 = np.zeros((X.shape))
Y_grad2 = np.zeros((X.shape))
for i in range(len(X)):
    sum_ = np.sum(np.exp(X[0:i+1]))
    sums[i] = sum_
    Y[i] = np.log(sum_)

    softmax = np.exp(X[i]) / sum_
    Y_grad1[i] = softmax
    Y_grad2[i] = softmax * (1 - softmax)


fig, ax = plt.subplots(figsize=(9, 7))
ax_logsumexp, = ax.plot(X, Y, '--', color='blue')
ax_grad1, = ax.plot(X, Y_grad1, '--', color='red')
ax_grad2, = ax.plot(X[1:], Y_grad2[1:], '-', color='green')
ax.legend([ax_logsumexp, ax_grad1, ax_grad2], ["logsumexp", "softmax (Jacobian)", "softmax_der (Hessian)"])
ax.set_xlabel("X")
ax.set_ylabel("Y")
plt.show()
aa = 6
