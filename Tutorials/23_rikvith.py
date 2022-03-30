import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_loss(w, xvals, yvals):
    return np.sum((w*xvals + w*xvals**2 - yvals)**2)

def get_gradient(w, xvals, yvals):
    grad = np.sum(2*(xvals + xvals**2)*(w*xvals + w*xvals**2 - yvals))
    return grad

numvals = 50
true_w = 0.5

xvals = np.sort(5*(np.random.random(numvals) - 0.5))
yvals = true_w*(xvals + xvals**2) + np.random.normal(0, 0.1, numvals)
yvals[int(numvals*0.1):] += 2  # shift all but some initial values up

fig = plt.figure(figsize=(7,7))
ax0 = fig.add_subplot(2, 2, 1)

ax0.scatter(xvals, yvals)
ax0.set_xlabel('x', fontsize=20)
ax0.set_ylabel('y', fontsize=20)
ax0.set_title('Data', fontsize=25)


# 2. Loss
wvals = np.arange(0,1.5,0.1)
lossvals = [get_loss(w, xvals, yvals) for w in wvals]

ax1 = fig.add_subplot(2, 2, 2)
ax1.plot(wvals, lossvals)
ax1.set_xlabel('w', fontsize=20)
ax1.set_ylabel('Loss', fontsize=20)
ax1.set_title('Loss Function', fontsize=25)


# 3. Grad desc ============
def perform_gradient_descent(init_w, eta, num_iters, get_gradient):
    w_vals = [init_w]
    for _ in range(num_iters):
        grad = get_gradient(w_vals[-1], xvals, yvals)
        w_vals.append(w_vals[-1] - eta*grad)
    return w_vals

w_vals = perform_gradient_descent(0, 0.01, 50, get_gradient)

ax2 = fig.add_subplot(2, 2, 3)
ax2.plot(w_vals)
ax2.set_xlabel('iteration', fontsize=20)
ax2.set_ylabel('w', fontsize=20)
ax2.set_title('Gradient Descent Progress\nFinal w: %s'%round(w_vals[-1],2), fontsize=5)

plt.show()