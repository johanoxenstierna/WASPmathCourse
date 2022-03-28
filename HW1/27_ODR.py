# ODR tutorial
from scipy.odr import ODR, Model, Data
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2)

from mpl_toolkits.mplot3d import Axes3D

def func(beta, data):
    # x,y = data
    # a,b,c = beta
    # return a*x+b*y+c

    x, y = data
    a, b, c = beta

    # return np.sin(np.pi * a * x * b * y)
    return np.sqrt(4 - 3 * x**2 * y)

def func2(beta, data):
    x, y = data
    a, b = beta
    return -np.pi * (x - a) + np.pi * (y - b)


# THE DATA ============================
N = 20
# x = np.random.randn(N)
# y = np.random.randn(N)
x = np.linspace(0.8, 1.2, N)
y = np.linspace(0.8, 1.2, N)

# z = func([-3,-1,2],[x,y])#+np.random.normal(size=N)  # second arg adds noise
z = func([1, 1, 1],[x,y])# +np.random.normal(size=N)*0.1  # second arg adds noise
# = =============================================

data = Data([x,y],z)
model = Model(func)
odr = ODR(data, model, [1, 1, 1])
odr.set_job(fit_type = 0)
res = odr.run()

Y,X = np.mgrid[y.min():y.max():20j,x.min():x.max():20j]
Z = func(res.beta, [X,Y])
f = plt.figure()
pl = f.add_subplot(111,projection='3d')
pl.scatter3D(x,y,z)
pl.plot_surface(X,Y,Z,alpha=0.4)
pl.set_xlabel("X")
pl.set_ylabel("Y")
pl.set_zlabel("Z")
plt.show()