import random
import numpy as np

x = np.linspace(0, 1, 20)
y = np.linspace(0, 2, 20)
X, Y = np.meshgrid(x, y)
Z = (X + Y) / 3

# ??