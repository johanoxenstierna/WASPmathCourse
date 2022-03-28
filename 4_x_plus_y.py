

import random
import numpy as np

x = np.linspace(0, 1, 100)
y = 2 * x
Y1, Y2 = np.meshgrid(y, y)
Z = Y1 + Y2
where_less_1 = np.argwhere(Z <= 1.0)
prob = len(where_less_1) / Y1.size

gg = 7