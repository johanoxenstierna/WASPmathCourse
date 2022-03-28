import numpy as np

x = np.linspace(0.9, 1.1)
y = np.linspace(0.9, 1.1)
z = np.sin(x*y)
# y=np.array([0,1,1.9,3.1])
aa = np.polyfit(x, y, 1)

aa = 6


