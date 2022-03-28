

import numpy as np
import matplotlib.pyplot as plt

# # 10 ============================
# probs = [1.0]
#
# for i in range(1, 100):
#     numerator = 365 - i
#     p = (numerator / 365) * probs[i - 1]
#     probs.append(p)
#
# probs = np.asarray(probs)
# index = np.argwhere(probs < 0.5)[0][0]
# fig, ax = plt.subplots()
# ax.plot(probs)
#
# ax.axvline(x=22, color='red')
# ax.axhline(y=0.5, color='red')
# ax.set_xlabel("Number of persons")
# ax.set_ylabel("Probability")

# 12 ===========================
# n = np.linspace(1, 100, 1000)

n = range(1, 100)
y = [1 / (n * (n + 1)) for n in range(1, 100)]
sum_y = np.sum(y)
assert([_y < 1.0 for _y in y])  # each point must be less than 1
assert(sum_y <= 1.0)  # the likelihoods must not have a sum greater than 1

fig, ax = plt.subplots()
ax.plot(n, y)
plt.show()

