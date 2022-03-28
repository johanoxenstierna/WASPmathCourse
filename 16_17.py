
import numpy as np
import random
import matplotlib.pyplot as plt

NUM_SAMPLES = np.arange(1, 2000, 1)

result_outer = np.zeros((len(NUM_SAMPLES), ), dtype=float)

for k in range(len(result_outer)):
    num_experiments = NUM_SAMPLES[k]
    result_inner = np.zeros((num_experiments,), dtype=int)
    for i in range(num_experiments):
        flag_head = False
        num_trials = 0
        while flag_head == False:
            num_trials += 1
            rand = random.random()
            if rand < 0.5:
                flag_head = True
        result_inner[i] = num_trials

    expectation = np.mean(result_inner)
    result_outer[k] = expectation

fig, ax = plt.subplots()
ax.plot(NUM_SAMPLES, result_outer)
ax.set_xlabel('Number of samples', fontsize=10)
ax.set_ylabel('Expected value', fontsize=10)
plt.show()

aa = 5

