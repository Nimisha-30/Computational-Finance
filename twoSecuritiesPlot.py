import matplotlib.pyplot as plt
import numpy as np
import math

# SBI
# mu = 0.11; sigma = 0.265

# ICICI
# mu = 0.15; sigma = 0.231

sigma = []
mu = []

weights = np.linspace(0, 1)
for w in weights:
    mu.append(w * 0.11 + (1 - w) * 0.15)
    sigma.append(math.sqrt(w ** 2 * 0.265 ** 2 + (1 - w) ** 2 * 0.231 ** 2 + 2 * w * (1 - w) * 0.805 * 0.265 * 0.231))

plt.plot(sigma, mu)
plt.scatter(0.265, 0.11, label = 'SBI')
plt.scatter(0.231, 0.15, label = 'ICICI')
plt.grid(True)
plt.title('Risk-Return Graph')
plt.xlabel('Risk')
plt.ylabel('Expected Return')
plt.legend()
plt.show()