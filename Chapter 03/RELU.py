import numpy as np
import matplotlib.pylab as plt


def relu(x):
    return np.maximum(0, x)


x = np.arange(-6.0, 6.0, 0.2)
y = relu(x)
plt.plot(x, y, linestyle='--')
plt.ylim(-1.0, 6.0)
plt.show()
