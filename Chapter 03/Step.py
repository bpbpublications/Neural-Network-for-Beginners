import numpy as np
import matplotlib.pylab as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int)


X = np.arange(-6.0, 6.0, 0.2)
Y = step_function(X)
plt.plot(X, Y, linestyle='--')
plt.ylim(-0.1, 1.2)
plt.show()
