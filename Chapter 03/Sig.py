import numpy as np
import matplotlib.pylab as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))    


X = np.arange(-6.0, 6.0, 0.2)
Y = sigmoid(X)
plt.plot(X, Y, linestyle='--')
plt.ylim(-0.2, 1.2)
plt.show()
