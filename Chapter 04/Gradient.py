import numpy as np
import matplotlib.pylab as plt
from Gradient_2D import numerical_gradient


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append( x.copy() )

        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x, np.array(x_history)


def function(x):
    return x[0]**2 + x[1]**2


init_x = np.array([-4.0, 5.0])

lr = 0.1
step_num = 30
x, x_history = gradient_descent(function, init_x, lr=lr, step_num=step_num)

plt.plot( [-6, 6], [0,0], '--b')
plt.plot( [0,0], [-6, 6], '--b')
plt.plot(x_history[:,0], x_history[:,1], 'o')

plt.xlim(-4.5, 4.5)
plt.ylim(-5.5, 5.5)
plt.xlabel("X0")
plt.ylabel("X1")
plt.show()
