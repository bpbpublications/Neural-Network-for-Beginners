import numpy as np
import matplotlib.pylab as plt


def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


def function(x):
    return 0.01*x**2 + 0.1*x 


def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y


x = np.arange(0.0, 30.0, 0.2)
y = function(x)
plt.xlabel("x")
plt.ylabel("f(x)")

tf = tangent_line(function, 6)
y2 = tf(x)

plt.plot(x, y)
plt.plot(x, y2, linestyle='--')
plt.show()
