import sys, os
sys.path.append(os.pardir)
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.optimizer import *


def f(a, b):
    return a**2 / 20.0 + b**2


def df(a, b):
    return a / 10.0, 2.0*b

init_pos = (-7.0, 2.0)
params = {}
params['a'], params['b'] = init_pos[0], init_pos[1]
grads = {}
grads['a'], grads['b'] = 0, 0


optimizers = OrderedDict()
optimizers["SGD"] = SGD(lr=0.95)
optimizers["Momentum"] = Momentum(lr=0.1)
optimizers["AdaGrad"] = AdaGrad(lr=1.5)
optimizers["Adam"] = Adam(lr=0.3)

idx = 1

for key in optimizers:
    optimizer = optimizers[key]
    a_history = []
    b_history = []
    params['a'], params['b'] = init_pos[0], init_pos[1]
    
    for i in range(30):
        a_history.append(params['a'])
        b_history.append(params['b'])
        
        grads['a'], grads['b'] = df(params['a'], params['b'])
        optimizer.update(params, grads)
    

    a = np.arange(-10, 10, 0.01)
    b = np.arange(-5, 5, 0.01)
    
    X, Y = np.meshgrid(a, b) 
    Z = f(X, Y)
    
    # for simple contour line  
    mask = Z > 7
    Z[mask] = 0
    
    # plot 
    plt.subplot(2, 2, idx)
    idx += 1
    plt.plot(a_history, b_history, 'o-', color="blue")
    plt.contour(X, Y, Z)
    plt.ylim(-10, 10)
    plt.xlim(-10, 10)
    plt.plot(0, 0, '+')
    plt.title(key)
    plt.xlabel("x")
    plt.ylabel("y")
    
plt.show()
