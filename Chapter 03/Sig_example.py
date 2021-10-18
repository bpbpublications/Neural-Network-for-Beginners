import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identity_function(a): 
    return a

def init_network(): 
    network = {}
    network['W1'] = np.array([[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]] )
    network['b1'] = np.array([0.5, 1.0, 0.7])
    network['W2'] = np.array([[0.2, 0.7], [0.6, 0.9], [0.5, 0.7]] )
    network['b2'] = np.array([0.2, 0.3])
    network['W3'] = np.array([[0.2, 0.5], [0.7, 0.8]]) 
    network['b3'] = np.array([0.2, 0.3])
    return network
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2)  +  b2 
    z2 = sigmoid(a2)
    a3 = np.dot(z2,  W3)  +  b3
    y = identity_function(a3)
    return y
network = init_network() 
x =  np.array([1.5, 1.0]) 
y = forward(network, x)
print(y) 

