import sys, os
sys.path.append(os.pardir)
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['X1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['X2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['X1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['X2'])

        self.lastLayer = SoftmaxWithLoss()

    def calculate(self, a):
        for layer in self.layers.values():
            a = layer.forward(a)

        return a
        
    def loss(self, a, t):
        b = self.calculate(a)
        return self.lastLayer.forward(b, t)

    def accuracy(self, a, t):
        b = self.calculate(a)
        b = np.argmax(b, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(b == t) / float(a.shape[0])
        return accuracy

    def numerical_gradient(self, a, t):
        loss_W = lambda W: self.loss(a, t)
        
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['X1'] = numerical_gradient(loss_W, self.params['X1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['X2'] = numerical_gradient(loss_W, self.params['X2'])
        
        return grads

    def gradient(self, a, t):
        # forward
        self.loss(a, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'], grads['X1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['X2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
