import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from TwoLayerNet import TwoLayerNet


(a_train, b_train), (a_test, b_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

a_batch = a_train[:4]
b_batch = b_train[:4]

grad_numerical = network.numerical_gradient(a_batch, b_batch)
grad_backprop = network.gradient(a_batch, b_batch)

for key in grad_numerical.keys():
    diff = np.average( np.abs(grad_backprop[key] - grad_numerical[key]) )
    print(key + ":" + str(diff))
