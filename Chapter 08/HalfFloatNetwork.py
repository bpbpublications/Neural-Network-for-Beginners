import sys, os
sys.path.append(os.pardir)
import numpy as np
from DeepConvNetwork import DeepConvNet
from dataset.mnist import load_mnist


(a_train, b_train), (a_test, b_test) = load_mnist(flatten=False)

network = DeepConvNet()
network.load_params("deep_convnet_params.pkl")

sampled = 1000
a_test = a_test[:sampled]
b_test = b_test[:sampled]

print("Calculate Accuracy (float64) ... ")
print(network.accuracy(a_test, b_test))

# Convert to float16
a_test = a_test.astype(np.float16)
for param in network.params.values():
    param[...] = param.astype(np.float16)

print("Calculate Accuracy (float16) ... ")
print(network.accuracy(a_test, b_test))
