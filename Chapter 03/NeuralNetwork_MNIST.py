import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (a_train, b_train), (a_test, b_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return a_test, b_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def calculate(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
count = 0
for i in range(len(x)):
    y = calculate(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        count += 1

print("Accuracy:" + str(float(count) / len(x)))
