import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from TwoLayerNet import TwoLayerNet


(a_train, b_train), (a_test, b_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = a_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    a_batch = a_train[batch_mask]
    b_batch = b_train[batch_mask]

    #grad = network.numerical_gradient(a_batch, b_batch)
    grad = network.gradient(a_batch, b_batch)

    for key in ('W1', 'X1', 'W2', 'X2'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(a_batch, b_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(a_train, b_train)
        test_acc = network.accuracy(a_test, b_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)
