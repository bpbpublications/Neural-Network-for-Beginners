import os, sys
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD


(a_train, b_train), (a_test, b_test) = load_mnist(normalize=True)

a_train = a_train[:400]
b_train = b_train[:400]

# weight decay setting ==============================
weight_decay_lambda = 0.1
# ====================================================

network = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                        weight_decay_lambda=weight_decay_lambda)
optimizer = SGD(lr=0.01)

max_epochs = 31
train_size = a_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(1000000000):
    batch_mask = np.random.choice(train_size, batch_size)
    a_batch = a_train[batch_mask]
    b_batch = b_train[batch_mask]

    grads = network.gradient(a_batch, b_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(a_train, b_train)
        test_acc = network.accuracy(a_test, b_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print("Epoch:" + str(epoch_cnt) + ", Train acc:" + str(train_acc) + ", Test acc:" + str(test_acc))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break


markers = {'Train': 'o', 'Test': 's'}
x = np.arange(max_epochs)
plt.plot(x, train_acc_list, marker='o', label='Train', markevery=10)
plt.plot(x, test_acc_list, marker='s', label='Test', markevery=10)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
