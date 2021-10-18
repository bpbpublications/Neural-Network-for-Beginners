import os, sys
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from common.multi_layer_net import MultiLayerNet
from common.optimizer import SGD


(a_train, b_train), (a_test, b_test) = load_mnist(normalize=True)

train_size = a_train.shape[0]
batch_size = 128
max_iterations = 500


weight_init_types = {'STD=0.01': 0.01, 'Eden': 'Sigmoid', 'Kia': 'relu'}
optimizer = SGD(lr=0.01)

networks = {}
train_loss = {}
for key, weight_type in weight_init_types.items():
    networks[key] = MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100],
                                  output_size=10, weight_init_std=weight_type)
    train_loss[key] = []


for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    a_batch = a_train[batch_mask]
    b_batch = b_train[batch_mask]
    
    for key in weight_init_types.keys():
        grads = networks[key].gradient(a_batch, b_batch)
        optimizer.update(networks[key].params, grads)
    
        loss = networks[key].loss(a_batch, b_batch)
        train_loss[key].append(loss)
    
    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for key in weight_init_types.keys():
            loss = networks[key].loss(a_batch, b_batch)
            print(key + ":" + str(loss))


markers = {'STD=0.01': 'o', 'Eden': 's', 'Kia': 'D'}
x = np.arange(max_iterations)
for key in weight_init_types.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 2.5)
plt.legend()
plt.show()
