import os, sys
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.trainer import Trainer


(a_train, b_train), (a_test, b_test) = load_mnist(normalize=True)

a_train = a_train[:400]
b_train = b_train[:400]

# Dropout setting ====================================
use_dropout = True
dropout_ratio = 0.2
# ====================================================

network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100],
                              output_size=10, use_dropout=use_dropout, dropout_ration=dropout_ratio)
trainer = Trainer(network, a_train, b_train, a_test, b_test,
                  epochs=31, mini_batch_size=100,
                  optimizer='sgd', optimizer_param={'lr': 0.01}, verbose=True)
trainer.train()

train_acc_list, test_acc_list = trainer.train_acc_list, trainer.test_acc_list


markers = {'Train': 'O', 'Test': 'S'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, marker='O', label='Train', markevery=10)
plt.plot(x, test_acc_list, marker='S', label='Test', markevery=10)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
