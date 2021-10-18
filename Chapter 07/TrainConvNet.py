import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from SimpleConvNetwork import SimpleConvNet
from common.trainer import Trainer


(a_train, b_train), (a_test, b_test) = load_mnist(flatten=False)

#Reduce data if it takes too long to process
a_train, b_train = a_train[:5000], b_train[:5000]
a_test, b_test = a_test[:1000], b_test[:1000]

max_epochs = 2 #Increse the count in case required

network = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
                        
trainer = Trainer(network, a_train, b_train, a_test, b_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

network.save_params("params.pkl")
print("Saved Network Parameters!")

markers = {'Train': 'o', 'Test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='Train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='Test', markevery=2)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
