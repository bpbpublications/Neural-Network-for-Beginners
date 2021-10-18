import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from DeepConvNetwork import DeepConvNet
from common.trainer import Trainer

(a_train, b_train), (a_test, b_test) = load_mnist(flatten=False)

network = DeepConvNet()  
trainer = Trainer(network, a_train, b_train, a_test, b_test,
                  epochs=2, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr':0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# Save parameters
network.save_params("deep_convnet_params.pkl")
print("Saved Network Parameters!")
