import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from DeepConvNetwork import DeepConvNet
from dataset.mnist import load_mnist


(a_train, b_train), (a_test, b_test) = load_mnist(flatten=False)

network = DeepConvNet()
network.load_params("deep_convnet_params.pkl")

print("Calculating Test Accuracy ... ")
sampled = 1000
a_test = a_test[:sampled]
b_test = b_test[:sampled]

classified_ids = []

acc = 0.0
batch_size = 100

for i in range(int(a_test.shape[0] / batch_size)):
    tx = a_test[i*batch_size:(i+1)*batch_size]
    tt = b_test[i*batch_size:(i+1)*batch_size]
    y = network.predict(tx, train_flg=False)
    y = np.argmax(y, axis=1)
    classified_ids.append(y)
    acc += np.sum(y == tt)
    
acc = acc / a_test.shape[0]
print("Test Accuracy:" + str(acc))

classified_ids = np.array(classified_ids)
classified_ids = classified_ids.flatten()
 
max_view = 20
current_view = 1

fig = plt.figure()
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2)

mis_pairs = {}
for i, val in enumerate(classified_ids == b_test):
    if not val:
        ax = fig.add_subplot(4, 5, current_view, xticks=[], yticks=[])
        ax.imshow(a_test[i].reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
        mis_pairs[current_view] = (b_test[i], classified_ids[i])
            
        current_view += 1
        if current_view > max_view:
            break

print("======= Misclassified Result =======")
print("{View Index: (Label, Inference), ...}")
print(mis_pairs)

plt.show()
