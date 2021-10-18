import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(a_train, b_train), (a_test, b_test) = load_mnist(flatten=True, normalize=False)

img = a_train[0]
label = b_train[0]
print(label)  # 5

print(img.shape)  # (784,)
img = img.reshape(28, 28)
print(img.shape)  # (28, 28)

img_show(img)
