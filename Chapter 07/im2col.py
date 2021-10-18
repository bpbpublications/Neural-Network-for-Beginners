import sys, os
import numpy as np
sys.path.append(os.pardir)
from common.util import im2col
a1 = np.random.rand(5, 9, 11, 11)
col1 = im2col(a1, 10, 10, stride=2, pad=1)
print(col1.shape) # (20, 900)
a2 = np.random.rand(30, 7, 9, 9)
col2 = im2col(a2, 10, 10, stride=2, pad=1)
print(col2.shape) # (30, 700)
