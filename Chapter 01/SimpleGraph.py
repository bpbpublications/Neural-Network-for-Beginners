import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 8, 0.2)
y = np.sin(x)

plt.plot(x, y, linestyle="--")
plt.show()
