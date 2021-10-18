import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 8, 0.2)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, linestyle="--", label="sin")
plt.plot(x, y2, label="cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title('sin & cos')
plt.legend()
plt.show()
