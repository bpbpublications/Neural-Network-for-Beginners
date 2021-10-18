import numpy as np


def OR(a1, a2):
    x = np.array([a1, a2])
    y = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(y*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    for i in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        z = OR(i[0], i[1])
        print(str(i) + " -> " + str(z))
