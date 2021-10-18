from AND_Gate import AND
from OR_Gate import OR
from NAND_Gate import NAND


def XOR(a1, a2):
    x1 = NAND(a1, a2)
    x2 = OR(a1, a2)
    y = AND(x1, x2)
    return y

if __name__ == '__main__':
    for i in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = XOR(i[0], i[1])
        print(str(i) + " -> " + str(y))
