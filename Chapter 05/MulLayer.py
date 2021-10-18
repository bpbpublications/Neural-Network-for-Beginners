class MulLayer:
    def __init__(self):
        self.a = None
        self.b = None

    def forward(self, a, b):
        self.a = a
        self.b = b                
        out = a * b

        return out

    def backward(self, dout):
        da = dout * self.b
        db = dout * self.a

        return da, db


class AddLayer:
    def __init__(self):
        pass

    def forward(self, a, b):
        out = a + b

        return out

    def backward(self, dout):
        da = dout * 1
        db = dout * 1

        return da, db
