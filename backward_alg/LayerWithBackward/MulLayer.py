class MulLayer(object):
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        return x * y

    def backword(self, dout):
        dx = dout * self.y
        dy = dout * self.x
        return dx, dy


