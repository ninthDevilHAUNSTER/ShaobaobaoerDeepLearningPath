class AddLayer(object):
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
