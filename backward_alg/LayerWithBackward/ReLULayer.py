class ReluLayer(object):
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        # 备注，这是np的一种写法，传入布尔值，则会对所有为true的元素进行操作
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

#
# import numpy as np
#
# if __name__ == '__main__':
#     x = np.array([[1.2, -0.5],
#                   [-2.0, 3.0]])
#     relu = Relu()
#     xp = relu.forward(x)
#     print(relu.mask)
#     x[relu.mask] = 0
#     print(x)
