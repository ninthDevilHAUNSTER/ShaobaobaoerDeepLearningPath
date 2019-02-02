# Sigmod函数的逆向传播函数为 dL * y^2 * (1-y)
# 我自己推了一遍
import numpy as np


class SigmodLayer(object):
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * self.out ** 2 * (1.0 - self.out)
        return dx
