import sys, os
import numpy as np
from misc.mathlab.shaobaobaoer_layer_lab import *
from misc.mathlab.shaobaobaoer_dl_lab import numerical_gradient
from collections import OrderedDict


class TwoLayerNet(object):
    def __init__(self
                 , input_size, hidden_size, output_size,
                 weight_hinit_std=.01):
        # 初始权重
        self.params = {}
        self.params['W1'] = weight_hinit_std * \
                            np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_hinit_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)
        # 生成层
        self.layers = OrderedDict()
        self.layers['Affine1'] = \
            Affine(self.params['W1'], self.params['b1'])
        self.layers['Affine2'] = \
            Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    # x : input value
    # t : observe value
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t / float(x.shape[0]))
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        g = {}
        g['W1'] = numerical_gradient(loss_W, self.params['W1'])
        g['b1'] = numerical_gradient(loss_W, self.params['b1'])
        g['W2'] = numerical_gradient(loss_W, self.params['W2'])
        g['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return g

    def gradient(self, x, t):
        self.loss(x, t)

        # back
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        g = {}
        g['W1'] = self.layers['Affine1'].dW
        g['W2'] = self.layers['Affine2'].dW
        g['b1'] = self.layers['Affine1'].db
        g['b2'] = self.layers['Affine2'].db

        return g
