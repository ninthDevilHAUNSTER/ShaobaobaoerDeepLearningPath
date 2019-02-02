class SGD(object):
    '''
    随机梯度下降法
    把之前得复习一下
    '''

    def __init__(self, lr=.001):
        self.lr = lr

    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
