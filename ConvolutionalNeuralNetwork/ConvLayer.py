import numpy as np
from misc.mathlab.shaobaobaoer_dl_lab import im2col, col2im


class ConvLayer(object):
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(
            1 + (H + 2 * self.pad - FH / self.stride)
        )
        out_w = int(
            1 + (W + 2 * self.pad - FW / self.stride)
        )

        col = im2col(
            x, FH, FW, self.stride, self.pad
        )
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b

        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        # tranpose 会改变周的次序，对于四轴数据而言，经过以上transpose变换
        # 会将 第二维 换为 第四维，第三维换为第二维 。以此类推。

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
