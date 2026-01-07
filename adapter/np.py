import numpy as np
from .base import BackendAdapter
from .factory import register_adapter

"""
cpu上的np运算
设备不可用时采用此种方式
"""


@register_adapter("cpu")
class CpuAdapter(BackendAdapter):

    def tensor(self, data):
        return np.array(data)

    def to_numpy(self, tensor):
        return tensor

    def add(self, a, b):
        return a + b

    def matmul(self, a, b):
        return a @ b

    def relu(self, x):
        return np.maximum(x, 0)

    def conv2d(self, x, w, b=None, stride=(1, 1), padding=(0, 0)):
        N, C, H, W = x.shape
        out, _, kH, kW = w.shape
        sH, sW = stride
        pH, pW = padding

        x = np.pad(x, ((0, 0), (0, 0), (pH, pH), (pW, pW)))
        outH = (H + 2 * pH - kH) // sH + 1
        outW = (W + 2 * pW - kW) // sW + 1

        y = np.zeros((N, out, outH, outW))
        for n in range(N):
            for c in range(out):
                for i in range(outH):
                    for j in range(outW):
                        region = x[n, :, i * sH:i * sH + kH, j * sW:j * sW + kW]
                        y[n, c, i, j] = np.sum(region * w[c])
                if b is not None:
                    y[n, c] += b[c]
        return y

    def global_avg_pool(self, x):
        return x.mean(axis=(2, 3), keepdims=True)

    def transpose(self, x):
        return x.T

    def mul_scalar(self, x, scalar):
        return x * scalar

    def max_pool2d(self, x, kernel_size, stride, padding):
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        kh, kw = kernel_size
        sh, sw = stride
        ph, pw = padding

        # 填充
        x_padded = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode='constant', constant_values=-np.inf)

        H_out = (H + 2*ph - kh) // sh + 1
        W_out = (W + 2*pw - kw) // sw + 1
        y = np.zeros((N, C, H_out, W_out), dtype=np.float32)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        h_start = i * sh
                        w_start = j * sw
                        h_end = h_start + kh
                        w_end = w_start + kw
                        y[n, c, i, j] = np.max(x_padded[n, c, h_start:h_end, w_start:w_end])
        return y

    def flatten(self, x, axis: int = 1):
        shape = x.shape
        # 保留前 axis 维，展平后面的维度
        new_shape = shape[:axis] + (-1,)
        return x.reshape(new_shape)