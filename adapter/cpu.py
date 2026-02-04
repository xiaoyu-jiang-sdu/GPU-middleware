import numpy as np
import torch
from .base import BackendAdapter
from .factory import register_adapter

"""
cpu上的np运算
当没有 GPU/MLU/NPU 时使用，全部算子基于 numpy 实现。
"""


@register_adapter("cpu")
class CpuAdapter(BackendAdapter):
    # =========================
    # 张量管理
    # =========================
    def tensor(self, data, cache=False, cache_key=None):
        # 把输入数据转换为 numpy.ndarray
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            return data
        if hasattr(data, 'detach'):  # torch.Tensor
            return data.detach().cpu().numpy().astype(np.float32)
        return np.array(data, dtype=np.float32)

    def to_numpy(self, tensor):
        # 直接返回 numpy 张量
        return np.array(tensor, dtype=np.float32)

    # =========================
    # 基础算子, 二元运算符
    # =========================
    def add(self, a, b):
        return a + b

    def sub(self, a, b):
        return np.subtract(a, b)

    def mul(self, a, b):
        return np.multiply(a, b)

    def div(self, a, b):
        return np.divide(a, b)

    def pow(self, a, b):
        return np.power(a, b)

    def mod(self, a, b):
        return np.mod(a, b)

    def mul_scalar(self, x, scalar):
        return x * scalar

    # =========================
    # NN 算子
    # =========================
    def matmul(self, a, b, alpha=1.0, beta=0.0, transA=False, transB=False, C=None):
        out = np.matmul(a.swapaxes(-1, -2) if transA else a,
                        b.swapaxes(-1, -2) if transB else b)
        out = alpha * out
        if C is not None:
            out += beta * C
        return out

    def conv2d(self, x, w, b=None, stride=(1, 1), padding=(0, 0)):
        x = self.tensor(x)
        w = self.tensor(w)
        if b is not None:
            b = self.tensor(b)
        x_t = torch.from_numpy(x)
        w_t = torch.from_numpy(w)
        b_t = torch.from_numpy(b) if b is not None else None

        y_t = torch.nn.functional.conv2d(x_t, w_t, bias=b_t, stride=stride, padding=padding)
        y = self.tensor(y_t)
        return y

    # =========================
    # 激活算子
    # =========================
    def relu(self, x):
        return np.maximum(x, 0)

    def erf(self, x):
        x_tensor = torch.from_numpy(self.tensor(x))
        return torch.erf(x_tensor).numpy()

    def sqrt(self, x):
        return np.sqrt(x)

    # =========================
    # 池化算子
    # =========================
    def max_pool2d(self, x, kernel_size, stride, padding):
        x = self.tensor(x)
        x_t = torch.from_numpy(x)
        y_t = torch.nn.functional.max_pool2d(x_t, kernel_size=kernel_size, stride=stride, padding=padding)
        y = self.tensor(y_t)
        return y

    def global_avg_pool(self, x):
        """全局平均池化"""
        x = self.tensor(x)
        return x.mean(axis=(2, 3), keepdims=True)

    # =========================
    # 归一化
    # =========================
    def batch_norm_2d(self, x, weight, bias, running_mean, running_var, eps=1e-5):
        """二维批归一化 (推理)"""
        x = self.tensor(x)
        weight = self.tensor(weight)
        bias = self.tensor(bias)
        running_mean = self.tensor(running_mean)
        running_var = self.tensor(running_var)

        # reshape 为广播形式
        mean = running_mean.reshape(1, -1, 1, 1)
        var = running_var.reshape(1, -1, 1, 1)
        weight = weight.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)

        y = (x - mean) / np.sqrt(var + eps)
        y = y * weight + bias
        return y

    # =========================
    # shape view
    # =========================
    def transpose(self, x, perm):
        x = self.tensor(x)
        return np.transpose(x, axes=perm)

    def unsqueeze(self, x: np, axes):
        axes = sorted(axes)
        in_shape = list(x.shape)
        out_rank = len(in_shape) + len(axes)
        out_shape = []

        in_i = 0
        axes_set = set(axes)
        for i in range(out_rank):
            if i in axes_set:
                out_shape.append(1)
            else:
                out_shape.append(in_shape[in_i])
                in_i += 1
        return x.reshape(out_shape)

    def reshape(self, x, shape):
        shape = list(shape)
        in_shape = list(x.shape)

        new_shape = []
        for i, s in enumerate(shape):
            if s == 0:
                new_shape.append(in_shape[i])
            else:
                new_shape.append(s)

        # 支持 -1 reshape
        return x.reshape(new_shape)

    # =========================
    # transform
    # =========================
    def flatten(self, x, axis: int = 1):
        """展平张量"""
        shape = x.shape
        new_shape = shape[:axis] + (-1,)
        return x.reshape(new_shape)

    def concat(self, xs, axis):
        return np.concatenate(xs, axis=axis)
