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
        if hasattr(data, 'detach'):  # torch.Tensor
            return data.detach().cpu().numpy().astype(np.float32)
        return np.array(data, dtype=np.float32)

    def to_numpy(self, tensor):
        # 直接返回 numpy 张量
        return np.array(tensor, dtype=np.float32)

    # =========================
    # 基础算子
    # =========================
    def add(self, a, b):
        a, b = self.tensor(a), self.tensor(b)
        return a + b

    def matmul(self, a, b):
        a, b = self.tensor(a), self.tensor(b)
        return a @ b

    def relu(self, x):
        x = self.tensor(x)
        return np.maximum(x, 0)

    def transpose(self, x):
        x = self.tensor(x)
        return np.transpose(x)

    def mul_scalar(self, x, scalar):
        x = self.tensor(x)
        return x * scalar

    # =========================
    # CNN算子
    # =========================
    def conv2d(self, x, w, b=None, stride=(1, 1), padding=(0, 0)):
        x, w = self.tensor(x), self.tensor(w)
        if b is not None:
            b = self.tensor(b)
        x_t = torch.from_numpy(x)
        w_t = torch.from_numpy(w)
        b_t = torch.from_numpy(b) if b is not None else None

        y_t = torch.nn.functional.conv2d(x_t, w_t, bias=b_t, stride=stride, padding=padding)
        y = self.tensor(y_t)
        return y

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

    def flatten(self, x, axis: int = 1):
        """展平张量"""
        x = self.tensor(x)
        shape = x.shape
        new_shape = shape[:axis] + (-1,)
        return x.reshape(new_shape)

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
