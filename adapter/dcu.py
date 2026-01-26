import numpy as np
import torch

from adapter.base import BackendAdapter
from .factory import register_adapter
import dcu  # pybind11 绑定的模块


@register_adapter("dcu")
class DcuAdapter(BackendAdapter):
    def __init__(self, **kwargs):
        # 创建 DCU context 和 engine
        self.ctx = dcu.Context()
        self.engine = dcu.Engine()

        # 参数缓存
        self._param_cache = {}

    # =========================
    # 张量管理
    # =========================
    def _to_dcu_tensor(self, data, cache=False, cache_key=None):
        if data is None:
            return None

        if isinstance(data, dcu.DCUTensor):
            return data

        if cache and cache_key is not None and cache_key in self._param_cache:
            return self._param_cache[cache_key]

        # torch.Tensor → numpy
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu()
            if data.dim() == 0:
                data = data.view(1)  # 或者 data.unsqueeze(0)
            data = data.contiguous().numpy().astype(np.float32)

        elif isinstance(data, np.ndarray):
            if data.ndim == 0:
                data = data.reshape(1)
            data = data.astype(np.float32, copy=False)

        else:
            data = np.array(data, dtype=np.float32)
            if data.ndim == 0:
                data = data.reshape(1)

        t = dcu.DCUTensor(data)
        if cache and cache_key is not None:
            self._param_cache[cache_key] = t

        return t

    def tensor(self, data, cache=False, cache_key=None):
        return self._to_dcu_tensor(data, cache, cache_key)

    def to_numpy(self, tensor):
        """
        将 DCUTensor 转回 numpy
        """
        if isinstance(tensor, dcu.DCUTensor):
            return tensor.to_numpy()
        return tensor

    # =========================
    # 基础算子
    # =========================

    # 推算广播加法的shape
    @staticmethod
    def _broadcast_shape(a_shape, b_shape):
        na, nb = len(a_shape), len(b_shape)
        ndim = max(na, nb)
        out = []

        for i in range(ndim):
            da = a_shape[i - (ndim - na)] if i >= ndim - na else 1
            db = b_shape[i - (ndim - nb)] if i >= ndim - nb else 1

            if da == db or da == 1 or db == 1:
                out.append(max(da, db))
            else:
                raise ValueError(f"Invalid broadcast: {a_shape} vs {b_shape}")

        return tuple(out)

    def add(self, a, b):
        a = self._to_dcu_tensor(a)
        b = self._to_dcu_tensor(b)

        out_shape = self._broadcast_shape(a.shape(), b.shape())
        out = dcu.DCUTensor(list(out_shape))

        self.engine.add(
            a, b, out,
            list(a.shape()), list(b.shape()),
            self.ctx
        )
        return out
    def matmul(self, a, b):
        a = self._to_dcu_tensor(a)  # [M, N]
        b = self._to_dcu_tensor(b)  # [N, K]

        M, N = a.shape()
        N2, K = b.shape()
        assert N == N2

        out = dcu.DCUTensor([M, K])
        self.engine.matmul(a, b, out, M, K, N, self.ctx)
        return out

    def relu(self, x):
        x = self._to_dcu_tensor(x)
        out = dcu.DCUTensor(x.shape())
        self.engine.relu(x, out, np.prod(x.shape()), self.ctx)
        return out

    def transpose(self, x):
        x = self._to_dcu_tensor(x)
        rows, cols = x.shape()
        out = dcu.DCUTensor([cols, rows])
        self.engine.transpose(x, out, rows, cols, self.ctx)
        return out

    def mul_scalar(self, x, scalar):
        x = self._to_dcu_tensor(x)
        out = dcu.DCUTensor(x.shape())
        self.engine.mul_scalar(x, out, float(scalar), np.prod(x.shape()), self.ctx)
        return out

    # =========================
    # CNN算子
    # =========================
    def conv2d(self, x, w, b=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
        x = self._to_dcu_tensor(x)
        w = self._to_dcu_tensor(w, cache=True, cache_key=id(w))
        N, C, H, W = x.shape()
        K, _, R, S = w.shape()
        stride_h, stride_w = stride
        pad_h, pad_w = padding
        dilation_h, dilation_w = dilation

        Ho = (H + 2 * pad_h - dilation_h * (R - 1) - 1) // stride_h + 1
        Wo = (W + 2 * pad_w - dilation_w * (S - 1) - 1) // stride_w + 1

        out = dcu.DCUTensor([N, K, Ho, Wo])
        self.engine.conv2d(x, w, out,
                           N, C, H, W,
                           K, R, S,
                           stride_h, stride_w,
                           pad_h, pad_w,
                           dilation_h, dilation_w,
                           groups,
                           self.ctx)

        if b is not None:
            # 保留 NCHW 维度，bias shape [K] → [1,K,1,1]
            self.engine.add_broadcast_nd(out, self._to_dcu_tensor(b), out,
                                         out.shape(), [1, K, 1, 1], self.ctx)

        return out

    def max_pool2d(self, x, kernel_size, stride, padding):
        x = self._to_dcu_tensor(x)
        N, C, H, W = x.shape()
        outH = (H + 2*padding[0] - kernel_size[0]) // stride[0] + 1
        outW = (W + 2*padding[1] - kernel_size[1]) // stride[1] + 1
        out = dcu.DCUTensor([N, C, outH, outW])
        self.engine.max_pool2d(x, out,
                               N, C, H, W,
                               kernel_size[0], kernel_size[1],
                               stride[0], stride[1],
                               padding[0], padding[1],
                               self.ctx)
        return out

    def global_avg_pool(self, x):
        x = self._to_dcu_tensor(x)
        N, C, H, W = x.shape()
        out = dcu.DCUTensor([N, C, 1, 1])  # 保留 1x1 输出维度
        self.engine.global_avg_pool(x, out, N, C, H, W, self.ctx)
        return out

    def flatten(self, x, axis=1):
        x = self._to_dcu_tensor(x)
        shape = x.shape()
        outer = np.prod(shape[:axis])
        inner = np.prod(shape[axis:])
        out = dcu.DCUTensor([outer, inner])
        self.engine.flatten(x, out, outer, inner, self.ctx)
        return out

    # =========================
    # 归一化
    # =========================
    def batch_norm_2d(self, x, weight, bias, running_mean, running_var, eps=1e-5):
        x = self._to_dcu_tensor(x)
        weight = self._to_dcu_tensor(weight, cache=True, cache_key=id(weight))
        bias = self._to_dcu_tensor(bias, cache=True, cache_key=id(bias))
        running_mean = self._to_dcu_tensor(running_mean, cache=True, cache_key=id(running_mean))
        running_var = self._to_dcu_tensor(running_var, cache=True, cache_key=id(running_var))

        out = dcu.DCUTensor(x.shape())
        N, C, H, W = x.shape()
        self.engine.batch_norm_2d(x, out,
                                  weight, bias,
                                  running_mean, running_var,
                                  N, C, H, W,
                                  eps, self.ctx)
        return out

    def debug_tensor(t, name):
        if isinstance(t, dcu.DCUTensor):
            print(f"[DCU] {name}: shape={t.shape()}")
