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
                data = data.view(1)
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
    # 基础算子, 二元运算符
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
        out_shape = self._broadcast_shape(a.shape, b.shape)
        out = dcu.DCUTensor(list(out_shape), dtype=a.dtype)

        self.engine.add(
            a, b, out,
            self.ctx
        )
        return out

    def sub(self, a, b):
        a = self._to_dcu_tensor(a)
        b = self._to_dcu_tensor(b)
        out_shape = self._broadcast_shape(a.shape, b.shape)
        out = dcu.DCUTensor(list(out_shape), dtype=a.dtype)

        self.engine.sub(
            a, b, out,
            self.ctx
        )
        return out

    def mul(self, a, b):
        a = self._to_dcu_tensor(a)
        b = self._to_dcu_tensor(b)
        out_shape = self._broadcast_shape(a.shape, b.shape)
        out = dcu.DCUTensor(list(out_shape), dtype=a.dtype)

        self.engine.mul(
            a, b, out,
            self.ctx
        )
        return out

    def div(self, a, b):
        a = self._to_dcu_tensor(a)
        b = self._to_dcu_tensor(b)
        out_shape = self._broadcast_shape(a.shape, b.shape)
        out = dcu.DCUTensor(list(out_shape), dtype=a.dtype)

        self.engine.div(
            a, b, out,
            self.ctx
        )
        return out

    def pow(self, a, b):
        a = self._to_dcu_tensor(a)
        b = self._to_dcu_tensor(b)
        out_shape = self._broadcast_shape(a.shape, b.shape)
        out = dcu.DCUTensor(list(out_shape), dtype=a.dtype)

        self.engine.pow(
            a, b, out,
            self.ctx
        )
        return out

    def mod(self, a, b):
        a = self._to_dcu_tensor(a)
        b = self._to_dcu_tensor(b)
        out_shape = self._broadcast_shape(a.shape, b.shape)
        out = dcu.DCUTensor(list(out_shape), dtype=a.dtype)

        self.engine.mod(
            a, b, out,
            self.ctx
        )
        return out

    def mul_scalar(self, x, scalar):
        x = self._to_dcu_tensor(x)
        out = dcu.DCUTensor(x.shape, dtype=x.dtype)
        self.engine.mul_scalar(x, out, float(scalar), self.ctx)
        return out

    # =========================
    # NN 算子
    # =========================
    def matmul(self, a, b, alpha=1.0, beta=0.0, transA=False, transB=False, C=None):
        a = self._to_dcu_tensor(a)
        b = self._to_dcu_tensor(b)

        if len(a.shape) == 1:
            a = self.unsqueeze(a, [0])  # [features] → [1, features]
        if len(b.shape) == 1:
            b = self.unsqueeze(b, [0])

        batch_shape = a.shape[:-2]  # 前 N-2 维是 batch
        M, K_a = a.shape[-2], a.shape[-1]
        K_b, N = b.shape[-2], b.shape[-1]

        # 校验矩阵维度
        if transA:
            M, K_a = K_a, M
        if transB:
            K_b, N = N, K_b
        assert K_a == K_b, f"Inner dimensions mismatch: {K_a} vs {K_b}"

        # 输出 shape
        out_shape = batch_shape + (M, N)
        out = dcu.DCUTensor(out_shape, dtype=a.dtype)

        self.engine.matmul(a, b, out, transA, transB, alpha, beta, self.ctx)

        # C 不为空，累加
        if C is not None:
            C = self._to_dcu_tensor(C)
            out = self.add(out, self.mul_scalar(C, beta))

        return out

    def conv2d(self, x, w, b=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
        x = self._to_dcu_tensor(x)
        w = self._to_dcu_tensor(w, cache=True, cache_key=id(w))
        N, C, H, W = x.shape
        K, _, R, S = w.shape
        stride_h, stride_w = stride
        pad_h, pad_w = padding
        dilation_h, dilation_w = dilation

        Ho = (H + 2 * pad_h - dilation_h * (R - 1) - 1) // stride_h + 1
        Wo = (W + 2 * pad_w - dilation_w * (S - 1) - 1) // stride_w + 1

        out = dcu.DCUTensor([N, K, Ho, Wo], dtype=x.dtype)
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
                                         out.shape, [1, K, 1, 1], self.ctx)

        return out

    # =========================
    # 激活算子
    # =========================
    def relu(self, x):
        x = self._to_dcu_tensor(x)
        out = dcu.DCUTensor(x.shape, dtype=x.dtype)
        self.engine.relu(x, out, self.ctx)
        return out

    def erf(self, x):
        x = self._to_dcu_tensor(x)
        out = dcu.DCUTensor(x.shape, dtype=x.dtype)
        self.engine.erf(x, out, self.ctx)
        return out

    def sqrt(self, x):
        x = self._to_dcu_tensor(x)
        out = dcu.DCUTensor(x.shape, dtype=x.dtype)
        self.engine.sqrt(x, out, self.ctx)
        return out

    # =========================
    # 池化算子
    # =========================
    def max_pool2d(self, x, kernel_size, stride, padding):
        x = self._to_dcu_tensor(x)
        N, C, H, W = x.shape
        outH = (H + 2*padding[0] - kernel_size[0]) // stride[0] + 1
        outW = (W + 2*padding[1] - kernel_size[1]) // stride[1] + 1
        out = dcu.DCUTensor([N, C, outH, outW], dtype=x.dtype)
        self.engine.max_pool2d(x, out,
                               N, C, H, W,
                               kernel_size[0], kernel_size[1],
                               stride[0], stride[1],
                               padding[0], padding[1],
                               self.ctx)
        return out

    def global_avg_pool(self, x):
        x = self._to_dcu_tensor(x)
        N, C, H, W = x.shape
        out = dcu.DCUTensor([N, C, 1, 1], dtype=x.dtype)  # 保留 1x1 输出维度
        self.engine.global_avg_pool(x, out, N, C, H, W, self.ctx)
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

        out = dcu.DCUTensor(x.shape, dtype=x.dtype)
        N, C, H, W = x.shape
        self.engine.batch_norm_2d(x, out,
                                  weight, bias,
                                  running_mean, running_var,
                                  N, C, H, W,
                                  eps, self.ctx)
        return out

    # =========================
    # shape view
    # =========================
    def transpose(self, x, perm):
        x = self._to_dcu_tensor(x)
        out = self.engine.transpose(x, perm, self.ctx)
        return out

    def unsqueeze(self, x, axes):
        x = self._to_dcu_tensor(x)
        return self.engine.unsqueeze(x, list(axes), self.ctx)

    def reshape(self, x, shape):
        x = self._to_dcu_tensor(x)
        return self.engine.reshape(x, shape, self.ctx)

    # =========================
    # transform
    # =========================
    def flatten(self, x, axis=1):
        x = self._to_dcu_tensor(x)
        shape = x.shape
        ndim = x.ndim
        if axis < 0:
            axis += ndim
        if axis < 0 or axis > ndim:
            raise ValueError(f"flatten: axis out of range {axis} for shape {shape}")

        # 计算输出 shape
        outer = int(np.prod(shape[:axis], dtype=np.int64))
        inner = int(np.prod(shape[axis:], dtype=np.int64))
        out_shape = [outer, inner]

        out = dcu.DCUTensor(out_shape, dtype=x.dtype)
        self.engine.flatten(x, out, axis, self.ctx)
        return out

    def concat(self, xs, axis):
        dcu_tensors = [self._to_dcu_tensor(x) for x in xs]
        return self.engine.concat(dcu_tensors, axis, self.ctx)
