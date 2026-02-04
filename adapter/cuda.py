import torch
from adapter.base import BackendAdapter
from .factory import register_adapter

"""
CUDA adapter
使用pytorch的function
"""


@register_adapter("cuda")
class CudaAdapter(BackendAdapter):
    def __init__(self, **kwargs):
        # 确保设备可用
        assert torch.cuda.is_available(), "CUDA is not available"
        device = kwargs.get("device", "cuda:0")
        if torch.cuda.is_available():
            self.device = torch.device(device)

        # 参数张量缓存
        self._param_cache = {}

    # 张量迁移，采用缓存方式避免重复
    def _to_device(self, tensor, cache=False, cache_key=None):
        """
        将 tensor 迁移到目标 device
        - cache=True：用于参数张量（weight / bias 等）
        """
        if tensor is None:
            return None

        if not isinstance(tensor, torch.Tensor):
            return tensor

        # 已在目标 device
        if tensor.device == self.device:
            return tensor

        # 参数缓存逻辑
        if cache and cache_key is not None:
            if cache_key in self._param_cache:
                return self._param_cache[cache_key]
            t = tensor.to(self.device)
            self._param_cache[cache_key] = t
            return t

        return tensor.to(self.device)

    # =========================
    # 张量管理
    # =========================
    def tensor(self, data, cache=False, cache_key=None):
        if isinstance(data, torch.Tensor):
            return self._to_device(data)
        return torch.tensor(data, device=self.device)

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy()

    # =========================
    # 基础算子
    # =========================
    def add(self, a, b):
        a = self._to_device(a)
        b = self._to_device(b)
        return a + b

    def sub(self, a, b):
        a = self._to_device(a)
        b = self._to_device(b)
        return torch.sub(a, b)

    def mul(self, a, b):
        a = self._to_device(a)
        b = self._to_device(b)
        return torch.mul(a, b)

    def div(self, a, b):
        a = self._to_device(a)
        b = self._to_device(b)
        return torch.div(a, b)

    def pow(self, a, b):
        a = self._to_device(a)
        b = self._to_device(b)
        return torch.pow(a, b)

    def mod(self, a, b):
        a = self._to_device(a)
        b = self._to_device(b)
        return torch.remainder(a, b)

    def mul_scalar(self, x, scalar: float):
        x = self._to_device(x)
        return x * scalar

    # =========================
    # NN 算子
    # =========================
    def matmul(self, a, b, alpha=1.0, beta=0.0, transA=False, transB=False, C=None):
        a = self._to_device(a)
        b = self._to_device(b)

        out = torch.matmul(a.transpose(-2, -1) if transA else a,
                           b.transpose(-2, -1) if transB else b)
        out = alpha * out
        if C is not None:
            out += beta * self.tensor(C)
        return out

    def conv2d(self, x, w, b=None, stride=(1, 1), padding=(0, 0)):
        x = self._to_device(x)
        w = self._to_device(w, cache=True, cache_key=id(w))
        b = self._to_device(b, cache=True, cache_key=id(b)) if b is not None else None

        return torch.nn.functional.conv2d(
            x,
            w,
            bias=b,
            stride=stride,
            padding=padding
        )

    # =========================
    # 激活算子
    # =========================
    def relu(self, x):
        x = self._to_device(x)
        return torch.relu(x)

    def erf(self, x):
        x = self._to_device(x)
        return torch.erf(x)

    def sqrt(self, x):
        x = self._to_device(x)
        return torch.sqrt(x)

    # =========================
    # 池化算子
    # =========================
    def max_pool2d(self, x, kernel_size, stride, padding):
        x = self._to_device(x)
        return torch.nn.functional.max_pool2d(
            x,
            kernel_size,
            stride=stride,
            padding=padding
        )

    def global_avg_pool(self, x):
        x = self._to_device(x)
        return torch.mean(x, dim=(2, 3), keepdim=True)

    # =========================
    # 归一化
    # =========================
    def batch_norm_2d(self, x, weight, bias, running_mean, running_var, eps):
        # 确保所有张量在 CUDA 上
        x = self._to_device(x)
        weight = self._to_device(weight, cache=True, cache_key=id(weight))
        bias = self._to_device(bias, cache=True, cache_key=id(bias))
        running_mean = self._to_device(running_mean, cache=True, cache_key=id(running_mean))
        running_var = self._to_device(running_var, cache=True, cache_key=id(running_var))

        # reshape 为 broadcast 形式
        mean = running_mean.view(1, -1, 1, 1)
        var = running_var.view(1, -1, 1, 1)
        weight = weight.view(1, -1, 1, 1)
        bias = bias.view(1, -1, 1, 1)

        # BN 推理公式
        y = (x - mean) / torch.sqrt(var + eps)
        y = y * weight + bias

        return y

    # =========================
    # shape view
    # =========================
    def transpose(self, x, perm):
        x = self._to_device(x)
        return x.permute(*perm)

    def unsqueeze(self, x, axes):
        x = self._to_device(x)
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
        return x.view(*out_shape)

    def reshape(self, x, shape):
        if isinstance(shape, torch.Tensor):
            shape = shape.tolist()
        in_shape = list(x.shape)

        x = self._to_device(x)
        new_shape = []
        for i, s in enumerate(shape):
            if s == 0:
                new_shape.append(in_shape[i])
            else:
                new_shape.append(s)

        return x.reshape(new_shape)

    # =========================
    # transform
    # =========================
    def flatten(self, x, axis: int = 1):
        x = self._to_device(x)
        shape = x.shape
        new_shape = shape[:axis] + (-1,)
        return x.view(new_shape)

    def concat(self, xs, axis):
        xs = self._to_device(xs)
        return torch.cat(xs, dim=axis)
