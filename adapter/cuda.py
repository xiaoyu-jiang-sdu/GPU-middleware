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

    def tensor(self, data):
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        return torch.tensor(data, device=self.device)

    def to_numpy(self, tensor):
        return tensor.detach().cpu().numpy()

    def add(self, a, b):
        return a + b

    def matmul(self, a, b):
        return torch.matmul(a, b)

    def relu(self, x):
        return torch.relu(x)

    def conv2d(self, x, w, b=None, stride=(1, 1), padding=(0, 0)):
        return torch.nn.functional.conv2d(
            x,
            w,
            bias=b,
            stride=stride,
            padding=padding
        )

    def global_avg_pool(self, x):
        return torch.mean(x, dim=(2, 3), keepdim=True)

    def transpose(self, x):
        return x.t()

    def mul_scalar(self, x, scalar: float):
        return x * scalar

    def max_pool2d(self, x, kernel_size, stride, padding):
        return torch.nn.functional.max_pool2d(x, kernel_size, stride=stride, padding=padding)

    def flatten(self, x, axis: int = 1):
        # x: torch.Tensor
        shape = x.shape
        new_shape = shape[:axis] + (-1,)
        return x.view(new_shape)
