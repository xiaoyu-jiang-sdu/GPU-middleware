import torch
import torch.nn as nn
from test_engine import compare_torch_and_backend


class Add(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 模拟 residual
        return x + x


def test_add():
    torch.manual_seed(0)
    x = torch.randn(1, 16, 32, 32, device="cpu")
    compare_torch_and_backend(
        Add(),
        x,
        name="Add"
    )


if __name__ == "__main__":
    test_add()
