import torch
import torch.nn as nn
from test_engine import compare_torch_and_backend


class MaxPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

    def forward(self, x):
        return self.pool(x)


def test_max_pool():
    torch.manual_seed(0)
    x = torch.randn(1, 8, 32, 32, device="cpu")
    compare_torch_and_backend(
        MaxPool(),
        x,
        name="MaxPool"
    )


if __name__ == "__main__":
    test_max_pool()
