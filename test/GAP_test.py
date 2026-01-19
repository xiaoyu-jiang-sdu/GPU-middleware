import torch
import torch.nn as nn
from test_engine import compare_torch_and_backend


class GlobalAvgPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        return self.gap(x)


def test_GAP():
    torch.manual_seed(0)
    x = torch.randn(1, 16, 7, 7, device="cpu")
    compare_torch_and_backend(
        GlobalAvgPool(),
        x,
        name="GlobalAvgPool"
    )


if __name__ == "__main__":
    test_GAP()
