import torch
import torch.nn as nn
import torch.nn.functional as F
from test_engine import compare_torch_and_backend


class Conv2DTest(nn.Module):
    def __init__(self, w, b=None):
        super().__init__()
        self.weight = nn.Parameter(w)
        if b is not None:
            self.bias = nn.Parameter(b)
        else:
            self.bias = None

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, stride=1, padding=1)


def test_conv2d():
    torch.manual_seed(0)
    x = torch.randn(1, 3, 32, 32, device="cpu")
    w = torch.randn(8, 3, 3, 3, device="cpu")
    b = None

    conv = Conv2DTest(w, b)
    compare_torch_and_backend(
        conv,
        x,
        name="Conv2d"
    )


if __name__ == "__main__":
    test_conv2d()
