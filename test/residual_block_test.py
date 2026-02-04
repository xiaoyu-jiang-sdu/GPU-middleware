import torch
import torch.nn as nn
import torch.nn.functional as F
from test_engine import compare_torch_and_backend


class ResidualBlockTest(nn.Module):
    def __init__(self, w1, w2, b1=None, b2=None):
        super().__init__()
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)

        self.b1 = nn.Parameter(b1) if b1 is not None else None
        self.b2 = nn.Parameter(b2) if b2 is not None else None

    def forward(self, x):
        out = F.conv2d(x, self.w1, self.b1, stride=1, padding=1)
        out = F.conv2d(out, self.w2, self.b2, stride=1, padding=1)
        out = out + x
        out = F.relu(out)
        return out


def test_residual_block():
    torch.manual_seed(0)

    # 输入: [N, C, H, W]
    x = torch.randn(64, 8, 32, 32, device="cpu")

    # 卷积权重
    w1 = torch.randn(8, 8, 3, 3, device="cpu")
    w2 = torch.randn(8, 8, 3, 3, device="cpu")

    b1 = None
    b2 = None

    block = ResidualBlockTest(w1, w2, b1, b2)

    compare_torch_and_backend(
        block,
        x,
        name="ResidualBlock"
    )


if __name__ == "__main__":
    test_residual_block()
