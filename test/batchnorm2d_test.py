import torch
import torch.nn as nn
from test_engine import compare_torch_and_backend


class BatchNormModule(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(
            C,
            eps=1e-5,
            affine=True,
            track_running_stats=True
        )
        self.bn.eval()  # 强制 inference

    def forward(self, x):
        return self.bn(x)


def test_batch_norm():
    torch.manual_seed(0)
    x = torch.randn(4, 16, 32, 32, device="cpu")
    compare_torch_and_backend(
        BatchNormModule(16),
        x,
        name="BatchNorm2d"
    )


if __name__ == "__main__":
    test_batch_norm()
