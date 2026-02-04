import torch
import torch.nn as nn
from test_engine import compare_torch_and_backend


class Tensor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def test_tensor():
    torch.manual_seed(0)
    x = torch.randn(4, 512, device="cpu")
    compare_torch_and_backend(
        Tensor(),
        x,
        name="Tensor"
    )


if __name__ == "__main__":
    test_tensor()
