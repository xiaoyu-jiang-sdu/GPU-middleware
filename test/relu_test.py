import torch
import torch.nn as nn
from test_engine import compare_torch_and_backend


class Relu(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        return self.relu(x)


def test_relu():
    torch.manual_seed(0)
    x = torch.randn(1, 8, 32, 32, device="cpu")
    compare_torch_and_backend(
        Relu(),
        x,
        name="ReLU"
    )


if __name__ == "__main__":
    test_relu()
