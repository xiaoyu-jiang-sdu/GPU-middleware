import torch
import torch.nn as nn
from test_engine import compare_torch_and_backend


class Flatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)


def test_flatten():
    torch.manual_seed(0)
    x = torch.randn(2, 32, 7, 7, device="cpu")
    compare_torch_and_backend(
        Flatten(),
        x,
        name="Flatten"
    )


if __name__ == "__main__":
    test_flatten()
