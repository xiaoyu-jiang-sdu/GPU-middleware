import torch
import torch.nn as nn
from test_engine import compare_torch_and_backend


class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(512, 10, bias=False)

    def forward(self, x):
        return self.fc(x)


def test_linear():
    torch.manual_seed(0)
    x = torch.randn(4, 512, device="cpu")
    compare_torch_and_backend(
        Linear(),
        x,
        name="Linear"
    )


if __name__ == "__main__":
    # torch.manual_seed(0)
    # x = torch.randn(4, 512, device="cuda")
    # model = ONNXAwareModel(
    #     Linear(),
    #     input_shape=x.shape,
    #     backend="cuda"
    # )
    # out = model(x).detach().cpu().numpy()
    # print(out)
    test_linear()
