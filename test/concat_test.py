import torch
import torch.nn as nn
from test_engine import compare_torch_and_backend
import numpy as np


def generate_arange_tensor(shape, start=0.0, step=1.0):
    """生成可预测浮点张量，用于 transform 验证"""
    numel = np.prod(shape)
    return torch.arange(start, start + numel * step, step=step, dtype=torch.float32).reshape(shape)


class ConcatTest(nn.Module):
    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.axis)


def test_concat_2d_axis0():
    x1 = generate_arange_tensor((2, 4), start=0)
    x2 = generate_arange_tensor((3, 4), start=100)

    model = ConcatTest(axis=0)

    compare_torch_and_backend(
        model,
        (x1, x2),
        name="Concat_2d_axis0",
        backend="cuda"
    )

def test_concat_2d_axis1():
    x1 = generate_arange_tensor((3, 2), start=0)
    x2 = generate_arange_tensor((3, 5), start=100)

    model = ConcatTest(axis=1)

    compare_torch_and_backend(
        model,
        (x1, x2),
        name="Concat_2d_axis1",
        backend="cuda"
    )

def test_concat_3d_middle_axis():
    x1 = generate_arange_tensor((2, 3, 4), start=0)
    x2 = generate_arange_tensor((2, 5, 4), start=100)

    model = ConcatTest(axis=1)

    compare_torch_and_backend(
        model,
        (x1, x2),
        name="Concat_3d_axis1",
        backend="cuda"
    )

def test_concat_3d_tail_axis():
    x1 = generate_arange_tensor((2, 3, 4), start=0)
    x2 = generate_arange_tensor((2, 3, 6), start=100)

    model = ConcatTest(axis=2)

    compare_torch_and_backend(
        model,
        (x1, x2),
        name="Concat_3d_axis2",
        backend="cuda"
    )

def test_concat_negative_axis():
    x1 = generate_arange_tensor((2, 3, 4), start=0)
    x2 = generate_arange_tensor((2, 3, 5), start=100)

    model = ConcatTest(axis=-1)

    compare_torch_and_backend(
        model,
        (x1, x2),
        name="Concat_negative_axis",
        backend="cuda"
    )

def run_all_concat_tests():
    tests = [
        test_concat_2d_axis0,
        test_concat_2d_axis1,
        test_concat_3d_middle_axis,
        test_concat_3d_tail_axis,
        test_concat_negative_axis,
    ]

    for test_fn in tests:
        print(f"\n=== Running {test_fn.__name__} ===")
        test_fn()


if __name__ == "__main__":
    run_all_concat_tests()
