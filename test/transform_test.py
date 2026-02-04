from typing import List

import torch
import torch.nn as nn
from test_engine import compare_torch_and_backend
import numpy as np


def generate_arange_tensor(shape, start=0.0, step=1.0):
    """生成可预测浮点张量，用于 transform 验证"""
    numel = np.prod(shape)
    return torch.arange(start, start + numel * step, step=step, dtype=torch.float32).reshape(shape)


# Transpose
class TransposeTest(nn.Module):
    def __init__(self, perm=None):
        super().__init__()
        self.perm = perm

    def forward(self, x):
        if self.perm is None:
            return x.permute(*reversed(range(x.dim())))
        return x.permute(*self.perm)


def test_transpose_2d():
    x = generate_arange_tensor((3, 5))
    module = TransposeTest(perm=[1, 0])
    compare_torch_and_backend(module, x, name="transpose_2d")


def test_transpose_3d():
    x = generate_arange_tensor((2, 3, 4))
    module = TransposeTest(perm=[0, 2, 1])
    compare_torch_and_backend(module, x, name="transpose_3d")


def test_transpose_default():
    """如果没有 perm，默认反转所有维度"""
    x = generate_arange_tensor((2, 3, 4))
    module = TransposeTest()  # perm=None
    compare_torch_and_backend(module, x, name="transpose_default_reverse")


# Unsqueeze
class UnsqueezeTest(nn.Module):
    def __init__(self, axes: List[int]):
        super().__init__()
        self.axes = axes

    def forward(self, x):
        out = x
        # ONNX Unsqueeze 语义：一次性插入多个轴
        for axis in sorted(self.axes):
            out = torch.unsqueeze(out, axis)
        return out


def test_unsqueeze_single_axis():
    # 输入: [4, 8, 16] -> 用固定整数填充
    x = torch.arange(4 * 8 * 16, dtype=torch.float32).reshape(4, 8, 16)

    model = UnsqueezeTest(axes=[1])

    compare_torch_and_backend(
        model,
        x,
        name="Unsqueeze_single_axis",
    )


def test_unsqueeze_multi_axis():
    # 输入: [8, 32]
    x = torch.arange(8 * 32, dtype=torch.float32).reshape(8, 32)

    model = UnsqueezeTest(axes=[0, 2])

    compare_torch_and_backend(
        model,
        x,
        name="Unsqueeze_multi_axis",
    )


def test_unsqueeze_tail_axis():
    # 输入: [2, 3, 4]
    x = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)

    model = UnsqueezeTest(axes=[3])

    compare_torch_and_backend(
        model,
        x,
        name="Unsqueeze_tail_axis",
    )


# Reshape
class ReshapeTest(nn.Module):
    def __init__(self, shape: List[int]):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        shape = []
        for i, s in enumerate(self.shape):
            if s == 0:
                shape.append(x.shape[i])
            else:
                shape.append(s)
        return torch.reshape(x, shape)


def test_reshape_2d_to_3d():
    x = generate_arange_tensor((6, 4))

    model = ReshapeTest(shape=[2, 3, 4])

    compare_torch_and_backend(
        model,
        x,
        name="reshape_2d_to_3d",
    )


def test_reshape_with_minus_one():
    x = generate_arange_tensor((2, 3, 4))

    model = ReshapeTest(shape=[-1, 4])

    compare_torch_and_backend(
        model,
        x,
        name="reshape_with_minus_one",
    )


def test_reshape_with_zero_dim():
    x = generate_arange_tensor((2, 3, 4))

    # 0 → 继承 input[0] = 2
    model = ReshapeTest(shape=[0, -1])

    compare_torch_and_backend(
        model,
        x,
        name="reshape_with_zero_dim",
    )


def test_reshape_flatten():
    x = generate_arange_tensor((2, 3, 4))

    model = ReshapeTest(shape=[2, -1])

    compare_torch_and_backend(
        model,
        x,
        name="reshape_flatten",
    )


def run_all_transform_tests():
    tests = [
        test_transpose_2d,
        test_transpose_3d,
        test_transpose_default,
        test_unsqueeze_single_axis,
        test_unsqueeze_multi_axis,
        test_unsqueeze_tail_axis,
        test_reshape_2d_to_3d,
        test_reshape_with_minus_one,
        test_reshape_with_zero_dim,
        test_reshape_flatten,
    ]
    for test_fn in tests:
        print(f"\n=== Running {test_fn.__name__} ===")
        test_fn()


if __name__ == "__main__":
    run_all_transform_tests()
