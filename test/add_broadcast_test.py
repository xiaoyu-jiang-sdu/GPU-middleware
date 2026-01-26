from test_engine import compare_torch_and_backend


import torch
import torch.nn as nn
import numpy as np

class AddBroadcastTest(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.register_buffer("bias", bias)

    def forward(self, x):
        return x + self.bias


def generate_arange_tensor(shape, start=0.0, step=1.0):
    """生成可预测的浮点数张量，便于调试误差"""
    numel = np.prod(shape)
    return torch.arange(start, start + numel * step, step=step, dtype=torch.float32).reshape(shape)


def test_add_broadcast_nd_1d():
    """1D + 0D (标量广播)"""
    torch.manual_seed(0)
    x = generate_arange_tensor((128,))
    b = torch.tensor(3.14159, dtype=torch.float32)          # 标量
    module = AddBroadcastTest(b)
    compare_torch_and_backend(module, x, name="broadcast_1d_scalar")


def test_add_broadcast_nd_1d_to_1d():
    """1D + 1D (相同形状)"""
    torch.manual_seed(0)
    x = generate_arange_tensor((256,))
    b = torch.randn(256, dtype=torch.float32) * 10
    module = AddBroadcastTest(b)
    compare_torch_and_backend(module, x, name="add_1d_1d")


def test_add_broadcast_nd_2d_channel():
    """2D + (1,C) → 经典 channel-wise bias"""
    torch.manual_seed(0)
    x = generate_arange_tensor((64, 16))
    b = torch.tensor([0.5, -1.2, 2.7, 0.0, -3.1, 4.4, 1.8, 2.2,
                      -0.9, 1.1, 3.3, -2.5, 0.7, -1.6, 5.0, 0.1],
                     dtype=torch.float32).view(1, 16)
    module = AddBroadcastTest(b)
    compare_torch_and_backend(module, x, name="broadcast_2d_channel")


def test_add_broadcast_nd_3d_batch_channel():
    """3D + (1,C,1)"""
    torch.manual_seed(0)
    x = generate_arange_tensor((4, 8, 32))
    b = torch.randn(8, dtype=torch.float32).view(1, 8, 1) * 5
    module = AddBroadcastTest(b)
    compare_torch_and_backend(module, x, name="broadcast_3d_batch_channel")


def test_add_broadcast_nd_4d_height_broadcast():
    """4D + (1,1,H,1) → 只在高度维度广播"""
    torch.manual_seed(0)
    x = generate_arange_tensor((2, 5, 12, 12))
    b = torch.linspace(-2.0, 2.0, 12, dtype=torch.float32).view(1, 1, 12, 1)
    module = AddBroadcastTest(b)
    compare_torch_and_backend(module, x, name="broadcast_4d_height_only")


def test_add_broadcast_nd_4d_mixed_broadcast():
    """4D + (C,1,1) → 只广播 channel 之外的维度"""
    torch.manual_seed(0)
    x = generate_arange_tensor((3, 6, 8, 8))  # shape: (N=3, C=6, H=8, W=8)
    b = torch.randn(6, dtype=torch.float32).view(1, 6, 1, 1) * 3

    module = AddBroadcastTest(b)
    compare_torch_and_backend(module, x, name="broadcast_4d_channel_prefix")


def test_add_broadcast_nd_5d():
    """5D 张量 + 部分广播"""
    torch.manual_seed(0)
    x = generate_arange_tensor((2, 3, 4, 5, 6))
    b = torch.randn(1, 3, 1, 1, 6, dtype=torch.float32) * 4
    module = AddBroadcastTest(b)
    compare_torch_and_backend(module, x, name="broadcast_5d_mixed")


def test_add_broadcast_nd_scalar_to_5d():
    """标量广播到 5D"""
    torch.manual_seed(0)
    x = generate_arange_tensor((2, 2, 4, 4, 4))
    b = torch.tensor(7.777, dtype=torch.float32)
    module = AddBroadcastTest(b)
    compare_torch_and_backend(module, x, name="broadcast_scalar_to_5d")


def test_add_broadcast_nd_shape_mismatch_right():
    """右对齐广播：右侧维度匹配，左侧广播"""
    torch.manual_seed(0)
    x = generate_arange_tensor((8, 1, 16, 32))
    b = generate_arange_tensor((16, 32)).view(1, 1, 16, 32) * 0.1
    module = AddBroadcastTest(b)
    compare_torch_and_backend(module, x, name="broadcast_right_aligned")


def test_add_broadcast_nd_shape_mismatch_left():
    """左侧广播：左侧维度为1"""
    torch.manual_seed(0)
    x = generate_arange_tensor((1, 4, 8, 16, 32))
    b = torch.randn(4, 8, 16, 32, dtype=torch.float32) * 2
    module = AddBroadcastTest(b)
    compare_torch_and_backend(module, x, name="broadcast_left_broadcast")


def run_all_broadcast_tests():
    tests = [
        test_add_broadcast_nd_1d,
        test_add_broadcast_nd_1d_to_1d,
        test_add_broadcast_nd_2d_channel,
        test_add_broadcast_nd_3d_batch_channel,
        test_add_broadcast_nd_4d_height_broadcast,
        test_add_broadcast_nd_4d_mixed_broadcast,
        test_add_broadcast_nd_5d,
        test_add_broadcast_nd_scalar_to_5d,
        test_add_broadcast_nd_shape_mismatch_right,
        test_add_broadcast_nd_shape_mismatch_left,
    ]

    for test_fn in tests:
        print(f"\n=== Running {test_fn.__name__} ===")
        test_fn()


if __name__ == "__main__":
    run_all_broadcast_tests()
