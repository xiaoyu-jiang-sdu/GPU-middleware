from typing import Optional, List
import torch
import torch.nn as nn
import numpy as np
from test_engine import compare_torch_and_backend


def generate_arange_tensor(shape: List[int], dtype=torch.float32) -> torch.Tensor:
    """生成可预测的浮点张量，用于精确比较"""
    numel = np.prod(shape)
    return torch.arange(0, numel, dtype=dtype).reshape(shape)


# -------------------------------
# 基础 Linear 测试（有/无 bias）
# -------------------------------
class LinearTest(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        return self.linear(x)


def test_linear_basic_2d_bias():
    """经典 2D 输入 + bias"""
    x = generate_arange_tensor([8, 64])          # batch=8, features=64
    model = LinearTest(in_features=64, out_features=32, bias=True)
    compare_torch_and_backend(model, x, name="linear_2d_bias")


def test_linear_2d_no_bias():
    """无 bias 的 2D Linear"""
    x = generate_arange_tensor([4, 128])
    model = LinearTest(in_features=128, out_features=10, bias=False)
    compare_torch_and_backend(model, x, name="linear_2d_no_bias")


def test_linear_small_dims():
    """极小维度，验证边界情况"""
    x = generate_arange_tensor([1, 3])   # batch=1, features=3
    model = LinearTest(3, 1, bias=True)
    compare_torch_and_backend(model, x, name="linear_small_1x3_to_1")


# -------------------------------
# 多维输入（nn.Linear 只作用于最后一维）
# -------------------------------
def test_linear_3d_sequence():
    """序列模型常见形状： [batch, seq_len, embed_dim]"""
    x = generate_arange_tensor([4, 20, 512])   # B=4, S=20, D=512
    model = LinearTest(512, 256, bias=True)
    compare_torch_and_backend(model, x, name="linear_3d_seq_proj")


def test_linear_4d_vision_after_flatten():
    """视觉模型中常见的 flatten 后 Linear，例如 MLP head"""
    # 模拟 [B, C, H, W] → view/flatten → [B, C*H*W]
    x_flat = generate_arange_tensor([2, 512*7*7])   # B=2, features=25088 (像 ResNet 的 7x7 avg pool 后)
    model = LinearTest(512*7*7, 1000, bias=True)
    compare_torch_and_backend(model, x_flat, name="linear_2d_vision_mlp_head")


def test_linear_4d_direct():
    """直接在 4D 上应用 Linear（最后一维变换）"""
    x = generate_arange_tensor([2, 3, 28, 192])   # e.g. [B, heads, seq, head_dim]
    model = LinearTest(192, 64, bias=False)
    compare_torch_and_backend(model, x, name="linear_4d_no_bias")


# -------------------------------
# 特殊情况
# -------------------------------
def test_linear_1d_input():
    """单样本无 batch 维度（PyTorch 会自动加 batch 维处理）"""
    x = generate_arange_tensor([256])
    model = LinearTest(256, 10, bias=True)
    compare_torch_and_backend(model, x, name="linear_1d_input")


def test_linear_batch_1():
    """batch=1 的情况，验证是否与无 batch 一致"""
    x = generate_arange_tensor([1, 512])
    model = LinearTest(512, 64, bias=True)
    compare_torch_and_backend(model, x, name="linear_batch_1")


def test_linear_large_out_features():
    """输出维度很大（常见于 embedding 投影或分类头）"""
    x = generate_arange_tensor([16, 768])   # 像 BERT 的 hidden_size=768
    model = LinearTest(768, 30522, bias=False)  # 像 vocab size 级别的输出
    compare_torch_and_backend(model, x, name="linear_large_out_no_bias")


# -------------------------------
# 运行所有 Linear 测试
# -------------------------------
def run_all_linear_tests():
    tests = [
        # test_linear_2d_no_bias,
        # test_linear_basic_2d_bias,
        # test_linear_small_dims,
        test_linear_3d_sequence,
        test_linear_4d_vision_after_flatten,
        test_linear_4d_direct,
        test_linear_1d_input,
        test_linear_batch_1,
        test_linear_large_out_features,
    ]
    for test_fn in tests:
        print(f"\n=== Running {test_fn.__name__} ===")
        test_fn()


if __name__ == "__main__":
    run_all_linear_tests()