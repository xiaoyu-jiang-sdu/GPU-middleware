import torch
import torch.nn as nn
from test_engine import compare_torch_and_backend
import numpy as np


def generate_arange_tensor(shape, start=0.0, step=1.0):
    """生成可预测浮点张量"""
    numel = np.prod(shape)
    return torch.arange(start, start + numel * step, step=step, dtype=torch.float32).reshape(shape)


class MatMulTest(nn.Module):
    def __init__(self, cached_input=None, transpose_a=False, transpose_b=False):
        """
        cached_input: 缓存的输入 tensor，总是作为 a
        transpose_a: 是否转置 a
        transpose_b: 是否转置 b
        """
        super().__init__()
        self.cached_input = cached_input
        self.transpose_a = transpose_a
        self.transpose_b = transpose_b

    def forward(self, x):
        # 缓存的输入总是作为 a，传入的 x 作为 b
        a = self.cached_input
        b = x

        if self.transpose_a and a.dim() == 2:
            a = a.t()
        if self.transpose_b and b.dim() == 2:
            b = b.t()

        # 处理 batch matmul
        if a.dim() > 2 or b.dim() > 2:
            return torch.matmul(a, b)
        else:
            return a @ b

# ------------------------
# 测试函数
# ------------------------
def test_matmul_2d():
    """2D 矩阵相乘"""
    cached_a = generate_arange_tensor((3, 4))
    b = generate_arange_tensor((4, 5))
    module = MatMulTest(cached_input=cached_a)
    compare_torch_and_backend(module, b, name="matmul_2d")

def test_matmul_2d_transpose_a():
    """2D 矩阵相乘，转置缓存输入 a"""
    cached_a = generate_arange_tensor((4, 3))
    b = generate_arange_tensor((4, 5))
    module = MatMulTest(cached_input=cached_a, transpose_a=True)
    compare_torch_and_backend(module, b, name="matmul_2d_transpose_a")

def test_matmul_2d_transpose_b():
    """2D 矩阵相乘，转置动态输入 b"""
    cached_a = generate_arange_tensor((3, 4))
    b = generate_arange_tensor((5, 4))
    module = MatMulTest(cached_input=cached_a, transpose_b=True)
    compare_torch_and_backend(module, b, name="matmul_2d_transpose_b")

# ------------------------
# 批量运行所有测试
# ------------------------
def run_all_matmul_tests():
    tests = [
        test_matmul_2d,
        test_matmul_2d_transpose_a,
        test_matmul_2d_transpose_b,
    ]
    for test_fn in tests:
        print(f"\n=== Running {test_fn.__name__} ===")
        test_fn()

# ------------------------
# 入口
# ------------------------
if __name__ == "__main__":
    run_all_matmul_tests()