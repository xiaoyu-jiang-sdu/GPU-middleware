import torch
import torch.nn as nn

from test.test_engine import compare_torch_and_backend


class EqualTest(nn.Module):
    def __init__(self, other):
        super().__init__()
        self.register_buffer("other", other)

    def forward(self, x):
        return x == self.other


def test_equal_1d_same_shape():
    x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    y = torch.tensor([1, 0, 3, 5], dtype=torch.float32)
    module = EqualTest(y)
    compare_torch_and_backend(module, x, name="equal_1d_same_shape")


def test_equal_1d_scalar():
    x = torch.arange(10, dtype=torch.float32)
    y = torch.tensor(5.0, dtype=torch.float32)
    module = EqualTest(y)
    compare_torch_and_backend(module, x, name="equal_1d_scalar")


def test_equal_2d_broadcast_row():
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    y = torch.tensor([0, 1, 2, 3], dtype=torch.float32).view(1, 4)
    module = EqualTest(y)
    compare_torch_and_backend(module, x, name="equal_2d_row_broadcast")


def test_equal_3d_channel_broadcast():
    x = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    y = torch.tensor([0, 5, 10], dtype=torch.float32).view(1, 3, 1)
    module = EqualTest(y)
    compare_torch_and_backend(module, x, name="equal_3d_channel_broadcast")


def test_equal_scalar_to_4d():
    x = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    y = torch.tensor(7.0, dtype=torch.float32)
    module = EqualTest(y)
    compare_torch_and_backend(module, x, name="equal_scalar_to_4d")


class WhereTest(nn.Module):
    def __init__(self, y, z):
        super().__init__()
        self.register_buffer("y", y)
        self.register_buffer("z", z)

    def forward(self, x):
        # x 是 cond
        return torch.where(x, self.y, self.z)


def test_where_1d_basic():
    cond = torch.tensor([True, False, True, False])
    y = torch.tensor([1.0, 1.0, 1.0, 1.0])
    z = torch.tensor([0.0, 0.0, 0.0, 0.0])

    module = WhereTest(y, z)
    compare_torch_and_backend(module, cond, name="where_1d_basic")


def test_where_1d_scalar_yz():
    cond = torch.tensor([True, False, True, False])
    y = torch.tensor(10.0)
    z = torch.tensor(-1.0)

    module = WhereTest(y, z)
    compare_torch_and_backend(module, cond, name="where_1d_scalar_yz")


def test_where_2d_channel_broadcast():
    cond = torch.tensor([[True, False, True],
                         [False, True, False]])
    y = torch.tensor([1.0, 2.0, 3.0]).view(1, 3)
    z = torch.zeros(1, 3)

    module = WhereTest(y, z)
    compare_torch_and_backend(module, cond, name="where_2d_channel_broadcast")


def test_where_3d_mixed_broadcast():
    cond = torch.tensor([[[True], [False], [True]]])  # (1,3,1)
    y = torch.arange(3, dtype=torch.float32).view(1, 3, 1)
    z = torch.tensor(100.0)

    module = WhereTest(y, z)
    compare_torch_and_backend(module, cond, name="where_3d_mixed_broadcast")


class EqualWhereTest(nn.Module):
    def __init__(self, threshold, y, z):
        super().__init__()
        self.register_buffer("threshold", threshold)
        self.register_buffer("y", y)
        self.register_buffer("z", z)

    def forward(self, x: torch.Tensor):
        cond = x == self.threshold
        return torch.where(cond, self.y, self.z)


def test_equal_where_1d():
    x = torch.arange(8, dtype=torch.float32)
    threshold = torch.tensor(3.0)
    y = torch.tensor(1.0)
    z = torch.tensor(0.0)

    module = EqualWhereTest(threshold, y, z)
    compare_torch_and_backend(module, x, name="equal_where_1d")


def test_equal_where_2d_channel():
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    threshold = torch.tensor([0, 5, 6, 7], dtype=torch.float32).view(1, 4)
    y = torch.ones(1, 4)
    z = torch.zeros(1, 4)

    module = EqualWhereTest(threshold, y, z)
    compare_torch_and_backend(module, x, name="equal_where_2d_channel")


def test_equal_where_4d_full():
    x = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
    threshold = torch.tensor(10.0)
    y = torch.ones(1, 1, 1, 1)
    z = torch.zeros(1, 1, 1, 1)

    module = EqualWhereTest(threshold, y, z)
    compare_torch_and_backend(module, x, name="equal_where_4d_full")


def run_all_logical_tests():
    tests = [
        # equal
        test_equal_1d_same_shape,
        test_equal_1d_scalar,
        test_equal_2d_broadcast_row,
        test_equal_3d_channel_broadcast,
        test_equal_scalar_to_4d,

        # where
        test_where_1d_basic,
        test_where_1d_scalar_yz,
        test_where_2d_channel_broadcast,
        test_where_3d_mixed_broadcast,

        # equal + where
        test_equal_where_1d,
        test_equal_where_2d_channel,
        test_equal_where_4d_full,
    ]

    for fn in tests:
        print(f"\n=== Running {fn.__name__} ===")
        fn()


if __name__ == "__main__":
    run_all_logical_tests()
