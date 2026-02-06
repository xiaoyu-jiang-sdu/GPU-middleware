import torch
import torch.nn as nn

from test.test_engine import compare_torch_and_backend


class CastTest(nn.Module):
    def __init__(self, to_dtype):
        super().__init__()
        self.to_dtype = to_dtype

    def forward(self, x):
        return x.to(self.to_dtype)


def test_cast_bool_to_float():
    x = torch.tensor([True, False, True, False])
    module = CastTest(torch.float32)
    compare_torch_and_backend(module, x, name="cast_bool_to_float")


def test_cast_bool_to_int():
    x = torch.tensor([True, False, True])
    module = CastTest(torch.int32)
    compare_torch_and_backend(module, x, name="cast_bool_to_int")


def test_cast_float_to_bool():
    x = torch.tensor([0.0, 1.0, -1.0, 2.5], dtype=torch.float32)
    module = CastTest(torch.bool)
    compare_torch_and_backend(module, x, name="cast_float_to_bool")


def test_cast_int_to_float():
    x = torch.tensor([1, 2, 3, 4], dtype=torch.int32)
    module = CastTest(torch.float32)
    compare_torch_and_backend(module, x, name="cast_int_to_float")


def test_cast_scalar_float_to_bool():
    x = torch.tensor(0.0)
    module = CastTest(torch.bool)
    compare_torch_and_backend(module, x, name="cast_scalar_float_to_bool")


class CastEqualTest(nn.Module):
    def forward(self, x):
        x = x.to(torch.float32)
        return x == 1.0


def test_cast_equal_chain():
    x = torch.tensor([True, False, True])
    module = CastEqualTest()
    compare_torch_and_backend(module, x, name="cast_equal_chain")


class CastWhereTest(nn.Module):
    def forward(self, x):
        cond = x.to(torch.bool)
        return torch.where(cond, torch.tensor(1.0), torch.tensor(0.0))


def test_cast_where_chain():
    x = torch.tensor([0.0, 1.0, -1.0, 2.0])
    module = CastWhereTest()
    compare_torch_and_backend(module, x, name="cast_where_chain")


def run_all_cast_tests():
    tests = [
        test_cast_bool_to_float,
        test_cast_bool_to_int,
        test_cast_float_to_bool,
        test_cast_int_to_float,
        test_cast_scalar_float_to_bool,
        test_cast_equal_chain,
        test_cast_where_chain,
    ]

    for fn in tests:
        print(f"\n=== Running {fn.__name__} ===")
        fn()


if __name__ == "__main__":
    run_all_cast_tests()
