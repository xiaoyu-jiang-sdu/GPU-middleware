from test.add_test import test_add
from test.batchnorm2d_test import test_batch_norm
from test.conv2d_test import test_conv2d
from test.flatten_test import test_flatten
from test.GAP_test import test_GAP
from test.linear_test import test_linear
from test.matmul_test import run_all_matmul_tests
from test.maxpool2d_test import test_max_pool
from test.relu_test import test_relu
from test.residual_block_test import test_residual_block
from test.add_broadcast_test import run_all_broadcast_tests
from test.tensor_test import test_tensor
from test.transform_test import run_all_transform_tests

if __name__ == '__main__':
    test_tensor()
    test_add()
    test_batch_norm()
    test_conv2d()
    test_flatten()
    test_GAP()
    test_linear()
    test_max_pool()
    test_relu()
    test_residual_block()
    run_all_broadcast_tests()
    run_all_matmul_tests()
    run_all_transform_tests()
    print("Test Succeed!")
