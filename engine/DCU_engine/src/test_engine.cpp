#include <iostream>
#include <vector>
#include <cstdlib>  // rand()
#include "dcu_context.h"
#include "dcu_tensor.h"
#include "dcu_engine.h"

#define CHECK_HIP(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(e) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

int main() {
    std::cout << "=== DCU Engine C++ Layer Test ===" << std::endl;

    // -----------------------------
    // 1. 初始化 DCU 上下文和 Engine
    // -----------------------------
    dcu::DCUContext ctx;       // 管理 HIP stream 和 MIOpen handle
    dcu::DCUEngine engine;     // 封装算子调用

    // -----------------------------
    // 2. 测试 MatMul 算子
    // -----------------------------
    int M = 4, K = 3, N = 5;
    size_t a_size = M*K;
    size_t b_size = K*N;
    size_t c_size = M*N;

    std::vector<float> host_A(a_size);
    std::vector<float> host_B(b_size);

    for (size_t i = 0; i < a_size; i++) host_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (size_t i = 0; i < b_size; i++) host_B[i] = static_cast<float>(rand()) / RAND_MAX;

    dcu::DCUTensor A({M, K});
    dcu::DCUTensor B({K, N});
    dcu::DCUTensor C({M, N});

    CHECK_HIP(hipMemcpy(A.data(), host_A.data(), a_size * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(B.data(), host_B.data(), b_size * sizeof(float), hipMemcpyHostToDevice));

    engine.matmul(&A, &B, &C, M, N, K, &ctx);

    std::vector<float> host_C(c_size);
    CHECK_HIP(hipMemcpy(host_C.data(), C.data(), c_size * sizeof(float), hipMemcpyDeviceToHost));

    std::cout << "--- MatMul Output Sample ---" << std::endl;
    for (int i = 0; i < std::min(10, (int)c_size); i++) {
        std::cout << host_C[i] << " ";
    }
    std::cout << std::endl;

    // -----------------------------
    // 3. 测试 Conv2D Forward 算子
    // -----------------------------
    int batch = 1, in_channels = 3, in_h = 32, in_w = 32;
    int out_channels = 8, kernel_h = 3, kernel_w = 3;
    int stride_h = 1, stride_w = 1;
    int pad_h = 1, pad_w = 1;
    int dilation_h = 1, dilation_w = 1;
    int groups = 1;

    size_t x_size = batch*in_channels*in_h*in_w;
    size_t w_size = out_channels*in_channels/groups*kernel_h*kernel_w;
    size_t y_size = batch*out_channels*in_h*in_w;  // assuming same padding

    std::vector<float> host_x(x_size);
    std::vector<float> host_w(w_size);
    for (size_t i = 0; i < x_size; i++) host_x[i] = static_cast<float>(rand()) / RAND_MAX;
    for (size_t i = 0; i < w_size; i++) host_w[i] = static_cast<float>(rand()) / RAND_MAX;

    dcu::DCUTensor d_x({batch, in_channels, in_h, in_w});
    dcu::DCUTensor d_w({out_channels, in_channels/groups, kernel_h, kernel_w});
    dcu::DCUTensor d_y({batch, out_channels, in_h, in_w});

    CHECK_HIP(hipMemcpy(d_x.data(), host_x.data(), x_size * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_w.data(), host_w.data(), w_size * sizeof(float), hipMemcpyHostToDevice));

    engine.conv2d_forward(&d_x, &d_w, &d_y,
                          batch, in_channels, in_h, in_w,
                          out_channels, kernel_h, kernel_w,
                          stride_h, stride_w,
                          pad_h, pad_w,
                          dilation_h, dilation_w,
                          groups,
                          &ctx);

    std::vector<float> host_y(y_size);
    CHECK_HIP(hipMemcpy(host_y.data(), d_y.data(), y_size * sizeof(float), hipMemcpyDeviceToHost));

    std::cout << "--- Conv2D Output Sample ---" << std::endl;
    for (int i = 0; i < std::min(10, (int)y_size); i++) {
        std::cout << host_y[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "=== Test Finished ===" << std::endl;

    return 0;
}
