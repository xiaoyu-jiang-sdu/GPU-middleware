#include <iostream>
#include <vector>
#include <cstdlib>  // for rand()
#include "dcu/dcu_context.h"
#include "dcu/dcu_tensor.h"
#include "dcu/dcu_engine.h"

#define CHECK_HIP(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(e) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)

int main() {
    try {
        std::cout << "=== DCU Engine C++ Layer Test ===" << std::endl;

        // -----------------------------
        // 1. 初始化 DCU 上下文和 Engine
        // -----------------------------
        dcu::DCUContext ctx;       // 管理 HIP stream 和 MIOpen handle
        dcu::DCUEngine engine;     // 封装算子调用

        // -----------------------------
        // 2. 定义输入输出张量维度
        // -----------------------------
        int N = 1, C = 3, H = 32, W = 32;
        int K = 8, R = 3, S = 3;
        int stride_h = 1, stride_w = 1;
        int pad_h = 1, pad_w = 1;

        size_t x_size = N * C * H * W;
        size_t w_size = K * C * R * S;
        size_t y_size = N * K * H * W;

        // -----------------------------
        // 3. 初始化 host 数据
        // -----------------------------
        std::vector<float> host_x(x_size);
        std::vector<float> host_w(w_size);

        for (size_t i = 0; i < x_size; i++) host_x[i] = static_cast<float>(rand()) / RAND_MAX;
        for (size_t i = 0; i < w_size; i++) host_w[i] = static_cast<float>(rand()) / RAND_MAX;

        // -----------------------------
        // 4. 分配 device 张量
        // -----------------------------
        dcu::DCUTensor d_x({N, C, H, W});
        dcu::DCUTensor d_w({K, C, R, S});
        dcu::DCUTensor d_y({N, K, H, W});  // 输出张量

        // 拷贝数据到 GPU
        CHECK_HIP(hipMemcpy(d_x.data(), host_x.data(), x_size * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_w.data(), host_w.data(), w_size * sizeof(float), hipMemcpyHostToDevice));

        // -----------------------------
        // 5. Conv2D Forward 测试
        // -----------------------------
        engine.conv2d_forward(
            static_cast<float*>(d_x.data()),
            static_cast<float*>(d_w.data()),
            static_cast<float*>(d_y.data()),
            N, C, H, W,
            K, R, S,
            stride_h, stride_w,
            pad_h, pad_w,
            ctx.get_handle()  // MIOpen handle
        );

        std::cout << "Conv2D forward completed." << std::endl;

        // -----------------------------
        // 6. 拷贝输出回 host 检查
        // -----------------------------
        std::vector<float> host_y(y_size);
        CHECK_HIP(hipMemcpy(host_y.data(), d_y.data(), y_size * sizeof(float), hipMemcpyDeviceToHost));

        std::cout << "Output sample: ";
        for (int i = 0; i < 10 && i < y_size; i++) {
            std::cout << host_y[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "=== Test Finished ===" << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
