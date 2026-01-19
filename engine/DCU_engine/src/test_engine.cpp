#include <iostream>
#include <vector>
#include <cstdlib>  // rand()
#include "dcu_context.h"
#include "dcu_tensor.h"
#include "dcu_engine.h"

int main() {
    std::cout << "=== DCU Engine C++ Layer Test ===" << std::endl;

    // -----------------------------
    // 初始化 DCU 上下文和 Engine
    // -----------------------------
    dcu::DCUContext ctx;       // 管理 HIP stream 和 MIOpen handle
    dcu::DCUEngine engine;     // 封装算子调用

    // -----------------------------
    // 测试 Add 算子
    // -----------------------------
    {int n = 16;
    std::vector<float> ha(n), hb(n);
    for (int i = 0; i < n; i++) {
        ha[i] = i;
        hb[i] = 2 * i;
    }

    dcu::DCUTensor A({n});
    dcu::DCUTensor B({n});
    dcu::DCUTensor O({n});

    CHECK_HIP(hipMemcpy(A.data(), ha.data(), n * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(B.data(), hb.data(), n * sizeof(float), hipMemcpyHostToDevice));

    engine.add(&A, &B, &O, n, &ctx);

    std::vector<float> ho(n);
    CHECK_HIP(hipMemcpy(ho.data(), O.data(), n * sizeof(float), hipMemcpyDeviceToHost));

    std::cout << "[Add] ";
    for (int i = 0; i < 5; i++) std::cout << ho[i] << " ";
    std::cout << std::endl;}

    // -----------------------------
    // 测试 MatMul 算子
    // -----------------------------
    {
    int M = 4, K = 3, N = 5;
    size_t a_size = M * K;
    size_t b_size = K * N;
    size_t c_size = M * N;

    srand(0);  // 固定随机种子

    std::vector<float> host_A(a_size);
    std::vector<float> host_B(b_size);

    for (size_t i = 0; i < a_size; ++i)
        host_A[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    for (size_t i = 0; i < b_size; ++i)
        host_B[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    // ---------- DCU Tensor ----------
    dcu::DCUTensor A({M, K});
    dcu::DCUTensor B({K, N});
    dcu::DCUTensor C({M, N});

    CHECK_HIP(hipMemcpy(A.data(), host_A.data(),
                        a_size * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(B.data(), host_B.data(),
                        b_size * sizeof(float),
                        hipMemcpyHostToDevice));

    engine.matmul(&A, &B, &C, M, N, K, &ctx);
    std::vector<float> host_C(c_size);
    CHECK_HIP(hipMemcpy(host_C.data(), C.data(),
                        c_size * sizeof(float),
                        hipMemcpyDeviceToHost));
    std::cout << "[MatMul] ";
    for (int i = 0; i < std::min(10, (int)c_size); ++i)
        std::cout << host_C[i] << " ";
    std::cout << std::endl;
    }

    // -----------------------------
    // 测试 Relu 算子
    // -----------------------------
    {int n = 8;
    std::vector<float> hx = {-4, -2, 0, 1, 3, -1, 2, 5};

    dcu::DCUTensor X({n});
    dcu::DCUTensor Y({n});

    CHECK_HIP(hipMemcpy(X.data(), hx.data(), n * sizeof(float), hipMemcpyHostToDevice));

    engine.relu(&X, &Y, n, &ctx);

    std::vector<float> hy(n);
    CHECK_HIP(hipMemcpy(hy.data(), Y.data(), n * sizeof(float), hipMemcpyDeviceToHost));

    std::cout << "[ReLU] ";
    for (float v : hy) std::cout << v << " ";
    std::cout << std::endl;}

    // -----------------------------
    // 测试 Transpose 算子
    // -----------------------------
    {int rows = 2, cols = 4;
    std::vector<float> hx = {
        1,2,3,4,
        5,6,7,8
    };

    dcu::DCUTensor X({rows, cols});
    dcu::DCUTensor Y({cols, rows});

    CHECK_HIP(hipMemcpy(X.data(), hx.data(),
        rows * cols * sizeof(float), hipMemcpyHostToDevice));

    engine.transpose(&X, &Y, rows, cols, &ctx);

    std::vector<float> hy(rows * cols);
    CHECK_HIP(hipMemcpy(hy.data(), Y.data(),
        rows * cols * sizeof(float), hipMemcpyDeviceToHost));

    std::cout << "[Transpose] ";
    for (float v : hy) std::cout << v << " ";
    std::cout << std::endl;}

    // -----------------------------
    // 测试 MulScalar 算子
    // -----------------------------
    {int n = 8;
    float s = 3.0f;
    std::vector<float> hx(n);
    for (int i = 0; i < n; i++) hx[i] = i;

    dcu::DCUTensor X({n});
    dcu::DCUTensor Y({n});

    CHECK_HIP(hipMemcpy(X.data(), hx.data(), n * sizeof(float), hipMemcpyHostToDevice));

    engine.mul_scalar(&X, &Y, s, n, &ctx);

    std::vector<float> hy(n);
    CHECK_HIP(hipMemcpy(hy.data(), Y.data(), n * sizeof(float), hipMemcpyDeviceToHost));

    std::cout << "[MulScalar] ";
    for (float v : hy) std::cout << v << " ";
    std::cout << std::endl;}

    // -----------------------------
    // 测试 Conv2D算子
    // -----------------------------
    {int batch = 1, in_channels = 3, in_h = 32, in_w = 32;
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
    for (size_t i = 0; i < x_size; i++) host_x[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    for (size_t i = 0; i < w_size; i++) host_w[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    dcu::DCUTensor d_x({batch, in_channels, in_h, in_w});
    dcu::DCUTensor d_w({out_channels, in_channels/groups, kernel_h, kernel_w});
    dcu::DCUTensor d_y({batch, out_channels, in_h, in_w});

    CHECK_HIP(hipMemcpy(d_x.data(), host_x.data(), x_size * sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_w.data(), host_w.data(), w_size * sizeof(float), hipMemcpyHostToDevice));

    engine.conv2d(&d_x, &d_w, &d_y,
                          batch, in_channels, in_h, in_w,
                          out_channels, kernel_h, kernel_w,
                          stride_h, stride_w,
                          pad_h, pad_w,
                          dilation_h, dilation_w,
                          groups,
                          &ctx);

    std::vector<float> host_y(y_size);
    CHECK_HIP(hipMemcpy(host_y.data(), d_y.data(), y_size * sizeof(float), hipMemcpyDeviceToHost));

    std::cout << "[Conv2d] ";
    for (int i = 0; i < std::min(10, (int)y_size); i++) {
        std::cout << host_y[i] << " ";
    }
    std::cout << std::endl;
}
    // -----------------------------
    // 测试 MaxPool2D 算子
    // -----------------------------
    {int N = 1, C = 1, H = 4, W = 4;
    std::vector<float> hx = {
        1,2,3,4,
        5,6,7,8,
        1,3,5,7,
        2,4,6,8
    };

    dcu::DCUTensor X({N, C, H, W});
    dcu::DCUTensor Y({N, C, 2, 2});

    CHECK_HIP(hipMemcpy(X.data(), hx.data(),
        hx.size() * sizeof(float), hipMemcpyHostToDevice));

    engine.max_pool2d(&X, &Y,
        N, C, H, W,
        2, 2, 2, 2, 0, 0,
        &ctx);

    std::vector<float> hy(4);
    CHECK_HIP(hipMemcpy(hy.data(), Y.data(), 4 * sizeof(float), hipMemcpyDeviceToHost));

    std::cout << "[MaxPool] ";
    for (float v : hy) std::cout << v << " ";
    std::cout << std::endl;}

    // -----------------------------
    // 测试 Global Avg Pool 算子
    // -----------------------------
    {int N = 1, C = 2, H = 2, W = 2;
    std::vector<float> hx = {
        1,2,3,4,
        5,6,7,8
    };

    dcu::DCUTensor X({N, C, H, W});
    dcu::DCUTensor Y({N, C});

    CHECK_HIP(hipMemcpy(X.data(), hx.data(),
        hx.size() * sizeof(float), hipMemcpyHostToDevice));

    engine.global_avg_pool(&X, &Y, N, C, H, W, &ctx);

    std::vector<float> hy(C);
    CHECK_HIP(hipMemcpy(hy.data(), Y.data(), C * sizeof(float), hipMemcpyDeviceToHost));

    std::cout << "[GAP] ";
    for (float v : hy) std::cout << v << " ";
    std::cout << std::endl;}

    // -----------------------------
    // 测试 Flatten 算子
    // -----------------------------
    {int outer = 2, inner = 6;
    std::vector<float> hx(outer * inner);
    for (int i = 0; i < outer * inner; i++) hx[i] = i;

    dcu::DCUTensor X({outer, inner});
    dcu::DCUTensor Y({outer * inner});

    CHECK_HIP(hipMemcpy(X.data(), hx.data(),
        hx.size() * sizeof(float), hipMemcpyHostToDevice));

    engine.flatten(&X, &Y, outer, inner, &ctx);

    std::vector<float> hy(hx.size());
    CHECK_HIP(hipMemcpy(hy.data(), Y.data(),
        hy.size() * sizeof(float), hipMemcpyDeviceToHost));

    std::cout << "[Flatten] ";
    for (float v : hy) std::cout << v << " ";
    std::cout << std::endl;}

    // -----------------------------
    // 测试 BatchNorm 算子
    // -----------------------------
    {int N = 1, C = 2, H = 2, W = 2;
    int size = N*C*H*W;

    std::vector<float> hx(size, 1.0f);
    std::vector<float> mean = {1.0f, 1.0f};
    std::vector<float> var  = {1.0f, 1.0f};
    std::vector<float> w    = {1.0f, 1.0f};
    std::vector<float> b    = {0.0f, 0.0f};

    dcu::DCUTensor X({N,C,H,W});
    dcu::DCUTensor Y({N,C,H,W});
    dcu::DCUTensor Mean({C});
    dcu::DCUTensor Var({C});
    dcu::DCUTensor Wt({C});
    dcu::DCUTensor Bs({C});

    CHECK_HIP(hipMemcpy(X.data(), hx.data(), size*sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(Mean.data(), mean.data(), C*sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(Var.data(), var.data(), C*sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(Wt.data(), w.data(), C*sizeof(float), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(Bs.data(), b.data(), C*sizeof(float), hipMemcpyHostToDevice));

    engine.batch_norm_2d(&X, &Y, &Wt, &Bs, &Mean, &Var,
                         N, C, H, W, 1e-5f, &ctx);

    std::vector<float> hy(size);
    CHECK_HIP(hipMemcpy(hy.data(), Y.data(), size*sizeof(float), hipMemcpyDeviceToHost));

    std::cout << "[BatchNorm] ";
    for (float v : hy) std::cout << v << " ";
    std::cout << std::endl;}

    std::cout << "=== Test Finished ===" << std::endl;
    return 0;
}
