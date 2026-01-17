#pragma once
#include "tensor.h"
#include "context.h"

namespace engine {

// Engine 基类模板
template <typename T, typename Ctx>
class Engine {
public:
    virtual ~Engine() = default;

    // 矩阵乘法
    virtual void matmul(T* A, T* B, T* Out, int M, int N, int K, Ctx* ctx) = 0;

    // 卷积
    virtual void conv2d_forward(T* x, T* w, T* y,
                                int N, int C, int H, int W,
                                int K, int R, int S,
                                int stride_h, int stride_w,
                                int pad_h, int pad_w,
                                int dilation_h, int dilation_w,
                                int groups,
                                Ctx* ctx) = 0;
};

} // namespace engine
