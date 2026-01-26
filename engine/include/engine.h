#pragma once
#include "tensor.h"
#include "context.h"

namespace engine {

// Engine 基类模板
template <typename T, typename Ctx>
class Engine {
public:
    virtual ~Engine() = default;

    // 基础算子
    virtual void add(T* a, T* b, T* out,
                        const std::vector<int>& a_shape,
                        const std::vector<int>& b_shape,
                        Ctx* ctx) = 0;

    virtual void matmul(T* A, T* B, T* Out, int M, int N, int K, Ctx* ctx) = 0;
    virtual void relu(T* x, T* out, int n, Ctx* ctx) = 0;
    virtual void transpose(T* x, T* y, int rows, int cols, Ctx* ctx) = 0;
    virtual void mul_scalar(T* x, T* out, float s, int n, Ctx* ctx) = 0;

    // CNN算子
    virtual void conv2d(T* x, T* w, T* y,
                                int N, int C, int H, int W,
                                int K, int R, int S,
                                int stride_h, int stride_w,
                                int pad_h, int pad_w,
                                int dilation_h, int dilation_w,
                                int groups,
                                Ctx* ctx) = 0;
    virtual void max_pool2d(T* x, T* y,
                                int N, int C, int H, int W,
                                int kH, int kW,
                                int sH, int sW,
                                int pH, int pW,
                                Ctx* ctx) = 0;
    virtual void global_avg_pool(T* x, T* y,
                                int N, int C, int H, int W,
                                Ctx* ctx) = 0;

    virtual void flatten(T* x, T* y,
                                int outer, int inner,
                                Ctx* ctx) = 0;
    // 归一化
    virtual void batch_norm_2d(T* x, T* y,
                                T* weight, T* bias,
                                T* running_mean, T* running_var,
                                int N, int C, int H, int W,
                                float eps,
                                Ctx* ctx) = 0;
};

} // namespace engine
