#pragma once
#include "tensor.h"
#include "context.h"

namespace engine {

// Engine 基类模板
template <typename T, typename Ctx>
class Engine {
public:
    virtual ~Engine() = default;

    // ------------------------
    // 二元运算算子
    // ------------------------
    virtual void add(T* a, T* b, T* out, Ctx* ctx) = 0;
    virtual void sub(T* a, T* b, T* out, Ctx* ctx) = 0;
    virtual void mul(T* a, T* b, T* out, Ctx* ctx) = 0;
    virtual void div(T* a, T* b, T* out, Ctx* ctx) = 0;
    virtual void pow(T* a, T* b, T* out, Ctx* ctx) = 0;
    virtual void mod(T* a, T* b, T* out, Ctx* ctx) = 0;
    virtual void mul_scalar(T* x, T* out, float s, Ctx* ctx) = 0;

    // ------------------------
    // logical 算子
    // ------------------------
    virtual void equal(T* a, T* b, T* out, Ctx* ctx) = 0;
    virtual void not_equal(T* a, T* b, T* out, Ctx* ctx) = 0;
    virtual void where(T* cond, T* x, T* y, T* out, Ctx* ctx) = 0;

    // ------------------------
    // NN算子
    // ------------------------
    virtual void matmul(T* A, T* B, T* Out,
                                bool transA, bool transB,
                                float alpha, float beta,
                                Ctx* ctx) = 0;
    virtual void conv2d(T* x, T* w, T* y,
                                int N, int C, int H, int W,
                                int K, int R, int S,
                                int stride_h, int stride_w,
                                int pad_h, int pad_w,
                                int dilation_h, int dilation_w,
                                int groups,
                                Ctx* ctx) = 0;

    // ------------------------
    // 激活算子
    // ------------------------
    virtual void relu(T* x, T* out, Ctx* ctx) = 0;
    virtual void erf(T* x, T* out, Ctx* ctx) = 0;
    virtual void sqrt(T* x, T* out, Ctx* ctx) = 0;

    // ------------------------
    // 池化算子
    // ------------------------
    virtual void max_pool2d(T* x, T* y,
                                int N, int C, int H, int W,
                                int kH, int kW,
                                int sH, int sW,
                                int pH, int pW,
                                Ctx* ctx) = 0;
    virtual void global_avg_pool(T* x, T* y,
                                int N, int C, int H, int W,
                                Ctx* ctx) = 0;

    // ------------------------
    // 归一化
    // ------------------------
    virtual void batch_norm_2d(T* x, T* y,
                                T* weight, T* bias,
                                T* running_mean, T* running_var,
                                int N, int C, int H, int W,
                                float eps,
                                Ctx* ctx) = 0;

    // ------------------------
    // shape view
    // ------------------------
    virtual std::shared_ptr<T> transpose(std::shared_ptr<T> x, const std::vector<int>& perm, Ctx* ctx) = 0;
    virtual std::shared_ptr<T> unsqueeze(std::shared_ptr<T> x, const std::vector<int>& axes, Ctx* ctx) = 0;
    virtual std::shared_ptr<T> reshape(std::shared_ptr<T> x, const std::vector<int>& shape, Ctx* ctx) = 0;

    // ------------------------
    // transform
    // ------------------------
    virtual void flatten(T* x, T* y,
                         int axis,
                         Ctx* ctx) = 0;
    virtual T* concat(const std::vector<T*>& inputs, int axis, Ctx* ctx) = 0;

    // ------------------------
    // type cast
    // ------------------------
    virtual void cast(T* x, T* out, Ctx* ctx) = 0;
};

} // namespace engine
