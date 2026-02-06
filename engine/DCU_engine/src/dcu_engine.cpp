#include "dcu_engine.h"
#include "dcu_common.h"
#include "device_array.h"
#include "dtype.h"
#include "dcu_op_kernel.h"
#include "dispatch/miopen_traits.h"
#include <miopen/miopen.h>
#include <hip/hip_runtime.h>
#include <cstdlib>

namespace dcu {
// ------------------------
// 返回连续tensor的shared_ptr
// ------------------------
std::shared_ptr<DCUTensor> DCUEngine::to_contiguous(
    const std::shared_ptr<DCUTensor>& x, DCUContext* ctx)
{
    if (!x) {
        throw std::runtime_error("to_contiguous: input is nullptr");
    }

    // 已经连续，直接返回
    if (x->is_contiguous()) {
        return x;
    }

    // 创建新的 contiguous tensor
    auto y = std::make_shared<DCUTensor>(x->shape(), x->dtype());
    unary_op_impl<CopyOp>(x, y, ctx);
    return y;
}

// 简化写法，确保某个张量连续，返回shared_ptr
inline std::shared_ptr<DCUTensor>
ensure_contiguous(DCUTensor* t, DCUContext* ctx)
{
    auto sp = std::shared_ptr<DCUTensor>(t, [](DCUTensor*) {});
    return DCUEngine::to_contiguous(sp, ctx);
}

// ------------------------
// 二元运算实现，广播Op
// ------------------------
void DCUEngine::add(DCUTensor* a, DCUTensor* b,
                    DCUTensor* out, DCUContext* ctx) {
    binary_op_impl<AddOp>(a, b, out, ctx);
}

void DCUEngine::sub(DCUTensor* a, DCUTensor* b,
                    DCUTensor* out, DCUContext* ctx) {
    binary_op_impl<SubOp>(a, b, out, ctx);
}

void DCUEngine::mul(DCUTensor* a, DCUTensor* b,
                    DCUTensor* out, DCUContext* ctx) {
    binary_op_impl<MulOp>(a, b, out, ctx);
}

void DCUEngine::div(DCUTensor* a, DCUTensor* b,
                    DCUTensor* out, DCUContext* ctx) {
    binary_op_impl<DivOp>(a, b, out, ctx);
}

void DCUEngine::pow(DCUTensor* a, DCUTensor* b,
                    DCUTensor* out, DCUContext* ctx) {
    binary_op_impl<PowOp>(a, b, out, ctx);
}

void DCUEngine::mod(DCUTensor* a, DCUTensor* b,
                    DCUTensor* out, DCUContext* ctx) {
    binary_op_impl<ModOp>(a, b, out, ctx);
}

// ------------------------
// mul_scalar 实现
// ------------------------
void DCUEngine::mul_scalar(DCUTensor* x, DCUTensor* o, float s, DCUContext* ctx)
{
    unary_scalar_op_impl<MulScalarOp>(
        x, o, s, ctx
    );
}

// ------------------------
// equal 实现
// ------------------------
void DCUEngine::equal(DCUTensor* a, DCUTensor* b, DCUTensor* out, DCUContext* ctx)
{
    binary_op_impl<EqualOp>(a, b, out, ctx);
}

// ------------------------
// not_equal 实现
// ------------------------
void DCUEngine::not_equal(DCUTensor* a, DCUTensor* b, DCUTensor* out, DCUContext* ctx)
{
    binary_op_impl<NotEqualOp>(a, b, out, ctx);
}


// ------------------------
// where 实现
// ------------------------
void DCUEngine::where(DCUTensor* cond, DCUTensor* x, DCUTensor* y, DCUTensor* out, DCUContext* ctx)
{
    DISPATCH_DTYPE(x->dtype(), [&](auto dtype_enum) {
        using T = typename engine::CType<decltype(dtype_enum)::value>::type;
        where_op_impl<T>(cond, x, y, out, ctx);
    });
}

// ------------------------
// matmul 实现
// ------------------------
void DCUEngine::matmul(DCUTensor* A, DCUTensor* B, DCUTensor* Out,
                       bool transA, bool transB,
                       float alpha, float beta,
                       DCUContext* ctx)
{
    // 确保连续
    auto A_sp = ensure_contiguous(A, ctx);
    auto B_sp = ensure_contiguous(B, ctx);

    // dispatch到不同类型的matmul 模板上执行
    DISPATCH_FLOAT_ONLY(A->dtype(), ([&](auto dtype_enum){
        using T = typename engine::CType<decltype(dtype_enum)::value>::type;
        matmul_template<T>(
            A_sp.get(),
            B_sp.get(),
            Out,
            transA,
            transB,
            alpha,
            beta,
            ctx
        );
    }));
}
// ------------------------
// Relu 实现
// ------------------------
void DCUEngine::relu(DCUTensor* x, DCUTensor* o, DCUContext* ctx) {
    unary_op_impl<ReluOp>(
        x, o, ctx
    );
}

// ------------------------
// Erf 实现
// ------------------------
void DCUEngine::erf(DCUTensor* x, DCUTensor* o, DCUContext* ctx) {
    unary_op_impl<ErfOp>(
        x, o, ctx
    );
}

// ------------------------
// Sqrt 实现
// ------------------------
void DCUEngine::sqrt(DCUTensor* x, DCUTensor* o, DCUContext* ctx) {
    unary_op_impl<SqrtOp>(
        x, o, ctx
    );
}

// ------------------------
// conv2d 实现
// ------------------------
void DCUEngine::conv2d(DCUTensor* x, DCUTensor* w, DCUTensor* y,
                       int N, int C, int H, int W,
                       int K, int R, int S,
                       int stride_h, int stride_w,
                       int pad_h, int pad_w,
                       int dilation_h, int dilation_w,
                       int groups,
                       DCUContext* ctx)
{
    // 确保x与w连续
    auto x_c = ensure_contiguous(x, ctx);
    auto w_c = ensure_contiguous(w, ctx);

    // x数据类型的大小
    size_t d_size = dtype_size(x->dtype());

    // 只dispatch float32
    DISPATCH_FLOAT_ONLY(x->dtype(), ([&](auto dtype_enum) {
        using Traits = MiopenTraits<decltype(dtype_enum)::value>;
        using T = typename Traits::CType;

        T* x_data = static_cast<T*>(x_c->data());
        T* w_data = static_cast<T*>(w_c->data());
        T* y_data = static_cast<T*>(y->data());

        int Ho = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
        int Wo = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;

        // y设置为0
        CHECK_HIP(hipMemset(y_data, 0, d_size * N * K * Ho * Wo));

        // Tensor 描述符
        miopenTensorDescriptor_t x_desc, w_desc, y_desc;
        miopenCreateTensorDescriptor(&x_desc);
        miopenCreateTensorDescriptor(&w_desc);
        miopenCreateTensorDescriptor(&y_desc);

        // 动态dispatch miopen type
        miopenSet4dTensorDescriptor(x_desc, Traits::miopen_type, N, C, H, W);
        miopenSet4dTensorDescriptor(w_desc, Traits::miopen_type, K, C / groups, R, S);
        miopenSet4dTensorDescriptor(y_desc, Traits::miopen_type, N, K, Ho, Wo);

        // 卷积描述符
        miopenConvolutionDescriptor_t conv_desc;
        miopenCreateConvolutionDescriptor(&conv_desc);
        miopenInitConvolutionDescriptor(conv_desc,
                                        miopenConvolution,
                                        pad_h, pad_w,
                                        stride_h, stride_w,
                                        dilation_h, dilation_w);
        miopenSetConvolutionGroupCount(conv_desc, groups);

        // 构造卷积 key
        ConvKey key{N, C, H, W, K, R, S,
                     stride_h, stride_w,
                     pad_h, pad_w,
                     dilation_h, dilation_w,
                     groups};

        miopenConvFwdAlgorithm_t algo;
        size_t workspace_size = 0;

        // 查缓存
        auto it = ctx->conv_algo_cache_.find(key);
        if (it != ctx->conv_algo_cache_.end()) {
            algo = it->second.algo;
            workspace_size = it->second.workspace_size;
        } else {
            // 第一次查找最优算法
            miopenConvAlgoPerf_t perfResults;
            int returnedAlgoCount = 0;

            // 获取可能的最大 workspace
            size_t max_ws_size = 0;
            miopenConvolutionForwardGetWorkSpaceSize(ctx->get_miopen(),
                                                    x_desc, w_desc, conv_desc, y_desc, &max_ws_size);

            // 若max_ws_size 为 0，强制设置一个保底值，避免 find 阶段警告
            if (max_ws_size == 0) {
                max_ws_size = 64ULL * 1024 * 1024;  // 64MB 保底
            }

            void* workspace = nullptr;
            if (max_ws_size > 0) CHECK_HIP(hipMalloc(&workspace, max_ws_size));

            // 查找最优算法
            miopenFindConvolutionForwardAlgorithm(ctx->get_miopen(),
                                                x_desc, x_data,
                                                w_desc, w_data,
                                                conv_desc,
                                                y_desc, y_data,
                                                1, &returnedAlgoCount,
                                                &perfResults,
                                                workspace, max_ws_size,
                                                false);

            algo = perfResults.fwd_algo;
            workspace_size = perfResults.memory;

            // 重新查询实际最大 workspace
            size_t real_ws = 0;
            miopenConvolutionForwardGetWorkSpaceSize(ctx->get_miopen(),
                                                    x_desc, w_desc, conv_desc, y_desc, &real_ws);
            workspace_size = std::max(workspace_size, real_ws);
            if (workspace_size == 0) {
                workspace_size = 64ULL * 1024 * 1024;  // 如果仍为 0，设置保底
            }

            // 缓存算法
            ctx->conv_algo_cache_[key] = {algo, workspace_size};

            if (workspace) CHECK_HIP(hipFree(workspace));
        }

        // 分配需要的 workspace 并执行卷积
        void* workspace = nullptr;
        if (workspace_size > 0) {
            CHECK_HIP(hipMalloc(&workspace, workspace_size));
        }

        float alpha = 1.0f, beta = 0.0f;
        miopenConvolutionForward(ctx->get_miopen(),
                                &alpha,
                                x_desc, x_data,
                                w_desc, w_data,
                                conv_desc,
                                algo,
                                &beta,
                                y_desc, y_data,
                                workspace, workspace_size);

        if (workspace) CHECK_HIP(hipFree(workspace));

        // 清理描述符
        miopenDestroyTensorDescriptor(x_desc);
        miopenDestroyTensorDescriptor(w_desc);
        miopenDestroyTensorDescriptor(y_desc);
        miopenDestroyConvolutionDescriptor(conv_desc);
    }));
}
// ------------------------
// max_pool2d 实现
// ------------------------
void DCUEngine::max_pool2d(DCUTensor* x, DCUTensor* y,
                                int N, int C, int H, int W,
                                int kH, int kW,
                                int sH, int sW,
                                int pH, int pW,
                                DCUContext* ctx)
{
    auto x_c = ensure_contiguous(x, ctx);

    // 只dispatch float32
    DISPATCH_FLOAT_ONLY(x->dtype(), ([&](auto dtype_enum) {
        using Traits = MiopenTraits<decltype(dtype_enum)::value>;
        using T = typename Traits::CType;

        miopenTensorDescriptor_t x_desc, y_desc;
        miopenCreateTensorDescriptor(&x_desc);
        miopenCreateTensorDescriptor(&y_desc);

        int outH = (H + 2 * pH - kH) / sH + 1;
        int outW = (W + 2 * pW - kW) / sW + 1;

        miopenSet4dTensorDescriptor(x_desc, Traits::miopen_type, N, C, H, W);
        miopenSet4dTensorDescriptor(y_desc, Traits::miopen_type, N, C, outH, outW);

        miopenPoolingDescriptor_t pool_desc;
        miopenCreatePoolingDescriptor(&pool_desc);
        miopenSet2dPoolingDescriptor(
            pool_desc,
            miopenPoolingMax,
            kH, kW,
            pH, pW,
            sH, sW
        );

        float alpha = 1.0f, beta = 0.0f;

        miopenPoolingForward(
            ctx->get_miopen(),
            pool_desc,
            &alpha,
            x_desc, x_c->data(),
            &beta,
            y_desc, y->data(),
            false, nullptr, 0
        );

        miopenDestroyPoolingDescriptor(pool_desc);
        miopenDestroyTensorDescriptor(x_desc);
        miopenDestroyTensorDescriptor(y_desc);
    }));
}

// ------------------------
// global avg pool 实现
// ------------------------
void DCUEngine::global_avg_pool(
    DCUTensor* x, DCUTensor* y,
    int N, int C, int H, int W,
    DCUContext* ctx)
{
    auto x_c = ensure_contiguous(x, ctx);

    // 只dispatch float32
    DISPATCH_FLOAT_ONLY(x->dtype(), ([&](auto dtype_enum) {
        using Traits = MiopenTraits<decltype(dtype_enum)::value>;
        using T = typename Traits::CType;

        miopenTensorDescriptor_t x_desc, y_desc;
        miopenCreateTensorDescriptor(&x_desc);
        miopenCreateTensorDescriptor(&y_desc);

        miopenSet4dTensorDescriptor(x_desc, Traits::miopen_type, N, C, H, W);
        miopenSet4dTensorDescriptor(y_desc, Traits::miopen_type, N, C, 1, 1);

        miopenPoolingDescriptor_t pool_desc;
        miopenCreatePoolingDescriptor(&pool_desc);

        miopenSet2dPoolingDescriptor(
            pool_desc,
            miopenPoolingAverage,
            H, W,
            0, 0,
            1, 1
        );

        float alpha = 1.0f;
        float beta = 0.0f;

        miopenPoolingForward(
            ctx->get_miopen(),
            pool_desc,
            &alpha,
            x_desc, x_c->data(),
            &beta,
            y_desc, y->data(),
            false,
            nullptr,
            0
        );
        miopenDestroyPoolingDescriptor(pool_desc);
        miopenDestroyTensorDescriptor(x_desc);
        miopenDestroyTensorDescriptor(y_desc);
    }));
}



// ------------------------
// flatten 实现
// ------------------------
void DCUEngine::flatten(DCUTensor* x, DCUTensor* y,
                        int axis, DCUContext* ctx) {
    auto x_c = ensure_contiguous(x, ctx);
    int ndim = x->ndim();

    int outer = 1;
    for (int i = 0; i < axis; ++i) outer *= x->shape()[i];

    int inner = 1;
    for (int i = axis; i < ndim; ++i) inner *= x->shape()[i];

    size_t d_size = dtype_size(x->dtype());

    DISPATCH_DTYPE(x->dtype(), ([&](auto dtype_enum){
        using T = typename engine::CType<decltype(dtype_enum)::value>::type;

        CHECK_HIP(hipMemcpy(y->data(), x_c->data(),
            d_size * outer * inner,
            hipMemcpyDeviceToDevice));
    }));
}

// ------------------------
// BatchNorm2d 实现
// ------------------------
void DCUEngine::batch_norm_2d(
    DCUTensor* x, DCUTensor* y,
    DCUTensor* weight, DCUTensor* bias,
    DCUTensor* running_mean, DCUTensor* running_var,
    int N, int C, int H, int W,
    float eps,
    DCUContext* ctx) {
    auto x_c = ensure_contiguous(x, ctx);

    // 只dispatch float32
    DISPATCH_FLOAT_ONLY(x->dtype(), ([&](auto dtype_enum) {
        using Traits = MiopenTraits<decltype(dtype_enum)::value>;
        using T = typename Traits::CType;

        miopenTensorDescriptor_t x_desc, y_desc, bn_desc;
        miopenCreateTensorDescriptor(&x_desc);
        miopenCreateTensorDescriptor(&y_desc);
        miopenCreateTensorDescriptor(&bn_desc);

        miopenSet4dTensorDescriptor(x_desc, Traits::miopen_type, N, C, H, W);
        miopenSet4dTensorDescriptor(y_desc, Traits::miopen_type, N, C, H, W);
        miopenDeriveBNTensorDescriptor(bn_desc, x_desc, miopenBNSpatial);

        float alpha = 1.0f, beta = 0.0f;

        miopenBatchNormalizationForwardInference(
            ctx->get_miopen(),
            miopenBNSpatial,
            &alpha, &beta,
            x_desc, x_c->data(),
            y_desc, y->data(),
            bn_desc,
            weight->data(),
            bias->data(),
            running_mean->data(),
            running_var->data(),
            eps
        );

        miopenDestroyTensorDescriptor(x_desc);
        miopenDestroyTensorDescriptor(y_desc);
        miopenDestroyTensorDescriptor(bn_desc);
    }));
}

// ------------------------
// transpose 实现
// ------------------------
std::shared_ptr<DCUTensor> DCUEngine::transpose(std::shared_ptr<DCUTensor> x,
                                const std::vector<int>& perm,
                                DCUContext* ctx)
{
    if (!x) throw std::runtime_error("transpose: input tensor is nullptr");

    int ndim = x->ndim();
    if ((int)perm.size() != ndim)
        throw std::runtime_error("transpose: perm rank mismatch");

    std::vector<int> new_shape(ndim);
    std::vector<int> new_strides(ndim);

    const auto& old_shape = x->shape();
    const auto& old_strides = x->strides();

    // 重新计算 strides，保证 view 展开顺序正确
    for (int i = 0; i < ndim; ++i) {
        new_shape[i] = old_shape[perm[i]];
        new_strides[i] = old_strides[perm[i]];
    }

    // 创建 view
    auto view = std::make_shared<DCUTensor>(
        x->data(),
        new_shape,
        new_strides,
        x->dtype(),
        false,
        x
    );

    return view;
}


// ------------------------
// unsqueeze 实现
// ------------------------
std::shared_ptr<DCUTensor> DCUEngine::unsqueeze(
    std::shared_ptr<DCUTensor> x,
    const std::vector<int>& axes,
    DCUContext* ctx)
{
    if (!x) throw std::runtime_error("unsqueeze: input tensor is nullptr");

    const auto& in_shape = x->shape();
    const auto& in_strides = x->strides();
    int ndim = x->ndim();

    std::vector<int> norm_axes;
    norm_axes.reserve(axes.size());
    for (int a : axes) {
        int axis = a < 0 ? a + ndim + 1 : a;
        if (axis < 0 || axis > ndim)
            throw std::runtime_error("unsqueeze: axis out of range");
        norm_axes.push_back(axis);
    }
    std::sort(norm_axes.begin(), norm_axes.end());

    // 新 shape
    std::vector<int> new_shape = in_shape;
    for (int axis : norm_axes) {
        new_shape.insert(new_shape.begin() + axis, 1);
    }

    // 新 strides
    std::vector<int> new_strides;
    new_strides.reserve(new_shape.size());
    int j = 0;
    for (int i = 0; i < new_shape.size(); ++i) {
        if (j < norm_axes.size() && i == norm_axes[j]) {
            new_strides.push_back(0); // unsqueeze 新维度 stride = 0
            j++;
        } else {
            new_strides.push_back(in_strides[i - j]);
        }
    }

    // 创建零拷贝 view tensor
    auto view = std::make_shared<DCUTensor>(
        x->data(),
        new_shape,
        new_strides,
        x->dtype(),
        false,
        x
    );

    return view;
}

// ------------------------
// reshape 实现
// ------------------------
std::shared_ptr<DCUTensor> DCUEngine::reshape(
    std::shared_ptr<DCUTensor> x,
    const std::vector<int>& shape,
    DCUContext* ctx)
{
    if (!x)
        throw std::runtime_error("reshape: input tensor is nullptr");

    const auto& in_shape   = x->shape();
    const auto& in_strides = x->strides();
    int in_ndim = x->ndim();

    std::vector<int> new_shape;
    new_shape.reserve(shape.size());

    int infer_axis = -1;
    int known_product = 1;

    for (int i = 0; i < shape.size(); ++i) {
        int s = shape[i];

        if (s == 0) {
            if (i >= in_ndim)
                throw std::runtime_error("reshape: 0-dim out of range");
            s = in_shape[i];
        }

        if (s == -1) {
            if (infer_axis != -1)
                throw std::runtime_error("reshape: multiple -1 not allowed");
            infer_axis = i;
            new_shape.push_back(1);
        } else {
            new_shape.push_back(s);
            known_product *= s;
        }
    }

    int total = x->size();
    if (infer_axis != -1) {
        if (total % known_product != 0)
            throw std::runtime_error("reshape: invalid -1 inference");
        new_shape[infer_axis] = total / known_product;
    }
    bool contiguous = x->is_contiguous();

    std::shared_ptr<DCUTensor> base = x;

    if (!contiguous) {
        base = to_contiguous(x, ctx);
    }

    std::vector<int> new_strides(new_shape.size());
    int stride = 1;
    for (int i = static_cast<int>(new_shape.size()) - 1; i >= 0; --i) {
        new_strides[i] = stride;
        stride *= new_shape[i];
    }

    auto view = std::make_shared<DCUTensor>(
        base->data(),
        new_shape,
        new_strides,
        x->dtype(),
        false,
        base
    );

    return view;
}


// ------------------------
// concat 实现
// ------------------------
static int prod(const std::vector<int>& v, int start, int end) {
    int r = 1;
    for (int i = start; i < end; ++i) r *= v[i];
    return r;
}

DCUTensor* DCUEngine::concat(
    const std::vector<DCUTensor*>& inputs,
    int axis,
    DCUContext* ctx
) {
    if (inputs.empty())
        throw std::runtime_error("concat: empty inputs");

    int ndim = inputs[0]->ndim();
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim)
        throw std::runtime_error("concat: axis out of range");

    std::vector<std::shared_ptr<DCUTensor>> owned_buffers;
    std::vector<DCUTensor*> tensors;
    tensors.reserve(inputs.size());

    for (auto* t : inputs) {
        if (!t)
            throw std::runtime_error("concat: nullptr input");

        if ((int)t->shape().size() != ndim)
            throw std::runtime_error("concat: rank mismatch");

        if (!t->is_contiguous()) {
            auto contig = ensure_contiguous(t, ctx);
            owned_buffers.push_back(contig);
            tensors.push_back(contig.get());
        } else {
            tensors.push_back(t);
        }
    }

    std::vector<int> out_shape = tensors[0]->shape();
    out_shape[axis] = 0;

    for (auto* t : tensors) {
        for (int d = 0; d < ndim; ++d) {
            if (d == axis) continue;
            if (t->shape()[d] != out_shape[d])
                throw std::runtime_error("concat: shape mismatch");
        }
        out_shape[axis] += t->shape()[axis];
    }
    DCUTensor* out = new DCUTensor(out_shape);

    float* out_data = static_cast<float*>(out->data());

    int outer = prod(out_shape, 0, axis);
    int inner = prod(out_shape, axis + 1, ndim);

    int out_axis_offset = 0;

    for (auto* t : tensors) {
        int axis_len = t->shape()[axis];
        int block = axis_len * inner;

        const float* src = static_cast<const float*>(t->data());

        for (int o = 0; o < outer; ++o) {
            size_t bytes = block * sizeof(float);

            const float* src_ptr =
                src + o * block;

            float* dst_ptr =
                out_data + o * (out_shape[axis] * inner)
                         + out_axis_offset * inner;

            CHECK_HIP(hipMemcpyAsync(
                dst_ptr,
                src_ptr,
                bytes,
                hipMemcpyDeviceToDevice,
                ctx->get_stream()
            ));
        }

        out_axis_offset += axis_len;
    }

    return out;
}
// ------------------------
// cast 实现
// ------------------------
void DCUEngine::cast(DCUTensor* x, DCUTensor* out, DCUContext* ctx)
{
    int total = x->size();
    if (total == 0) return;

    dim3 block(256);
    dim3 grid((total + block.x - 1) / block.x);

    auto dispatch_in = [&](auto in_enum) {
        using InT = typename engine::CType<in_enum>::type;

        auto dispatch_out = [&](auto out_enum) {
            using OutT = typename engine::CType<out_enum>::type;

            hipLaunchKernelGGL(
                (cast_kernel<InT, OutT>),
                grid, block, 0, ctx->get_stream(),
                static_cast<const InT*>(x->data()),
                static_cast<OutT*>(out->data()),
                total
            );
        };

        DISPATCH_ALL_TYPE(out->dtype(), dispatch_out);
    };

    DISPATCH_ALL_TYPE(x->dtype(), dispatch_in);
    CHECK_HIP(hipStreamSynchronize(ctx->get_stream()));
}
} // namespace dcu
