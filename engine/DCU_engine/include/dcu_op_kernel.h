#pragma once
#include "dcu_tensor.h"
#include "dcu_context.h"
#include "device_array.h"
#include "dcu_global_kernel.h"
#include "dispatch/dispatch.h"
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

namespace dcu
{

// 计算广播之后的shape
inline static std::vector<int> compute_broadcast_shape(
    const std::vector<int>& a_shape,
    const std::vector<int>& b_shape)
{
    int ndim = std::max(a_shape.size(), b_shape.size());
    std::vector<int> out_shape(ndim, 1);

    for (int i = 0; i < ndim; ++i) {
        int a_dim = (i < ndim - (int)a_shape.size()) ? 1 : a_shape[i - (ndim - a_shape.size())];
        int b_dim = (i < ndim - (int)b_shape.size()) ? 1 : b_shape[i - (ndim - b_shape.size())];

        if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
            throw std::runtime_error("Invalid broadcast shapes");
        }
        out_shape[i] = std::max(a_dim, b_dim);
    }
    return out_shape;
}

// 二元运算符
template <template<typename> class Op>
void binary_op_impl(
    DCUTensor* a,
    DCUTensor* b,
    DCUTensor* out,
    DCUContext* ctx)
{
    const auto& a_shape = a->shape();
    const auto& b_shape = b->shape();

    auto out_shape = compute_broadcast_shape(a_shape, b_shape);
    int ndim = out_shape.size();

    int total = 1;
    for (int d : out_shape) total *= d;
    if (total == 0) return;

    std::vector<int> a_shape_al(ndim,1), b_shape_al(ndim,1);
    std::vector<int> a_strides_al(ndim,0), b_strides_al(ndim,0);

    std::copy(a_shape.begin(), a_shape.end(),
              a_shape_al.begin() + (ndim - a_shape.size()));
    std::copy(b_shape.begin(), b_shape.end(),
              b_shape_al.begin() + (ndim - b_shape.size()));

    std::copy(a->strides().begin(), a->strides().end(),
              a_strides_al.begin() + (ndim - a->strides().size()));
    std::copy(b->strides().begin(), b->strides().end(),
              b_strides_al.begin() + (ndim - b->strides().size()));

    for (int i = 0; i < ndim; i++) {
        if (a_shape_al[i] == 1) a_strides_al[i] = 0;
        if (b_shape_al[i] == 1) b_strides_al[i] = 0;
    }

    dcu::DeviceArray<int> d_out_shape(out_shape, ctx->get_stream());
    dcu::DeviceArray<int> d_a_shape(a_shape_al, ctx->get_stream());
    dcu::DeviceArray<int> d_b_shape(b_shape_al, ctx->get_stream());
    dcu::DeviceArray<int> d_a_strides(a_strides_al, ctx->get_stream());
    dcu::DeviceArray<int> d_b_strides(b_strides_al, ctx->get_stream());

    dim3 block(256);
    dim3 grid((total + block.x - 1) / block.x);

    DISPATCH_DTYPE(a->dtype(), [&](auto dtype_enum) {
        using T = typename engine::CType<decltype(dtype_enum)::value>::type;
        hipLaunchKernelGGL(
            (binary_strided_kernel<T, Op<T>>),
            grid, block, 0, ctx->get_stream(),
            static_cast<const T*>(a->data()),
            static_cast<const T*>(b->data()),
            static_cast<T*>(out->data()),
            d_out_shape.data(),
            d_a_shape.data(),
            d_b_shape.data(),
            d_a_strides.data(),
            d_b_strides.data(),
            ndim,
            total
        );
    });
    CHECK_HIP(hipStreamSynchronize(ctx->get_stream()));
}

// 一元运算符
template <template <typename> class Op>
void unary_op_impl(
    DCUTensor* x,
    DCUTensor* o,
    DCUContext* ctx)
{
    int ndim  = x->ndim();
    size_t total = x->size();

    dim3 block(256);
    dim3 grid((total + block.x - 1) / block.x);

    DISPATCH_DTYPE(x->dtype(), [&](auto dtype_enum) {
        using T = typename engine::CType<decltype(dtype_enum)::value>::type;
        hipLaunchKernelGGL(
            (unary_strided_kernel<T, Op<T>>),
            grid, block, 0, ctx->get_stream(),
            static_cast<const T*>(x->data()),
            static_cast<T*>(o->data()),
            x->d_shape(),
            x->d_strides(),
            o->d_strides(),
            ndim,
            total
        );
    });
    CHECK_HIP(hipStreamSynchronize(ctx->get_stream()));
}

// 重载： shared_ptr
template <template <typename> class Op>
void unary_op_impl(
    const std::shared_ptr<DCUTensor>& x,
    const std::shared_ptr<DCUTensor>& o,
    DCUContext* ctx)
{
    if (!x || !o) {
        throw std::runtime_error("unary_op_impl: input or output shared_ptr is nullptr");
    }
    unary_op_impl<Op>(x.get(), o.get(), ctx);
}

// 一元运算符 + 标量
template <template <typename> class Op>
void unary_scalar_op_impl(
    DCUTensor* x,
    DCUTensor* o,
    float s,
    DCUContext* ctx)
{
    int ndim  = x->ndim();
    size_t total = x->size();

    dim3 block(256);
    dim3 grid((total + block.x - 1) / block.x);

    DISPATCH_DTYPE(x->dtype(), [&](auto dtype_enum) {
        using T = typename engine::CType<decltype(dtype_enum)::value>::type;
        hipLaunchKernelGGL(
            (unary_scalar_strided_kernel<T, Op<T>>),
            grid, block, 0, ctx->get_stream(),
            static_cast<const T*>(x->data()),
            static_cast<T*>(o->data()),
            static_cast<T>(s),
            x->d_shape(),
            x->d_strides(),
            o->d_strides(),
            ndim,
            total
        );
    });
    CHECK_HIP(hipStreamSynchronize(ctx->get_stream()));
}

template <typename T>
void matmul_template(DCUTensor* A, DCUTensor* B, DCUTensor* Out,
                     bool transA, bool transB,
                     float alpha, float beta,
                     DCUContext* ctx)
{
    int ndimA = A->ndim();
    int ndimB = B->ndim();
    const auto& Ashape = A->shape();
    const auto& Bshape = B->shape();

    // row-major
    int logical_M = transA ? Ashape[ndimA - 1] : Ashape[ndimA - 2];
    int logical_K = transA ? Ashape[ndimA - 2] : Ashape[ndimA - 1];
    int logical_N = transB ? Bshape[ndimB - 2] : Bshape[ndimB - 1];

    int B_inner = transB ? Bshape[ndimB - 1] : Bshape[ndimB - 2];
    if (logical_K != B_inner) {
        std::cerr << "Matmul shape mismatch on contraction dim\n";
        return;
    }

    // batch size
    int batch = 1;
    if (ndimA > 2) {
        for (int i = 0; i < ndimA - 2; ++i) {
            batch *= Ashape[i];
        }
    }
    if (batch <= 1) batch = 1;

    // 检查 B batch 匹配
    int batchB = 1;
    if (ndimB > 2) {
        for (int i = 0; i < ndimB - 2; ++i) {
            batchB *= Bshape[i];
        }
        if (batch != batchB) {
            std::cerr << "Matmul batch size mismatch between A and B\n";
            return;
        }
    }

    rocblas_operation opA_roc = transB ? rocblas_operation_transpose : rocblas_operation_none;  // op(B)
    rocblas_operation opB_roc = transA ? rocblas_operation_transpose : rocblas_operation_none;  // op(A)

    int m_roc = logical_N;
    int n_roc = logical_M;
    int k_roc = logical_K;

    int lda = (opA_roc == rocblas_operation_none) ? logical_N : logical_K;
    int ldb = (opB_roc == rocblas_operation_none) ? logical_K : logical_M;
    int ldc = logical_N;

    // batch stride
    int strideA = lda * ((opA_roc == rocblas_operation_none) ? logical_K : logical_N);  // for op(B)
    int strideB = ldb * ((opB_roc == rocblas_operation_none) ? logical_M : logical_K);  // for op(A)
    int strideC = ldc * logical_M;

    if (ndimB == 2 && batch > 1) {
        strideA = 0;
    }

    if (batch <= 1) {
        strideA = strideB = strideC = 0;
    }

    float alpha_h = alpha;
    float beta_h  = beta;

    // 列主序
    rocblas_status status = rocblas_sgemm_strided_batched(
        ctx->get_rocblas(),
        opA_roc,
        opB_roc,
        m_roc,
        n_roc,
        k_roc,
        &alpha_h,
        static_cast<const float*>(B->data()), lda, strideA,
        static_cast<const float*>(A->data()), ldb, strideB,
        &beta_h,
        static_cast<float*>(Out->data()),     ldc, strideC
        batch
    );

    if (status != rocblas_status_success) {
        std::cerr << "rocblas_sgemm_strided_batched failed: " << status << std::endl;
    }
}

}