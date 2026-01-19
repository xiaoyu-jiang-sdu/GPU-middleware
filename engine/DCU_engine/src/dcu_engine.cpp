#include "dcu_engine.h"
#include "dcu_common.h"
#include <miopen/miopen.h>
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <ctime>
#include <cstdlib>
#include <iostream>

namespace dcu {
// ------------------------
// 全局kernel函数
// ------------------------
__global__ void add_kernel(float* a, float* b, float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = a[i] + b[i];
    return;
}

__global__ void relu_kernel(float* x, float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = x[i] > 0 ? x[i] : 0;
    return;
}

__global__ void mul_scalar_kernel(float* x, float* o, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = x[i] * s;
    return;
}

__global__ void transpose_kernel(float* x, float* y,
                                 int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols) {
        y[c * rows + r] = x[r * cols + c];
    }
    return;
}

__global__ void add_broadcast_nd_kernel(
    const float* a, const float* b, float* out,
    const int* a_strides, const int* b_strides,
    const int* b_shape,
    int ndim, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int tmp = idx;
    int b_idx = 0;
    for (int d = 0; d < ndim; ++d) {
        int i = tmp / a_strides[d];
        tmp = tmp % a_strides[d];
        int bi = (b_shape[d] == 1) ? 0 : i;
        b_idx += bi * b_strides[d];
    }
    out[idx] = a[idx] + b[b_idx];
}
// ------------------------
// add 实现
// ------------------------
void DCUEngine::add(DCUTensor* a, DCUTensor* b, DCUTensor* o, int n, DCUContext*) {
    hipLaunchKernelGGL(add_kernel,
        dim3((n + 255) / 256), dim3(256), 0, 0,
        (float*)a->data(), (float*)b->data(), (float*)o->data(), n);
}
// ------------------------
// add_broadcast_nd 实现
// ------------------------
void DCUEngine::add_broadcast_nd(
    DCUTensor* a, DCUTensor* b, DCUTensor* out,
    const std::vector<int>& a_shape,
    const std::vector<int>& b_shape,
    DCUContext* ctx)
{
    int ndim = a_shape.size();
    std::vector<int> a_strides(ndim), b_strides(ndim);

    // 计算 strides
    a_strides[ndim-1] = 1;
    b_strides[ndim-1] = 1;
    for (int i = ndim-2; i >= 0; --i) {
        a_strides[i] = a_strides[i+1] * a_shape[i+1];
        b_strides[i] = b_strides[i+1] * b_shape[i+1];
    }

    int total = 1;
    for (auto d : a_shape) total *= d;

    // 分配 GPU 内存
    int *d_a_strides = nullptr, *d_b_strides = nullptr, *d_b_shape = nullptr;
    CHECK_HIP(hipMalloc(&d_a_strides, ndim * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_b_strides, ndim * sizeof(int)));
    CHECK_HIP(hipMalloc(&d_b_shape, ndim * sizeof(int)));

    // 拷贝到 GPU
    CHECK_HIP(hipMemcpy(d_a_strides, a_strides.data(), ndim*sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_b_strides, b_strides.data(), ndim*sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_b_shape, b_shape.data(), ndim*sizeof(int), hipMemcpyHostToDevice));

    // launch kernel
    int block = 256;
    int grid = (total + block - 1) / block;
    hipLaunchKernelGGL(add_broadcast_nd_kernel,
                       dim3(grid), dim3(block), 0, 0,
                       static_cast<float*>(a->data()),
                       static_cast<float*>(b->data()),
                       static_cast<float*>(out->data()),
                       d_a_strides, d_b_strides, d_b_shape,
                       ndim, total);

    // 释放 GPU 内存
    CHECK_HIP(hipFree(d_a_strides));
    CHECK_HIP(hipFree(d_b_strides));
    CHECK_HIP(hipFree(d_b_shape));
}
// ------------------------
// matmul 实现
// ------------------------
void DCUEngine::matmul(DCUTensor* A, DCUTensor* B, DCUTensor* Out,
                       int M, int N, int K,
                       DCUContext* ctx)
{
    float alpha = 1.0f, beta = 0.0f;


    rocblas_status status = rocblas_sgemm(
        ctx->get_rocblas(),
        rocblas_operation_none,
        rocblas_operation_none,
        N,
        M,
        K,
        &alpha,
        static_cast<const float*>(B->data()),
        N,
        static_cast<const float*>(A->data()),
        K,
        &beta,
        static_cast<float*>(Out->data()),
        N
    );
    if (status != rocblas_status_success) {
        std::cerr << "[rocBLAS][sgemm] failed, status = "
                  << status << std::endl;
    }


}
// ------------------------
// Relu 实现
// ------------------------
void DCUEngine::relu(DCUTensor* x, DCUTensor* o, int n, DCUContext*) {
    hipLaunchKernelGGL(relu_kernel,
        dim3((n + 255) / 256), dim3(256), 0, 0,
        (float*)x->data(), (float*)o->data(), n);
}
// ------------------------
// transpose 实现
// ------------------------
void DCUEngine::transpose(DCUTensor* x, DCUTensor* y,
                          int rows, int cols,
                          DCUContext*) {
    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);
    hipLaunchKernelGGL(transpose_kernel,
        grid, block, 0, 0,
        static_cast<float*>(x->data()),
        static_cast<float*>(y->data()),
        rows, cols);
}
// ------------------------
// mul_scalar 实现
// ------------------------
void DCUEngine::mul_scalar(DCUTensor* x, DCUTensor* o,
                           float s, int n,
                           DCUContext*) {
    hipLaunchKernelGGL(mul_scalar_kernel,
        dim3((n + 255) / 256), dim3(256), 0, 0,
        static_cast<float*>(x->data()),
        static_cast<float*>(o->data()),
        s, n);
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
    float* x_data = static_cast<float*>(x->data());
    float* w_data = static_cast<float*>(w->data());
    float* y_data = static_cast<float*>(y->data());

    int Ho = (H + 2 * pad_h - dilation_h * (R - 1) - 1) / stride_h + 1;
    int Wo = (W + 2 * pad_w - dilation_w * (S - 1) - 1) / stride_w + 1;

    CHECK_HIP(hipMemset(y_data, 0, sizeof(float) * N * K * Ho * Wo));

    // Tensor 描述符
    miopenTensorDescriptor_t x_desc, w_desc, y_desc;
    miopenCreateTensorDescriptor(&x_desc);
    miopenCreateTensorDescriptor(&w_desc);
    miopenCreateTensorDescriptor(&y_desc);

    miopenSet4dTensorDescriptor(x_desc, miopenFloat, N, C, H, W);
    miopenSet4dTensorDescriptor(w_desc, miopenFloat, K, C/groups, R, S);
    miopenSet4dTensorDescriptor(y_desc, miopenFloat, N, K, Ho, Wo);

    // 卷积描述符
    miopenConvolutionDescriptor_t conv_desc;
    miopenCreateConvolutionDescriptor(&conv_desc);
    miopenInitConvolutionDescriptor(conv_desc,
                                    miopenConvolution,
                                    pad_h, pad_w,
                                    stride_h, stride_w,
                                    dilation_h, dilation_w);
    miopenSetConvolutionGroupCount(conv_desc, groups);

    // 工作空间
    size_t workspace_size = 0;
    miopenConvolutionForwardGetWorkSpaceSize(ctx->get_miopen(),
                                             x_desc, w_desc, conv_desc, y_desc, &workspace_size);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_HIP(hipMalloc(&workspace, workspace_size));
    }

    // 搜索最优算法
    miopenConvAlgoPerf_t perfResults;
    int returnedAlgoCount = 0;
    miopenFindConvolutionForwardAlgorithm(ctx->get_miopen(),
                                          x_desc, x_data,
                                          w_desc, w_data,
                                          conv_desc,
                                          y_desc, y_data,
                                          1, &returnedAlgoCount,
                                          &perfResults,
                                          workspace, workspace_size,
                                          false);

    float alpha = 1.0f, beta = 0.0f;
    miopenConvolutionForward(ctx->get_miopen(),
                             &alpha,
                             x_desc, x_data,
                             w_desc, w_data,
                             conv_desc,
                             perfResults.fwd_algo,
                             &beta,
                             y_desc, y_data,
                             workspace, workspace_size);

    if (workspace) CHECK_HIP(hipFree(workspace));

    // 清理描述符
    miopenDestroyTensorDescriptor(x_desc);
    miopenDestroyTensorDescriptor(w_desc);
    miopenDestroyTensorDescriptor(y_desc);
    miopenDestroyConvolutionDescriptor(conv_desc);
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
    miopenTensorDescriptor_t x_desc, y_desc;
    miopenCreateTensorDescriptor(&x_desc);
    miopenCreateTensorDescriptor(&y_desc);

    int outH = (H + 2 * pH - kH) / sH + 1;
    int outW = (W + 2 * pW - kW) / sW + 1;

    miopenSet4dTensorDescriptor(x_desc, miopenFloat, N, C, H, W);
    miopenSet4dTensorDescriptor(y_desc, miopenFloat, N, C, outH, outW);

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
        x_desc, x->data(),
        &beta,
        y_desc, y->data(),
        false, nullptr, 0
    );

    miopenDestroyPoolingDescriptor(pool_desc);
    miopenDestroyTensorDescriptor(x_desc);
    miopenDestroyTensorDescriptor(y_desc);
}

// ------------------------
// global avg pool 实现
// ------------------------
void DCUEngine::global_avg_pool(
    DCUTensor* x, DCUTensor* y,
    int N, int C, int H, int W,
    DCUContext* ctx)
{
    // 1. 创建 tensor 描述符
    miopenTensorDescriptor_t x_desc, y_desc;
    miopenCreateTensorDescriptor(&x_desc);
    miopenCreateTensorDescriptor(&y_desc);

    // 输入: NCHW
    miopenSet4dTensorDescriptor(x_desc, miopenFloat, N, C, H, W);

    // 输出: N x C x 1 x 1
    miopenSet4dTensorDescriptor(y_desc, miopenFloat, N, C, 1, 1);

    // 2. 创建池化描述符
    miopenPoolingDescriptor_t pool_desc;
    miopenCreatePoolingDescriptor(&pool_desc);

    // 全局平均池化: kernel size = H x W, stride=1, padding=0
    miopenSet2dPoolingDescriptor(
        pool_desc,
        miopenPoolingAverage,
        H, W,      // kernel
        0, 0,      // padding
        1, 1      // stride
    );

    float alpha = 1.0f;
    float beta = 0.0f;

    // 3. 调用 MIOpen 池化前向
    miopenPoolingForward(
        ctx->get_miopen(),
        pool_desc,
        &alpha,
        x_desc, x->data(),
        &beta,
        y_desc, y->data(),
        false,     // do_backward = false
        nullptr,   // workspace
        0          // workspace size
    );

    // 4. 清理描述符
    miopenDestroyPoolingDescriptor(pool_desc);
    miopenDestroyTensorDescriptor(x_desc);
    miopenDestroyTensorDescriptor(y_desc);
}



// ------------------------
// flatten 实现
// ------------------------
void DCUEngine::flatten(DCUTensor* x, DCUTensor* y,
                        int outer, int inner, DCUContext*) {
    CHECK_HIP(hipMemcpy(y->data(), x->data(),
                sizeof(float) * outer * inner,
                hipMemcpyDeviceToDevice));
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

    miopenTensorDescriptor_t x_desc, y_desc, bn_desc;
    miopenCreateTensorDescriptor(&x_desc);
    miopenCreateTensorDescriptor(&y_desc);
    miopenCreateTensorDescriptor(&bn_desc);

    miopenSet4dTensorDescriptor(x_desc, miopenFloat, N, C, H, W);
    miopenSet4dTensorDescriptor(y_desc, miopenFloat, N, C, H, W);
    miopenDeriveBNTensorDescriptor(bn_desc, x_desc, miopenBNSpatial);

    float alpha = 1.0f, beta = 0.0f;

    miopenBatchNormalizationForwardInference(
        ctx->get_miopen(),
        miopenBNSpatial,
        &alpha, &beta,
        x_desc, x->data(),
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
}
} // namespace dcu
