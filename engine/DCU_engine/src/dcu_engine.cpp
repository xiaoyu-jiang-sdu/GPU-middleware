#include "dcu_engine.h"
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
}

__global__ void relu_kernel(float* x, float* o, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = x[i] > 0 ? x[i] : 0;
}

__global__ void mul_scalar_kernel(float* x, float* o, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = x[i] * s;
}

__global__ void transpose_kernel(float* x, float* y,
                                 int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols) {
        y[c * rows + r] = x[r * cols + c];
    }
}

__global__ void gap_kernel(float* x, float* y, int HW) {
    int c = blockIdx.x;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < HW; i += blockDim.x)
        sum += x[c * HW + i];
    atomicAdd(&y[c], sum / HW);
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
// matmul 实现
// ------------------------
void DCUEngine::matmul(DCUTensor* A, DCUTensor* B, DCUTensor* Out,
                       int M, int N, int K,
                       DCUContext* ctx)
{
    float alpha = 1.0f, beta = 0.0f;
    rocblas_sgemm(ctx->get_rocblas(),
                  rocblas_operation_none,
                  rocblas_operation_none,
                  N, M, K,
                  &alpha,
                  static_cast<float*>(B->data()), N,
                  static_cast<float*>(A->data()), K,
                  &beta,
                  static_cast<float*>(Out->data()), N);
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

    // 初始化非零数据（测试用）
    std::srand(static_cast<unsigned>(time(0)));
    size_t x_size = static_cast<size_t>(N*C*H*W);
    size_t w_size = static_cast<size_t>(K*C/groups*R*S);
    size_t y_size = static_cast<size_t>(N*K*H*W);
    for (size_t i = 0; i < x_size; ++i) x_data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    for (size_t i = 0; i < w_size; ++i) w_data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    for (size_t i = 0; i < y_size; ++i) y_data[i] = 0.0f;

    // Tensor 描述符
    miopenTensorDescriptor_t x_desc, w_desc, y_desc;
    miopenCreateTensorDescriptor(&x_desc);
    miopenCreateTensorDescriptor(&w_desc);
    miopenCreateTensorDescriptor(&y_desc);

    miopenSet4dTensorDescriptor(x_desc, miopenFloat, N, C, H, W);
    miopenSet4dTensorDescriptor(w_desc, miopenFloat, K, C/groups, R, S);
    miopenSet4dTensorDescriptor(y_desc, miopenFloat, N, K, H, W);

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
                                             w_desc, x_desc, conv_desc, y_desc, &workspace_size);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        hipError_t e = hipMalloc(&workspace, workspace_size);
        if (e != hipSuccess) {
            std::cerr << "hipMalloc failed for workspace" << std::endl;
            workspace = nullptr;
        }
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

    if (workspace) hipFree(workspace);

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
// flatten 实现
// ------------------------
void DCUEngine::global_avg_pool(
    DCUTensor* x, DCUTensor* y,
    int N, int C, int H, int W,
    DCUContext*)
{
    int HW = H * W;
    hipMemset(y->data(), 0, sizeof(float) * N * C);

    dim3 grid(N * C);
    dim3 block(256);

    hipLaunchKernelGGL(gap_kernel,
        grid, block, 0, 0,
        static_cast<float*>(x->data()),
        static_cast<float*>(y->data()),
        HW);
}


// ------------------------
// flatten 实现
// ------------------------
void DCUEngine::flatten(DCUTensor* x, DCUTensor* y,
                        int outer, int inner, DCUContext*) {
    hipMemcpy(y->data(), x->data(),
              sizeof(float) * outer * inner,
              hipMemcpyDeviceToDevice);
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
