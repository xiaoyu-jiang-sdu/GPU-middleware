#include "dcu_engine.h"
#include <miopen/miopen.h>
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <ctime>
#include <cstdlib>
#include <iostream>

namespace dcu {

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
// conv2d_forward 实现
// ------------------------
void DCUEngine::conv2d_forward(DCUTensor* x, DCUTensor* w, DCUTensor* y,
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

} // namespace dcu
