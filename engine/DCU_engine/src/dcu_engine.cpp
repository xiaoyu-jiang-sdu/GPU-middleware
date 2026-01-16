#include "dcu/dcu_engine.h"
#include <miopen/miopen.h>
#include <rocblas/rocblas.h>
#include <vector>
#include <iostream>
#include <cstdlib> // rand
#include <ctime>   // time

namespace dcu {

    DCUEngine::DCUEngine() {}

    void DCUEngine::matmul(
        DCUTensor& A,
        DCUTensor& B,
        DCUTensor& C,
        int M, int N, int K) {

        float alpha = 1.0f, beta = 0.0f;

        rocblas_sgemm(
            ctx_.get_rocblas(),
            rocblas_operation_none,
            rocblas_operation_none,
            N, M, K,
            &alpha,
            (float*)B.data(), N,
            (float*)A.data(), K,
            &beta,
            (float*)C.data(), N);
    }

    // 卷积前向接口，带算法搜索和非零初始化
    void DCUEngine::conv2d_forward(float* x, float* w, float* y,
                                   int N, int C, int H, int W,
                                   int K, int R, int S,
                                   int stride_h, int stride_w,
                                   int pad_h, int pad_w,
                                   miopenHandle_t& handle) {

        // 初始化输入为随机非零数据
        std::srand(static_cast<unsigned>(time(0)));
        for (int i = 0; i < N*C*H*W; ++i) x[i] = static_cast<float>(rand()) / RAND_MAX;
        for (int i = 0; i < K*C*R*S; ++i) w[i] = static_cast<float>(rand()) / RAND_MAX;
        for (int i = 0; i < N*K*H*W; ++i) y[i] = 0.0f;

        float alpha = 1.0f;
        float beta = 0.0f;

        miopenTensorDescriptor_t x_desc, w_desc, y_desc;
        miopenCreateTensorDescriptor(&x_desc);
        miopenCreateTensorDescriptor(&w_desc);
        miopenCreateTensorDescriptor(&y_desc);

        miopenSet4dTensorDescriptor(x_desc, miopenFloat, N, C, H, W);
        miopenSet4dTensorDescriptor(w_desc, miopenFloat, K, C, R, S);
        miopenSet4dTensorDescriptor(y_desc, miopenFloat, N, K, H, W);

        miopenConvolutionDescriptor_t conv_desc;
        miopenCreateConvolutionDescriptor(&conv_desc);
        miopenInitConvolutionDescriptor(conv_desc,
                                        miopenConvolution,
                                        pad_h, pad_w,
                                        stride_h, stride_w,
                                        1, 1); // groups=1, dilation=1

        // 获取工作空间大小 (DTK MIOpen 接口是6个参数)
        size_t workspace_size = 0;
        miopenConvolutionForwardGetWorkSpaceSize(
            handle,
            w_desc, x_desc,
            conv_desc,
            y_desc,
            &workspace_size
        );

        void* workspace = nullptr;
        if (workspace_size > 0) {
            hipMalloc(&workspace, workspace_size);
        }

        // 寻找最优算法
        miopenConvAlgoPerf_t perfResults;
        int returnedAlgoCount = 0;
        miopenFindConvolutionForwardAlgorithm(
            handle,
            x_desc, x,
            w_desc, w,
            conv_desc,
            y_desc, y,
            1, // top 1 algorithm
            &returnedAlgoCount,
            &perfResults,
            workspace, workspace_size,
            false // exhaustive search = false
        );

        // 前向卷积
        miopenConvolutionForward(
            handle,
            &alpha,
            x_desc, x,
            w_desc, w,
            conv_desc,
            perfResults.fwd_algo,
            &beta,
            y_desc, y,
            workspace, workspace_size
        );

        if (workspace) hipFree(workspace);

        // 释放描述符
        miopenDestroyTensorDescriptor(x_desc);
        miopenDestroyTensorDescriptor(w_desc);
        miopenDestroyTensorDescriptor(y_desc);
        miopenDestroyConvolutionDescriptor(conv_desc);
    }

} // namespace dcu
