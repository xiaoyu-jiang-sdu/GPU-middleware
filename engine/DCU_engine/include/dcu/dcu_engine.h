//
// Created by 31437 on 26-1-16.
//

#ifndef DCU_ENGINE_H
#define DCU_ENGINE_H

#endif //DCU_ENGINE_H
#pragma once
#include "dcu_context.h"
#include "dcu_tensor.h"

namespace dcu {

    class DCUEngine {
    public:
        DCUEngine();

        void conv2d_forward(float* x, float* w, float* y,
                        int N, int C, int H, int W,
                        int K, int R, int S,
                        int stride_h, int stride_w,
                        int pad_h, int pad_w,
                        miopenHandle_t& handle);
        void matmul(
            DCUTensor& A,
            DCUTensor& B,
            DCUTensor& C,
            int M, int N, int K);

    private:
        DCUContext ctx_;
    };

}
