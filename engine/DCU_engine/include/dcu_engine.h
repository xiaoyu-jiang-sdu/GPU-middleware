#pragma once
#include "dcu_tensor.h"
#include "dcu_context.h"
#include "engine.h"
#include <ctime>
#include <cstdlib>

namespace dcu {

// DCU 后端 Engine
class DCUEngine : public engine::Engine<DCUTensor, DCUContext> {
public:
    DCUEngine() = default;

    // 矩阵乘法
    void matmul(DCUTensor* A, DCUTensor* B, DCUTensor* Out,
                int M, int N, int K,
                DCUContext* ctx);

    // 卷积前向
    void conv2d_forward(DCUTensor* x, DCUTensor* w, DCUTensor* y,
                        int N, int C, int H, int W,
                        int K, int R, int S,
                        int stride_h, int stride_w,
                        int pad_h, int pad_w,
                        int dilation_h, int dilation_w,
                        int groups,
                        DCUContext* ctx);
};

} // namespace dcu
