#pragma once
#include "dcu_tensor.h"
#include "dcu_context.h"
#include "engine.h"
#include <ctime>
#include <cstdlib>

namespace dcu {

class DCUEngine final : public engine::Engine<DCUTensor, DCUContext> {
public:
    DCUEngine() = default;
    ~DCUEngine() override = default;

    // 基础算子
    void add(DCUTensor*, DCUTensor*, DCUTensor*,
                 const std::vector<int>&,
                 const std::vector<int>&,
                  DCUContext*) override;
    void matmul(DCUTensor*, DCUTensor*, DCUTensor*, int, int, int, DCUContext*) override;
    void relu(DCUTensor*, DCUTensor*, int, DCUContext*) override;
    void transpose(DCUTensor*, DCUTensor*, int, int, DCUContext*) override;
    void mul_scalar(DCUTensor*, DCUTensor*, float, int, DCUContext*) override;

    // CNN
    void conv2d(DCUTensor*, DCUTensor*, DCUTensor*,
                int, int, int, int,
                int, int, int,
                int, int,
                int, int,
                int, int,
                int,
                DCUContext*) override;

    void max_pool2d(DCUTensor*, DCUTensor*,
                    int, int, int, int,
                    int, int,
                    int, int,
                    int, int,
                    DCUContext*) override;

    void global_avg_pool(DCUTensor*, DCUTensor*,
                         int, int, int, int,
                         DCUContext*) override;

    void flatten(DCUTensor*, DCUTensor*, int, int, DCUContext*) override;

    // 归一化
    void batch_norm_2d(DCUTensor*, DCUTensor*,
                       DCUTensor*, DCUTensor*,
                       DCUTensor*, DCUTensor*,
                       int, int, int, int,
                       float,
                       DCUContext*) override;
};

} // namespace dcu