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

    // 返回连续版本的张量
    static std::shared_ptr<DCUTensor> to_contiguous(const std::shared_ptr<DCUTensor>&, DCUContext*);

    // ------------------------
    // 二元运算OP
    // ------------------------
    void add(DCUTensor*, DCUTensor*, DCUTensor*, DCUContext*) override;
    void sub(DCUTensor*, DCUTensor*, DCUTensor*, DCUContext*) override;
    void mul(DCUTensor*, DCUTensor*, DCUTensor*, DCUContext*) override;
    void div(DCUTensor*, DCUTensor*, DCUTensor*, DCUContext*) override;
    void pow(DCUTensor*, DCUTensor*, DCUTensor*, DCUContext*) override;
    void mod(DCUTensor*, DCUTensor*, DCUTensor*, DCUContext*) override;
    void mul_scalar(DCUTensor*, DCUTensor*, float, DCUContext*) override;

    // ------------------------
    // NN算子
    // ------------------------
    void matmul(DCUTensor*, DCUTensor*, DCUTensor*,
                bool, bool,
                float, float,
                DCUContext*) override;
    void conv2d(DCUTensor*, DCUTensor*, DCUTensor*,
                int, int, int, int,
                int, int, int,
                int, int,
                int, int,
                int, int,
                int,
                DCUContext*) override;

    // ------------------------
    // 激活算子
    // ------------------------
    void relu(DCUTensor*, DCUTensor*, DCUContext*) override;
    void erf(DCUTensor*, DCUTensor*, DCUContext*) override;
    void sqrt(DCUTensor*, DCUTensor*, DCUContext*) override;

    // ------------------------
    // 池化算子
    // ------------------------
    void max_pool2d(DCUTensor*, DCUTensor*,
                    int, int, int, int,
                    int, int,
                    int, int,
                    int, int,
                    DCUContext*) override;

    void global_avg_pool(DCUTensor*, DCUTensor*,
                         int, int, int, int,
                         DCUContext*) override;
    // ------------------------
    // 归一化
    // ------------------------
    void batch_norm_2d(DCUTensor*, DCUTensor*,
                       DCUTensor*, DCUTensor*,
                       DCUTensor*, DCUTensor*,
                       int, int, int, int,
                       float,
                       DCUContext*) override;

    // ------------------------
    // shape view
    // ------------------------
    std::shared_ptr<DCUTensor> transpose(std::shared_ptr<DCUTensor>, const std::vector<int>&, DCUContext*) override;
    std::shared_ptr<DCUTensor> unsqueeze(std::shared_ptr<DCUTensor>, const std::vector<int>&, DCUContext*) override;
    std::shared_ptr<DCUTensor> reshape(std::shared_ptr<DCUTensor>, const std::vector<int>&, DCUContext*) override;

    // ------------------------
    // transform
    // ------------------------
    void flatten(DCUTensor*, DCUTensor*, int, DCUContext*) override;
    DCUTensor* concat(const std::vector<DCUTensor*>&, int, DCUContext*) override;
};

} // namespace dcu