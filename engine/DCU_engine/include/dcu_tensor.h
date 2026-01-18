#pragma once
#include <vector>
#include <hip/hip_runtime.h>
#include "tensor.h"


namespace dcu {

class DCUTensor : public engine::Tensor {
public:
    DCUTensor(const std::vector<int>& shape);
    ~DCUTensor();

    void* data() const override { return data_; }
    const std::vector<int>& shape() const override { return shape_; }
    engine::DeviceType device() const override { return engine::DeviceType::DCU; }

private:
    std::vector<int> shape_;
    void* data_;
};

} // namespace dcu
