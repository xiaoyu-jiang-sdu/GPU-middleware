#pragma once
#include <vector>
#include <hip/hip_runtime.h>
#include "tensor.h"


namespace dcu {

class DCUTensor : public engine::Tensor {
public:
    explicit DCUTensor(const std::vector<int>& shape);
    DCUTensor(const float* host_data,
              const std::vector<int>& shape);
    ~DCUTensor();

    void* data() const override { return data_; }
    const std::vector<int>& shape() const override { return shape_; }
    engine::DeviceType device() const override { return engine::DeviceType::DCU; }
    std::vector<int> shape_list() const { return shape_; }

    size_t size() const {          // 元素总数
        size_t s = 1;
        for (int d : shape_) s *= d;
        return s;
    }

private:
    std::vector<int> shape_;
    void* data_;

    void allocate();
};

} // namespace dcu
