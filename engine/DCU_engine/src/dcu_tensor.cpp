#include "dcu_tensor.h"
#include <hip/hip_runtime.h>

namespace dcu {

DCUTensor::DCUTensor(const std::vector<int>& shape) : shape_(shape) {
    size_t bytes = sizeof(float);
    for (int d : shape_) bytes *= d;
    hipMalloc(&data_, bytes);
}

DCUTensor::~DCUTensor() {
    hipFree(data_);
}

} // namespace dcu