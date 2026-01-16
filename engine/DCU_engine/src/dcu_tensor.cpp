//
// Created by 31437 on 26-1-16.
//
#include "dcu/dcu_tensor.h"

namespace dcu {

    DCUTensor::DCUTensor(const std::vector<int>& shape)
        : shape_(shape) {
        size_t bytes = sizeof(float);
        for (int d : shape) bytes *= d;
        hipMalloc(&data_, bytes);
    }

    DCUTensor::~DCUTensor() {
        hipFree(data_);
    }

}
