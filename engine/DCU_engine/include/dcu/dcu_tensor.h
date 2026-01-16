//
// Created by 31437 on 26-1-16.
//

#ifndef DCU_TENSOR_H
#define DCU_TENSOR_H

#endif //DCU_TENSOR_H
#pragma once
#include <vector>
#include <hip/hip_runtime.h>

namespace dcu {

    class DCUTensor {
    public:
        DCUTensor(const std::vector<int>& shape);
        ~DCUTensor();

        void* data() const { return data_; }
        const std::vector<int>& shape() const { return shape_; }

    private:
        std::vector<int> shape_;
        void* data_;
    };

}
