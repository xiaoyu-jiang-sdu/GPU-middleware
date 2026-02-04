#pragma once
#include "kernel_traits.h"

#define DISPATCH_DTYPE(dtype, FUNC)                                 \
    switch (dtype) {                                                \
        case engine::DataType::FLOAT32:                             \
            FUNC(std::integral_constant<engine::DataType,           \
                 engine::DataType::FLOAT32>{});                     \
            break;                                                  \
        case engine::DataType::INT32:                               \
            FUNC(std::integral_constant<engine::DataType,           \
                 engine::DataType::INT32>{});                       \
            break;                                                  \
        case engine::DataType::INT64:                               \
            FUNC(std::integral_constant<engine::DataType,           \
                 engine::DataType::INT64>{});                       \
            break;                                                  \
        default:                                                    \
            throw std::runtime_error("unsupported dtype");          \
    }

#define DISPATCH_FLOAT_ONLY(dtype, FUNC)                            \
    switch (dtype) {                                                \
        case engine::DataType::FLOAT32:                             \
            FUNC(std::integral_constant<engine::DataType,           \
                 engine::DataType::FLOAT32>{});                     \
            break;                                                  \
        default:                                                    \
            throw std::runtime_error("float only");                 \
    }