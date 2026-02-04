#pragma once

namespace engine
{
enum class DataType {
    FLOAT32,
    INT32,
    INT64,
    BOOL
};

inline size_t dtype_size(DataType dt) {
    switch (dt) {
        case DataType::FLOAT32: return 4;
        case DataType::INT32:   return 4;
        case DataType::INT64:   return 8;
        case DataType::BOOL:    return 1;
        default: throw std::runtime_error("Unknown dtype");
    }
}
}
