#pragma once
#include "dtype.h"
#include <miopen/miopen.h>

namespace engine {

template <DataType>
struct CType;

template <>
struct CType<DataType::FLOAT32> {
    using type = float;
};

template <>
struct CType<DataType::INT32> {
    using type = int;
};

template <>
struct CType<DataType::INT64> {
    using type = int64_t;
};

template <>
struct CType<DataType::BOOL> {
    using type = bool;
};
} // namespace engine
