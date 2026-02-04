// miopen_dtype_traits.h
#pragma once
#include <miopen/miopen.h>
#include "dtype.h"

namespace dcu {

template <engine::DataType>
struct MiopenTraits;

template <>
struct MiopenTraits<engine::DataType::FLOAT32> {
    using CType = float;
    static constexpr miopenDataType_t miopen_type = miopenFloat;
};
//
//template <>
//struct MiopenTraits<engine::DataType::FLOAT16> {
//    using CType = half;
//    static constexpr miopenDataType_t miopen_type = miopenHalf;
//};

} // namespace dcu
