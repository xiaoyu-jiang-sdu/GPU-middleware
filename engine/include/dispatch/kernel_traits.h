#pragma once
#include "dtype.h"

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
struct CType<DataType::UINT8> {
    using type = uint8_t;
};

template <typename OpT, typename InT, typename = void>
struct op_out_type {
    using type = InT;
};

// 如果 OpT::OutType 存在
template <typename OpT, typename InT>
struct op_out_type<OpT, InT, std::void_t<typename OpT::OutType>> {
    using type = typename OpT::OutType;
};
} // namespace engine
