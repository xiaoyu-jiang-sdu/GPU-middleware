#pragma once
namespace dcu
{
// ------------------------
// gpu上内联函数
// ------------------------
__device__ __forceinline__
size_t compute_offset(
    int linear_idx,
    const int* shape,
    const int* strides,
    int ndim
) {
    size_t off = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        int idx = linear_idx % shape[d];
        linear_idx /= shape[d];
        off += idx * strides[d];
    }
    return off;
}

// ------------------------
// 全局kernel函数
// ------------------------
template <typename T, typename Op>
__global__ void unary_strided_kernel(
    const T* x, T* o,
    const int* shape,
    const int* x_strides,
    const int* o_strides,
    int ndim, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    size_t x_off = compute_offset(i, shape, x_strides, ndim);
    size_t o_off = compute_offset(i, shape, o_strides, ndim);

    o[o_off] = Op::apply(x[x_off]);
}

template <typename T, typename Op>
__global__ void unary_scalar_strided_kernel(
    const T* x, T* o,
    T s,
    const int* shape,
    const int* x_strides,
    const int* o_strides,
    int ndim, int total)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;

    size_t x_off = compute_offset(i, shape, x_strides, ndim);
    size_t o_off = compute_offset(i, shape, o_strides, ndim);

    o[o_off] = Op::apply(x[x_off], s);
}

template <typename InT, typename OutT, typename Op>
__global__ void binary_strided_kernel(
    const InT* a, const InT* b, OutT* out,
    const int* out_shape,
    const int* a_strides,
    const int* b_strides,
    const int* out_strides,
    int ndim, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int tmp = idx;
    int a_idx = 0, b_idx = 0, out_idx = 0;

    for (int d = ndim - 1; d >= 0; --d) {
        int coord = tmp % out_shape[d];
        tmp /= out_shape[d];

        a_idx   += coord * a_strides[d];
        b_idx   += coord * b_strides[d];
        out_idx += coord * out_strides[d];
    }

    out[out_idx] = Op::apply(a[a_idx], b[b_idx]);
}

// 三元where函数kernel
template <typename T>
__global__ void where_strided_kernel(
    const uint8_t* cond, const T* x, const T* y, T* out,
    const int* shape,
    const int* cond_strides,
    const int* x_strides,
    const int* y_strides,
    const int* out_strides,
    int ndim,
    int total
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int tmp = idx;
    int cond_idx = 0, x_idx = 0, y_idx = 0, out_idx = 0;

    for (int d = ndim - 1; d >= 0; --d) {
        int coord = tmp % shape[d];
        tmp /= shape[d];

        cond_idx += (cond_strides[d] == 0 ? 0 : coord * cond_strides[d]);
        x_idx += (x_strides[d] == 0 ? 0 : coord * x_strides[d]);
        y_idx += (y_strides[d] == 0 ? 0 : coord * y_strides[d]);
        out_idx += coord * out_strides[d];
    }

    out[out_idx] = cond[cond_idx] ? x[x_idx] : y[y_idx];
}

template <typename InT, typename OutT>
__global__ void cast_kernel(
    const InT* in,
    OutT* out,
    int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        out[idx] = static_cast<OutT>(in[idx]);
    }
}

// ------------------------
// unary Op模板 struct
// ------------------------
template <typename T>
struct CopyOp {
    __device__ __forceinline__ static T apply(T x) {
        return x;
    }
};

template <typename T>
struct ReluOp {
    __device__ __forceinline__ static T apply(T x) {
        return x > T(0) ? x : T(0);
    }
};

template <typename T>
struct ErfOp {
    __device__ __forceinline__ static T apply(T x) {
        return erf(x);
    }
};

template <typename T>
struct SqrtOp {
    __device__ __forceinline__ static T apply(T x) {
        return sqrt(x);
    }
};

// ------------------------
// unary + scalar Op模板 struct
// ------------------------
template <typename T>
struct MulScalarOp {
    __device__ __forceinline__ static T apply(T x, T s) {
        return x * s;
    }
};

// ------------------------
// binary Op模板 struct
// ------------------------
template <typename T>
struct AddOp {
    __device__ __forceinline__ static T apply(T a, T b) {
        return a + b;
    }
};

template <typename T>
struct SubOp {
    __device__ __forceinline__ static T apply(T a, T b) {
        return a - b;
    }
};

template <typename T>
struct MulOp {
    __device__ __forceinline__ static T apply(T a, T b) {
        return a * b;
    }
};

template <typename T>
struct DivOp {
    __device__ __forceinline__ static T apply(T a, T b) {
        return a / b;
    }
};

template <typename T>
struct PowOp {
    __device__ __forceinline__ static T apply(T a, T b) {
        return pow(a, b);
    }
};

template <typename T>
struct ModOp {
    __device__ __forceinline__ static T apply(T x, T y) {
        return x - floor(x / y) * y;
    }
};

template <typename T>
struct EqualOp {
    using OutType = uint8_t;

    __device__ __forceinline__ static uint8_t apply(T a, T b) {
        return static_cast<uint8_t>(a == b);
    }
};

template <typename T>
struct NotEqualOp {
    using OutType = uint8_t;

    __device__ __forceinline__ static uint8_t apply(T a, T b) {
        return static_cast<uint8_t>(a != b);
    }
};
}