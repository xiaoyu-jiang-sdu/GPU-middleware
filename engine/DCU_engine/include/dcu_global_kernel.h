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

template <typename T, typename Op>
__global__ void binary_strided_kernel(
    const T* a, const T* b, T* out,
    const int* out_shape, const int* a_shape, const int* b_shape,
    const int* a_strides, const int* b_strides,
    int ndim, int total)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=total) return;
    int tmp = idx;
    int a_idx=0, b_idx=0;
    for(int d=ndim-1; d>=0; --d){
        int coord = tmp % out_shape[d];
        tmp /= out_shape[d];
        a_idx += ((a_shape[d]==1?0:coord)*a_strides[d]);
        b_idx += ((b_shape[d]==1?0:coord)*b_strides[d]);
    }
    out[idx] = Op::apply(a[a_idx], b[b_idx]);
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
}