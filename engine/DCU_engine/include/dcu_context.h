#pragma once
#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#include <hipblaslt/hipblaslt.h>
#include <rocblas/rocblas.h>
#include <unordered_map>
#include <cstring>
#include "context.h"
#include "dcu_common.h"
#include "dcu_tensor.h"

namespace dcu {

// 卷积参数唯一标识一次卷积
struct ConvKey {
    int N, C, H, W;
    int K, R, S;
    int stride_h, stride_w;
    int pad_h, pad_w;
    int dilation_h, dilation_w;
    int groups;

    bool operator==(const ConvKey& other) const {
        return memcmp(this, &other, sizeof(ConvKey)) == 0;
    }
};

// Conv 算法缓存结构
struct ConvFwdCache {
    miopenConvFwdAlgorithm_t algo;
    size_t workspace_size;
};

// Conv 算法key
struct ConvKeyHash {
    std::size_t operator()(const ConvKey& k) const {
        size_t h = 0;
        const int* p = reinterpret_cast<const int*>(&k);
        for (int i = 0; i < sizeof(ConvKey)/sizeof(int); ++i) {
            h ^= std::hash<int>()(p[i]) + 0x9e3779b9 + (h<<6) + (h>>2);
        }
        return h;
    }
};

class DCUContext : public engine::Context {
public:
    DCUContext();
    ~DCUContext();

    void sync() override { CHECK_HIP(hipStreamSynchronize(stream_)); }

    hipStream_t& get_stream() { return stream_; }
    miopenHandle_t& get_miopen() { return miopen_; }
    hipblasLtHandle_t& get_hipblasLt() { return hipblasLt_; }
    rocblas_handle& get_rocblas() { return rocblas_; }
    std::unordered_map<ConvKey, ConvFwdCache, ConvKeyHash> conv_algo_cache_; // conv算法缓存

private:
    hipStream_t stream_;
    miopenHandle_t miopen_;
    hipblasLtHandle_t hipblasLt_;
    rocblas_handle rocblas_;
};

} // namespace dcu