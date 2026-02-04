#pragma once
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <hipblaslt/hipblaslt.h>
#include <miopen/miopen.h>
#include <iostream>
#include <cstdlib>

// 全局 DCU Engine 宏
#define CHECK_HIP(cmd) do { \
    hipError_t e = cmd; \
    if (e != hipSuccess) { \
        std::cerr << "[HIP ERROR] " << hipGetErrorString(e) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)

#define CHECK_ROCBLAS(cmd)                                             \
    do {                                                               \
        rocblas_status status = (cmd);                                 \
        if (status != rocblas_status_success) {                        \
            fprintf(stderr,                                            \
                    "rocBLAS error %d at %s:%d\n",                     \
                    status, __FILE__, __LINE__);                       \
            std::abort();                                              \
        }                                                              \
    } while (0)

#define CHECK_HIPBLASLT(cmd)                                   \
    do {                                                       \
        hipblasStatus_t status = (cmd);                        \
        if (status != HIPBLAS_STATUS_SUCCESS) {               \
            fprintf(stderr,                                   \
                "hipBLASLt error %d at %s:%d\n",               \
                status, __FILE__, __LINE__);                   \
            abort();                                           \
        }                                                      \
    } while (0)


#define CHECK_MIOPEN(cmd) do { \
    miopenStatus_t stat = cmd; \
    if (stat != miopenStatusSuccess) { \
        std::cerr << "[MIOpen ERROR] " << miopenGetErrorString(stat) \
                  << " (" << static_cast<int>(stat) << ")" \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
} while(0)
