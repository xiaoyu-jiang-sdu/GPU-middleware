#pragma once
#include <hip/hip_runtime.h>
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