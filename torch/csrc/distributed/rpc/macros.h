#pragma once

#if defined(USE_ROCM) && !defined(__HIP_PLATFORM_HCC__)
#define USE_CUDA_NOT_ROCM
#endif
