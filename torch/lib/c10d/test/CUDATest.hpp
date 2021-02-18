#pragma once

#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>

namespace c10d {
namespace test {

#ifdef _WIN32
#define EXPORT_TEST_API __declspec(dllexport)
#else
#define EXPORT_TEST_API
#endif

EXPORT_TEST_API void cudaSleep(at::hip::HIPStreamMasqueradingAsCUDA& stream, uint64_t clocks);

EXPORT_TEST_API int cudaNumDevices();

} // namespace test
} // namespace c10d
