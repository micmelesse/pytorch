#include "hip/hip_runtime.h"
#include <c10d/test/CUDATest.hpp>
#include <ATen/hip/Exceptions.h>

namespace c10d {
namespace test {

namespace {
__global__ void waitClocks(const uint64_t count) {
  clock_t start = clock64();
  clock_t offset = 0;
  while (offset < count) {
    offset = clock() - start;
  }
}

} // namespace

void cudaSleep(at::hip::HIPStreamMasqueradingAsCUDA& stream, uint64_t clocks) {
  waitClocks<<<1, 1, 0, stream.stream()>>>(clocks);
  C10_HIP_KERNEL_LAUNCH_CHECK();
}

int cudaNumDevices() {
  int n = 0;
  C10_HIP_CHECK_WARN(hipGetDeviceCount(&n));
  return n;
}

} // namespace test
} // namespace c10d
