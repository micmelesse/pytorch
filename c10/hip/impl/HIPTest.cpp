// !!! This is a file automatically generated by hipify!!!
// Just a little test file to make sure that the HIP library works

#include <c10/hip/HIPException.h>
#include <c10/hip/impl/HIPTest.h>

#include <hip/hip_runtime.h>

namespace c10 {
namespace hip {
namespace impl {

bool has_hip_gpu() {
  int count;
  C10_HIP_CHECK(hipGetDeviceCount(&count));

  return count != 0;
}

int c10_hip_test() {
  int r = 0;
  if (has_hip_gpu()) {
    C10_HIP_CHECK(hipGetDevice(&r));
  }
  return r;
}

// This function is not exported
int c10_hip_private_test() {
  return 2;
}

}}} // namespace c10::hip::impl
