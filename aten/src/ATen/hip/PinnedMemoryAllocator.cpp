// !!! This is a file automatically generated by hipify!!!
#include <ATen/hip/PinnedMemoryAllocator.h>
#include <ATen/Context.h>
#include <ATen/Config.h>

#include <THH/THH.h>
#include <THH/THHGeneral.hpp>

#include <stdexcept>

namespace at { namespace cuda {

at::Allocator* getPinnedMemoryAllocator() {
  auto state = globalContext().lazyInitCUDA();
  return state->hipHostAllocator;
}

}} // namespace at::cuda
