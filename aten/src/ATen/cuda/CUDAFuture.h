#pragma once

#include <functional>
#include <vector>

#include <ATen/core/ivalue.h>
#include <ATen/core/ivalue_inl.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/core/Device.h>
#include <c10/macros/Export.h>
#include <c10/util/intrusive_ptr.h>

namespace at {
namespace cuda {

struct TORCH_CUDA_CPP_API CUDAFuture final : at::ivalue::Future {
 public:
  explicit CUDAFuture(at::TypePtr type);

  c10::intrusive_ptr<Future> createInstance(at::TypePtr type) override;

 protected:
  void preMarkCompletedHook(
      const at::IValue& value,
      c10::optional<std::vector<std::reference_wrapper<const at::DataPtr>>>
          dataPtrs) override;

  std::function<void(void)> wrapCallback(
      std::function<void(void)> callback) override;

  void postWaitHook(const at::IValue& value) override;

 private:
  // The device that was current when markCompleted was called, which we'll
  // restore when invoking callbacks.
  c10::DeviceIndex currentDevice_;

  // The events that correspond to the completion of the async I/O kernels. They
  // are recorded on the appropriate streams when the future is marked completed
  // and can then be queried/waited/blocked on. There is one event for each
  // distinct device on which the value's tensors reside.
  std::vector<at::cuda::CUDAEvent> cudaEvents_;

  // A cached version of the data ptrs extracted from the value when the future
  // is first marked completed.
  std::vector<std::reference_wrapper<const at::DataPtr>> dataPtrs_;
};

} // namespace cuda
} // namespace at
