#include <torch/csrc/autograd/profiler.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <roctracer/roctx.h>

#include <sstream>

namespace torch { namespace autograd { namespace profiler {

namespace {

static inline void cudaCheck(hipError_t result, const char * file, int line) {
  if(result != hipSuccess) {
    std::stringstream ss;
    ss << file << ":" << line << ": ";
    if (result == hipErrorInitializationError) {
      // It is common for users to use DataLoader with multiple workers
      // and the autograd profiler. Throw a nice error message here.
      ss << "CUDA initialization error. "
         << "This can occur if one runs the profiler in CUDA mode on code "
         << "that creates a DataLoader with num_workers > 0. This operation "
         << "is currently unsupported; potential workarounds are: "
         << "(1) don't use the profiler in CUDA mode or (2) use num_workers=0 "
         << "in the DataLoader or (3) Don't profile the data loading portion "
         << "of your code. https://github.com/pytorch/pytorch/issues/6313 "
         << "tracks profiler support for multi-worker DataLoader.";
    } else {
      ss << hipGetErrorString(result);
    }
    throw std::runtime_error(ss.str());
  }
}
#define TORCH_CUDA_CHECK(result) cudaCheck(result,__FILE__,__LINE__);

struct CUDAMethods : public CUDAStubs {
  void record(int* device, CUDAEventStub* event, int64_t* cpu_ns) const override {
    TORCH_CUDA_CHECK(hipGetDevice(device));
    ihipEvent_t* cuda_event_ptr;
    TORCH_CUDA_CHECK(hipEventCreate(&cuda_event_ptr));
    *event = std::shared_ptr<ihipEvent_t>(cuda_event_ptr, [](ihipEvent_t* ptr) {
      TORCH_CUDA_CHECK(hipEventDestroy(ptr));
    });
    auto stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    *cpu_ns = getTime();
    TORCH_CUDA_CHECK(hipEventRecord(cuda_event_ptr, stream));
  }

  float elapsed(const CUDAEventStub* event, const CUDAEventStub* event2) const override{
    TORCH_CUDA_CHECK(hipEventSynchronize(event->get()));
    TORCH_CUDA_CHECK(hipEventSynchronize(event2->get()));
    float ms;
    TORCH_CUDA_CHECK(hipEventElapsedTime(&ms, event->get(), event2->get()));
    return ms*1000.0;
  }

  void roctxMarkA(const char* name) const override {
    ::roctxMark(name);
  }

  void roctxRangePushA(const char* name) const override {
    ::roctxRangePushA(name);
  }

  void roctxRangePop() const override {
    ::roctxRangePop();
  }

  void onEachDevice(std::function<void(int)> op) const override {
    at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard;
    int count = at::cuda::device_count();
    for(int i = 0; i < count; i++) {
      device_guard.set_index(i);
      op(i);
    }
  }

  void synchronize() const override {
    hipDeviceSynchronize();
  }

  bool enabled() const override {
    return true;
  }
};

struct RegisterCUDAMethods {
  RegisterCUDAMethods() {
    static CUDAMethods methods;
    registerCUDAMethods(&methods);
  }
};
RegisterCUDAMethods reg;

} // namespaces
} // namespace profiler
} // namespace autograd
} // namespace torch
