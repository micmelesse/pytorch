echo "testing"

# ALL TESTS

# PYTORCH_TEST_WITH_ROCM=1 python test/test_spectral_ops.py --verbose \
#     2>&1 | tee scripts/amd/test_spectral_with_rocm.log

# python test/test_spectral_ops.py --verbose \
#     2>&1 | tee scripts/amd/test_spectral.log

# PASSING TESTS

# FAILING TESTS
python test/test_spectral_ops.py --verbose TestFFTCUDA.test_fft2_numpy_cuda_float64 #hipfftExecZ2D
# python test/test_spectral_ops.py --verbose TestFFTCUDA.test_fft2_numpy_cuda_complex128 #hipfftExecZ2D