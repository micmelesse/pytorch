echo "testing"

# PYTORCH_TEST_WITH_ROCM=1 python test/test_spectral_ops.py --verbose \
#     2>&1 | tee scripts/amd/test_spectral.log

# python test/test_spectral_ops.py --verbose \
#     2>&1 | tee scripts/amd/test_spectral.log


# python test/test_spectral_ops.py --verbose TestFFTCUDA.test_reference_nd_fft_irfftn_cuda_complex128 #hipfftExecZ2D

# python test/test_spectral_ops.py --verbose TestFFTCUDA.test_reference_nd_fft_irfftn_cuda_complex64 #hipfftExecC2R

# python test/test_spectral_ops.py --verbose TestFFTCUDA.test_reference_nd_fft_irfftn_cuda_float64 #hipfftExecZ2D

python test/test_spectral_ops.py --verbose TestFFTCUDA.test_reference_nd_fft_irfftn_cuda_float32 #hipfftExecC2R