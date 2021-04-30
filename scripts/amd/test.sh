echo "testing"

# ALL TESTS

# PYTORCH_TEST_WITH_ROCM=1 python test/test_spectral_ops.py --verbose \
#     2>&1 | tee scripts/amd/test_spectral_with_rocm.log

# python test/test_spectral_ops.py --verbose \
#     2>&1 | tee scripts/amd/test_spectral.log

# PASSING TESTS

# FAILING TESTS
python test/test_spectral_ops.py --verbose TestFFTCUDA.test_cufft_plan_cache_cuda_float64