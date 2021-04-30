echo "testing"

# PYTORCH_TEST_WITH_ROCM=1 python test/test_spectral_ops.py --verbose \
#     2>&1 | tee scripts/amd/test_spectral.log

# python test/test_spectral_ops.py --verbose \
#     2>&1 | tee scripts/amd/test_spectral.log

## FAILING TEST

python test/test_spectral_ops.py --verbose TestFFTCUDA.test_cufft_plan_cache_cuda_float64