echo "testing"

cd test
PYTORCH_TEST_WITH_ROCM=1 python3.6 test_spectral_ops.py TestFFTCUDA -v