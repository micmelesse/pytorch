
cd test
# PYTORCH_TEST_WITH_ROCM=1 python test_nn.py --verbose \
#     2>&1 | tee ../scripts/amd/test_nn.log

python test_nn.py --verbose TestNNDeviceTypeCUDA.test_softmax_cuda_float16 
cd ..
