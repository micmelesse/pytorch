cd scripts/amd
python3 Pytorch_LSTMModel.py --batch_size=32 --warm_up=2 --num_test=16 --distributed=False
cd ..

# cd test
# PYTORCH_TEST_WITH_ROCM=1 python test_nn.py --verbose TestNNDeviceTypeCUDA.test_variable_sequence_cuda_float16
# PYTORCH_TEST_WITH_ROCM=1 python test_nn.py --verbose TestNN.test_RNN_cpu_vs_cudnn_no_dropout
# PYTORCH_TEST_WITH_ROCM=1 python test_nn.py --verbose TestNN.test_RNN_cpu_vs_cudnn_with_dropout
# cd ..
