cd test
gdb -ex break istft --args python test_spectral_ops.py --verbose TestFFTCUDA.test_batch_istft_cuda
# python -m pdb test_spectral_ops.py --verbose TestFFTCUDA.test_batch_istft_cuda
cd ..
