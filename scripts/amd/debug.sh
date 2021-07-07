gdb -ex "set breakpoint pending on" \
    -ex 'break exec_cufft_plan' \
    --args python test/test_spectral_ops.py --verbose TestFFTCUDA.test_reference_1d_fft_hfft_cuda_complex64