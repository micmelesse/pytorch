export MAX_JOBS=16
pip uninstall torch -y

# BUILD_ENVIRONMENT is set in rocm docker conatiners
export PYTORCH_ROCM_ARCH="gfx908"
bash .jenkins/pytorch/build.sh