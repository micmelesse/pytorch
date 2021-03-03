set -e

# build pytorch
# sh scripts/amd/build_pytorch_jenkins.sh
sh scripts/amd/build_pytorch_develop.sh

# run test script
sh scripts/amd/test_spectral.sh
