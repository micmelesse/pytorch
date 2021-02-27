alias drun='docker run -it --rm --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined'
alias drun_nodevice='docker run -it --rm --network=host --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $HOME/dockerx:/dockerx'

# WORK_DIR=/var/lib/jenkins/pytorch
# WORK_DIR='/dockerx/pytorch'
WORK_DIR='/root/pytorch'

# IMAGE_NAME=rocm/pytorch
# IMAGE_NAME=rocm/pytorch:rocm4.0_ubuntu18.04_py3.6_pytorch
# IMAGE_NAME=rocm/pytorch:rocm4.0.1_ubuntu18.04_py3.6_pytorch
IMAGE_NAME=rocm/pytorch-private:rocm-3221-pytorch-rocblas-tuning-lstm-fp16-rnnfp16-miopen

drun -d --name pytorch_container -w $WORK_DIR $IMAGE_NAME
# docker cp . pytorch_container:/root/pytorch
docker attach pytorch_container
