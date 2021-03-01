alias drun='sudo docker run -it --rm --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined'
alias drun_nodevice='sudo docker run -it --rm --network=host --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined'

VOLUMES="-v $HOME/dockerx:/dockerx -v /data:/data"

# WORK_DIR=/var/lib/jenkins/pytorch
# WORK_DIR='/dockerx/pytorch'
WORK_DIR='/root/pytorch'

# IMAGE_NAME=rocm/pytorch
# IMAGE_NAME=rocm/pytorch:rocm4.0_ubuntu18.04_py3.6_pytorch
# IMAGE_NAME=rocm/pytorch:rocm4.0.1_ubuntu18.04_py3.6_pytorch
# IMAGE_NAME=rocm/pytorch-private:rocm4.0.1_ubuntu18.04_py3.6_pytorch_hipfft_issue
# IMAGE_NAME=compute-artifactory.amd.com:5000/rocm-plus-docker/framework/compute-rocm-rel-4.1:11_ubuntu18.04_py3.6_pytorch_rocm4.1_internal_testing_5ce11d0_14
IMAGE_NAME=rocm/pytorch-private:hipfft_4.1rc2_fail

drun -d --name pytorch_container -w $WORK_DIR $VOLUMES $IMAGE_NAME
docker cp . pytorch_container:/root/pytorch
docker attach pytorch_container
