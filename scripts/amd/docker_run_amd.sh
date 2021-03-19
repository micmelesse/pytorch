alias drun='sudo docker run -it --rm --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined'
alias drun_nodevice='sudo docker run -it --rm --network=host --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined'

VOLUMES="-v $HOME/dockerx:/dockerx -v /data:/data"

# WORK_DIR='-w /var/lib/jenkins/pytorch'
# WORK_DIR='-w /dockerx/pytorch'
WORK_DIR='-w /root/pytorch'

# IMAGE_NAME=rocm/pytorch
# IMAGE_NAME=rocm/pytorch:rocm4.0_ubuntu18.04_py3.6_pytorch
# IMAGE_NAME=rocm/pytorch:rocm4.0.1_ubuntu18.04_py3.6_pytorch
# IMAGE_NAME=rocm/pytorch-private:rocm4.0.1_ubuntu18.04_py3.6_pytorch_master
# IMAGE_NAME=rocm/pytorch-private:rocm4.0.1_ubuntu18.04_py3.6_pytorch_master-with-fft-changes
# IMAGE_NAME=compute-artifactory.amd.com:5000/rocm-plus-docker/framework/compute-rocm-rel-4.1:21_ubuntu18.04_py3.6_pytorch_rocm4.1_internal_testing_169a263_30
# IMAGE_NAME=rocm/pytorch-private:21_ubuntu18.04_py3.6_pytorch_rocm4.1_internal_testing_169a263_30
IMAGE_NAME=rocm/pytorch-private:21_ubuntu18.04_py3.6_pytorch_rocm4.1_internal_testing_169a263_30_master-with-fft-changes

CONTAINER_ID=$(drun -d $WORK_DIR $VOLUMES $IMAGE_NAME)
echo "CONTAINER_ID: $CONTAINER_ID"
docker cp . $CONTAINER_ID:/root/pytorch
docker attach $CONTAINER_ID
docker stop $CONTAINER_ID
docker rm $CONTAINER_ID
