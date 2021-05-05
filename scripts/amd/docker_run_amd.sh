set -o xtrace

alias drun='sudo docker run -it --rm --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined'
alias drun_nodevice='sudo docker run -it --rm --network=host --ipc=host --shm-size 16G --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined'

VOLUMES="-v $HOME/dockerx:/dockerx -v /data:/data"

# WORK_DIR='/var/lib/jenkins/pytorch'
# WORK_DIR='/dockerx/pytorch'
WORK_DIR='/root/pytorch'

# IMAGE_NAME=rocm/pytorch
# IMAGE_NAME=rocm/pytorch:rocm4.0_ubuntu18.04_py3.6_pytorch
# IMAGE_NAME=rocm/pytorch:rocm4.0.1_ubuntu18.04_py3.6_pytorch
# IMAGE_NAME=rocm/pytorch:rocm4.1.1_ubuntu18.04_py3.6_pytorch
# IMAGE_NAME=rocm/pytorch-private:rocm4.1.1_ubuntu18.04_py3.6_pytorch_cufft_plan_cache_issue
# IMAGE_NAME=compute-artifactory.amd.com:5000/rocm-plus-docker/framework/compute-rocm-rel-4.2:16_ubuntu18.04_py3.6_pytorch_rocm4.2_internal_testing_b2c58d0_18
IMAGE_NAME=rocm/pytorch-private:16_ubuntu18.04_py3.6_pytorch_rocm4.2_internal_testing_b2c58d0_18_cufft_plan_cache_issue


CONTAINER_ID=$(drun -d -w $WORK_DIR $VOLUMES $IMAGE_NAME)
echo "CONTAINER_ID: $CONTAINER_ID"
docker cp . $CONTAINER_ID:$WORK_DIR
docker attach $CONTAINER_ID
docker stop $CONTAINER_ID
docker rm $CONTAINER_ID
