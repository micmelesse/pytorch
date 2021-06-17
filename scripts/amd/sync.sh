set -o xtrace

CONTAINER_NAME=epic_swartz


WORK_DIR='/root/pytorch'

SRC=.
# SRC=test
# SRC=scripts

docker cp $SRC $CONTAINER_NAME:$WORK_DIR