set -o xtrace

CONTAINER_NAME=bold_hellman


WORK_DIR='/root/pytorch'

SRC=.
# SRC=test
# SRC=scripts

docker cp $SRC $CONTAINER_NAME:$WORK_DIR