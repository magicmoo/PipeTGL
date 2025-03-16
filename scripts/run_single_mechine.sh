#!/bin/bash

EPOCHS=${1}
DATA=${2}
NPROC_PER_NODE=${3:-1}

export OMP_NUM_THREADS=8

if [[ $NPROC_PER_NODE -gt 1 ]]; then
    cmd="torchrun --nproc_per_node=$NPROC_PER_NODE ./scripts/pipeTrain3.py --epoch $EPOCHS --data $DATA"
else
    cmd="python ./scripts/pipeTrain3.py --epoch $EPOCHS --data $DATA"
fi

echo "Executing command: $cmd"

eval $cmd