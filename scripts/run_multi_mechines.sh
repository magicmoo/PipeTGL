#!/bin/bash

EPOCHS=${1}
DATA=${2}
NPROC_PER_NODE=${3}
NNODES=${4}

export OMP_NUM_THREADS=8

# MASTERADDR and MASTERPORT are the address and port of the host mechine, NODERANK should be 0,1,2,3... on each machine, respectively
NODERANK = "0"
MASTERADDR = ""
MASTERPORT = ""

cmd = "torchrun --nproc_per_node=$NPROC_PER_NODE --nnodes=$NNODES --node_rank=$NODERANK --master_addr=$MASTERADDR --master_port=$MASTERPORT ./scripts/pipeTrain_multi_node.py --epoch $EPOCHS --data $DATA"

echo "Executing command: $cmd"

eval $cmd