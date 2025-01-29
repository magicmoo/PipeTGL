#!/bin/bash

EPOCHS=${1}
DATA=${2}
NPROC_PER_NODE=${3:-1}

export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export GLIBC=/HOME/scw6fd5/run/glibc-2.31/build
export OMP_NUM_THREADS=8

if [[ $NPROC_PER_NODE -gt 1 ]]; then
    cmd="GCONV_PATH=${GLIBC}/iconvdata LC_ALL=C ${GLIBC}/elf/ld.so --library-path ${GLIBC}:${GLIBC}/math:${GLIBC}/elf:${GLIBC}/dlfcn:${GLIBC}/nss:${GLIBC}/nis:${GLIBC}/rt:${GLIBC}/resolv:${GLIBC}/crypt:${GLIBC}/nptl:${GLIBC}/dfp:/data/apps/cuda/12.4/targets/x86_64-linux/lib:/lib64 $(which python) ~/.conda/envs/GNNFlow/bin/torchrun --nproc_per_node=$NPROC_PER_NODE --no_python ${GLIBC}/elf/ld.so --library-path ${GLIBC}:${GLIBC}/math:${GLIBC}/elf:${GLIBC}/dlfcn:${GLIBC}/nss:${GLIBC}/nis:${GLIBC}/rt:${GLIBC}/resolv:${GLIBC}/crypt:${GLIBC}/nptl:${GLIBC}/dfp:/data/apps/cuda/12.4/targets/x86_64-linux/lib:/lib64 $(which python) ./scripts/pipeTrain3.py --model TGN --epoch $EPOCHS --data $DATA"
else
    cmd="GCONV_PATH=${GLIBC}/iconvdata LC_ALL=C ${GLIBC}/elf/ld.so --library-path ${GLIBC}:${GLIBC}/math:${GLIBC}/elf:${GLIBC}/dlfcn:${GLIBC}/nss:${GLIBC}/nis:${GLIBC}/rt:${GLIBC}/resolv:${GLIBC}/crypt:${GLIBC}/nptl:${GLIBC}/dfp:/data/apps/cuda/12.4/targets/x86_64-linux/lib:/lib64 $(which python) ./scripts/pipeTrain3.py --model TGN --epoch $EPOCHS --data $DATA"
fi

echo "Executing command: $cmd"

eval $cmd &> /HOME/scw6fd5/run/PipeTGL/output/1.log