#!/bin/bash
# Usage | ./dist.sh train_pcsr.py --config configs/carn-pcsr-phase0.yaml --gpu 0
SCRIPT=$1
shift
ARGS=("$@")

for ((i=0; i<${#ARGS[@]}; i++)); do
    if [[ ${ARGS[i]} == "--gpu" ]]; then
        GPU=${ARGS[i+1]}
        unset ARGS[i]
        unset ARGS[i+1]
        break
    fi
done

ARGS=("${ARGS[@]}")
NPROC_PER_NODE=$(echo $GPU | tr -cd ',' | wc -c)
let NPROC_PER_NODE+=1
FREE_PORT=$(python find_port.py)
echo free port: $FREE_PORT
CUDA_VISIBLE_DEVICES=$GPU python -m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE --master_port=$FREE_PORT $SCRIPT ${ARGS[@]}