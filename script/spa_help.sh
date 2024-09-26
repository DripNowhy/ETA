#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

# 定义不同的alpha超参数值
alphas=(0.12 0.15 0.2)

# 循环遍历每个alpha值
for alpha in "${alphas[@]}"; do
    echo "Running spa_help_generation.py with alpha=${alpha}"
    python spa_help_generation.py --alpha "$alpha"
    echo "Finished running with alpha=${alpha}"
    echo "---------------------------------------"
done
