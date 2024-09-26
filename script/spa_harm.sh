#!/bin/bash

# export CUDA_VISIBLE_DEVICES=2

# 定义不同的alpha超参数值
post_thresholds=(0.04 0.06 0.08 0.1)
pre_thresholds=(12.0 14.0 16.0 18.0)

# 循环遍历每个post_threshold和pre_threshold值
for post_threshold in "${post_thresholds[@]}"; do
    for pre_threshold in "${pre_thresholds[@]}"; do
        echo "Running spa_harm_generation.py with post_threshold=${post_threshold} and pre_threshold=${pre_threshold}"
        python spa_harm_generation.py --post_threshold "$post_threshold" --pre_threshold "$pre_threshold" --gpu_id 1
        echo "Finished running with post_threshold=${post_threshold}, pre_threshold=${pre_threshold}"
        echo "---------------------------------------"
    done
done
