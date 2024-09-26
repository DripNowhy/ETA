#!/bin/bash

# Set CUDA device to GPU 5
# export CUDA_LAUNCH_BLOCKING=1
export CUDA_VISIBLE_DEVICES=4,5,6

# Run the Python script with debugpy for remote debugging
python -m debugpy --listen 0.0.0.0:9501 --wait-for-client text_generation.py
