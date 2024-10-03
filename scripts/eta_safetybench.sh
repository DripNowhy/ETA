#!/bin/bash

# Initialize variables with default values
DATASET=""
SAVEDIR=""
GPU_ID=""

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dataset) DATASET="$2"; shift ;;
        --save_dir) SAVEDIR="$2"; shift ;;
        --gpu_id) GPU_ID="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Ensure the required arguments are provided
if [[ -z "$DATASET" ]]; then
    echo "Error: --DATASET is required."
    exit 1
fi

if [[ -z "$GPU_ID" ]]; then
    echo "Error: --gpu_id is required."
    exit 1
fi

# Extract dataset name from the path
DATASET_NAME=$(basename "$DATASET")

# Display the dataset name being tested
echo "Testing ETA on ${DATASET_NAME}"

# Run the Python script with the specified arguments
python eta_safetybench.py --save_dir "$SAVEDIR" --gpu_id "$GPU_ID" --dataset "$DATASET"

# Notify completion
echo "Finished running ${DATASET_NAME}"
echo "---------------------------------------"

