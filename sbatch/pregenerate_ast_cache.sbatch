#!/bin/bash
#SBATCH --job-name=ast_cache_gen
#SBATCH --output=logs/ast_cache_gen-%j.out
#SBATCH --error=logs/ast_cache_gen-%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# Pre-generate AST embeddings cache for VEGAS dataset
# This script runs the cache generation on SLURM cluster

echo "=========================================="
echo "AST Embeddings Cache Generation"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "=========================================="

# Activate conda environment (adjust to your environment name)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate fedgfe

# Create logs directory if it doesn't exist
mkdir -p logs

# Project directory
cd /home/lpala/fedgfe

# Configuration
DATASET_PATH="dataset/Audio/VEGAS"
CACHE_DIR="cache/ast/vegas"
BATCH_SIZE=32
DEVICE="cuda"
NUM_WORKERS=8
SAVE_EVERY=200

# Optional: Specify classes to process (comment out to process all)
# SELECTED_CLASSES="baby_cry chainsaw dog drum fireworks helicopter printer rail_transport snoring water_flowing"

# Run cache generation
echo "Starting cache generation..."
echo "Dataset: $DATASET_PATH"
echo "Cache dir: $CACHE_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo "=========================================="

if [ -z "$SELECTED_CLASSES" ]; then
    # Process all classes
    python system/pregenerate_ast_cache.py \
        --dataset_path "$DATASET_PATH" \
        --cache_dir "$CACHE_DIR" \
        --batch_size "$BATCH_SIZE" \
        --device "$DEVICE" \
        --num_workers "$NUM_WORKERS" \
        --save_every "$SAVE_EVERY"
else
    # Process selected classes
    python system/pregenerate_ast_cache.py \
        --dataset_path "$DATASET_PATH" \
        --cache_dir "$CACHE_DIR" \
        --selected_classes $SELECTED_CLASSES \
        --batch_size "$BATCH_SIZE" \
        --device "$DEVICE" \
        --num_workers "$NUM_WORKERS" \
        --save_every "$SAVE_EVERY"
fi

EXIT_CODE=$?

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Cache generation completed successfully!"
    echo "Cache location: $CACHE_DIR"
else
    echo "Cache generation failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
