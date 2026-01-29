#!/bin/bash

# Pre-generate AST embeddings cache for VEGAS dataset
# Simple local execution script

set -e  # Exit on error

# Default configuration
DATASET_PATH="${DATASET_PATH:-dataset/Audio/VEGAS}"
CACHE_DIR="${CACHE_DIR:-cache/ast/vegas}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --cache_dir)
            CACHE_DIR="$2"
            shift 2
            ;;
        --classes)
            SELECTED_CLASSES="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --cpu)
            DEVICE="cpu"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset_path PATH    Path to VEGAS dataset (default: dataset/Audio/VEGAS)"
            echo "  --cache_dir DIR        Cache directory (default: cache/ast/vegas)"
            echo "  --classes \"class1 class2\"  Space-separated list of classes (default: all)"
            echo "  --batch_size N         Batch size (default: 16)"
            echo "  --device DEVICE        Device (cuda or cpu, default: cuda)"
            echo "  --cpu                  Use CPU instead of GPU"
            echo "  --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Generate cache for all classes"
            echo "  $0"
            echo ""
            echo "  # Generate cache for specific classes"
            echo "  $0 --classes \"dog cat chainsaw\""
            echo ""
            echo "  # Use CPU with larger batch"
            echo "  $0 --cpu --batch_size 32"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "AST Embeddings Cache Generation"
echo "=========================================="
echo "Dataset: $DATASET_PATH"
echo "Cache dir: $CACHE_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo "=========================================="

# Build command
CMD="python system/pregenerate_ast_cache.py \
    --dataset_path \"$DATASET_PATH\" \
    --cache_dir \"$CACHE_DIR\" \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --num_workers $NUM_WORKERS"

# Add selected classes if specified
if [ ! -z "$SELECTED_CLASSES" ]; then
    echo "Selected classes: $SELECTED_CLASSES"
    CMD="$CMD --selected_classes $SELECTED_CLASSES"
else
    echo "Processing: ALL classes"
fi

echo "=========================================="

# Execute
eval $CMD

EXIT_CODE=$?

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Cache generation completed successfully!"
    echo "Cache location: $CACHE_DIR"
else
    echo "✗ Cache generation failed (exit code: $EXIT_CODE)"
fi
echo "=========================================="

exit $EXIT_CODE
