#!/bin/bash
#SBATCH --job-name=gen_imgs_ckpt
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --ntasks=1
#SBATCH --time=24:00:00


cd $HOME/fedgfe
date

source $(conda info --base)/etc/profile.d/conda.sh
conda activate flvit

echo "=========================================="
echo "Generating Images from Embeddings Checkpoints"
echo "=========================================="

# Set checkpoint directory (modify as needed)
CHECKPOINT_DIR="checkpoints/embeddings"
OUTPUT_DIR="output_images/from_checkpoints"
BATCH_SIZE=2

echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo ""

# Generate images from all checkpoints
python -u system/generate_images_from_checkpoint.py \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --diffusion_type flux \
    --batch_size "$BATCH_SIZE" \
    --device cuda

date

echo "=========================================="
echo "Image generation completed!"
echo "=========================================="
