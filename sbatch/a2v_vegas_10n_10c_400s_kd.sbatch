#!/bin/bash
#SBATCH --job-name=a2v_10n_kd
#SBATCH --partition=gpu
#SBATCH --qos=train
#SBATCH --gres=gpu:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --ntasks=1


cd $HOME/fedgfe
date

source $(conda info --base)/etc/profile.d/conda.sh
conda activate flvit

# Step 1: Train generators (per-class VAE generators)
echo "=========================================="
echo "Step 1/3: Training Generators"
echo "=========================================="
python -u system/main.py --config configs/a2v_vegas_10n_10c_400s_train_generator.json
date

# Step 2: Generate images from pretrained models
echo "=========================================="
echo "Step 2/3: Image Generation"
echo "=========================================="
python -u system/main.py --config configs/a2v_vegas_10n_10c_400s_generate.json
date

# Step 3: Knowledge Distillation aggregation
echo "=========================================="
echo "Step 3/3: KD Aggregation"
echo "=========================================="
python -u system/main.py --config configs/a2v_vegas_10n_10c_400s_kd.json
date

echo "=========================================="
echo "KD batch completed!"
echo "=========================================="
