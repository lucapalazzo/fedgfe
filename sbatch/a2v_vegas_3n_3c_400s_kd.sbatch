#!/bin/bash
#SBATCH --job-name=3n_kd
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
echo "Step 1/1: Training Generators"
echo "=========================================="
python -u system/main.py --config configs/a2v_vegas_3n_3c_400s_kd.json
date

echo "=========================================="
echo "KD batch completed!"
echo "=========================================="
