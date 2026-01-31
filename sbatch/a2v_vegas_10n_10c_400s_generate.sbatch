#!/bin/bash
#SBATCH --job-name=10n_generate
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

# Step 1: Generate Synthetic - No aggregation
echo "=========================================="
echo "Step 1/1: Generate Synthetic"
echo "=========================================="
python -u system/main.py --config configs/a2v_vegas_10n_10c_400s_generate.json
date

echo "=========================================="
echo "Generate Synthetic batch completed!"
echo "=========================================="
