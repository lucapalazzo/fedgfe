#!/bin/bash
#SBATCH --job-name=a2v_10n_sa
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

# Step 1: SA (Stand-Alone) - No aggregation
echo "=========================================="
echo "Step 1/3: SA - No Aggregation"
echo "=========================================="
python -u system/main.py --config configs/a2v_vegas_10n_10c_400s_sa.json
date

# Step 2: FedAvg aggregation
echo "=========================================="
echo "Step 2/3: FedAvg Aggregation"
echo "=========================================="
python -u system/main.py --config configs/a2v_vegas_10n_10c_400s_avg.json
date

# Step 3: Train adapters
echo "=========================================="
echo "Step 3/3: Training Adapters"
echo "=========================================="
python -u system/main.py --config configs/a2v_vegas_10n_10c_400s_train_adapters.json
date

echo "=========================================="
echo "SA batch completed!"
echo "=========================================="
