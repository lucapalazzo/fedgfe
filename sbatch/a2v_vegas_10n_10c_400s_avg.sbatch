#!/bin/bash
#SBATCH --job-name=a2v_10n_avg
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
echo "Step 1/3: FedAvg Aggergation 10LE 20GR"
echo "=========================================="
python -u system/main.py --config configs/a2v_vegas_10n_10c_400s_avg.json
date

# Step 2: FedAvg aggregation
# echo "=========================================="
# echo "Step 2/3: FedAvg Aggregation 5LE 40GT"
# echo "=========================================="
# python -u system/main.py --config configs/a2v_vegas_10n_10c_400s_avg_5le_40gr.json
# date

echo "=========================================="
echo "AVG batch completed!"
echo "=========================================="
