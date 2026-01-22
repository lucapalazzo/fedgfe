#!/bin/bash
#SBATCH --job-name=a2v_kd
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
python -u system/main.py --config configs/a2v_vegas_2n_2c_200s_kd.json
date
# conda run -n flvit python system/main.py --config configs/a2v_generator_1n_50c.json
# date
