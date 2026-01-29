#!/bin/bash

# Script to launch Jupyter Notebook with srun
# Usage: bash sbatch/jupyter_notebook.sh

echo "Starting Jupyter Notebook session..."
echo "=================================="

# Get a compute node with GPU
srun --partition=gpu \
     --qos=train \
     --gres=gpu:1 \
     --time=4:00:00 \
     --ntasks=1 \
     --pty bash -c "
cd $HOME/fedgfe
source \$(conda info --base)/etc/profile.d/conda.sh
conda activate flvit

# Get the hostname and a random port
HOSTNAME=\$(hostname)
PORT=\$(shuf -i 8888-9999 -n 1)

echo ''
echo '=================================='
echo 'Jupyter Notebook is starting...'
echo 'Hostname: '\$HOSTNAME
echo 'Port: '\$PORT
echo '=================================='
echo ''
echo 'To connect from your local machine, run:'
echo 'ssh -L '\$PORT':'\$HOSTNAME':'\$PORT' lpala@<login-node>'
echo ''
echo 'Then open in your browser:'
echo 'http://localhost:'\$PORT
echo '=================================='
echo ''

jupyter notebook --no-browser --port=\$PORT --ip=0.0.0.0
"
