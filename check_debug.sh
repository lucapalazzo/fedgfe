#!/bin/bash

SESSION_NAME="debugvsc"
JOB_NAME="debug_job"
PORT=5678

# 1. Controlla se esiste un job Slurm attivo con quel nome
JOB_STATUS=$(squeue -n $JOB_NAME -t R -h -o %t)

if [ "$JOB_STATUS" == "R" ]; then
    echo "Il job Slurm è già in esecuzione."
else
    echo "Avvio nuovo job Slurm con tmux e debugpy..."
    # Lancia in background così VS Code può procedere
    srun --job-name=$JOB_NAME --pty \
    tmux new-session -s $SESSION_NAME -d \
    "python3 -m debugpy --listen 0.0.0.0:$PORT --wait-for-client mio_script.py"
    
    # Aspetta un attimo che il socket si apra
    sleep 2
fi
