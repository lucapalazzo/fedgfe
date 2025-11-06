#!/bin/bash

# Script per avviare FedGFE con supporto DDP
# Usage:
#   ./run_ddp.sh single [gpu_ids]              # Single process
#   ./run_ddp.sh multi <num_procs> [gpu_ids]   # Multi-process
#
# Examples:
#   ./run_ddp.sh single                # Single process, all GPUs
#   ./run_ddp.sh single "0,2"          # Single process, GPUs 0,2
#   ./run_ddp.sh multi 2               # 2 processes, all GPUs
#   ./run_ddp.sh multi 2 "0,1"         # 2 processes, GPUs 0,1
#   ./run_ddp.sh multi 4 "0-3"         # 4 processes, GPUs 0,1,2,3

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EXAMPLE_SCRIPT="$PROJECT_DIR/examples/ddp_example.py"

# Default arguments
DATASET="cifar10"
NUM_CLIENTS=4
GLOBAL_ROUNDS=10
SSL_ROUNDS=5
BATCH_SIZE=32
LEARNING_RATE=0.01

# Function to run single process
run_single() {
    local gpu_ids=$1
    
    echo "Running FedGFE in single process mode..."
    if [ -n "$gpu_ids" ]; then
        echo "Using GPUs: $gpu_ids"
        export CUDA_VISIBLE_DEVICES="$gpu_ids"
    fi
    
    cd "$PROJECT_DIR"
    python "$EXAMPLE_SCRIPT" \
        --dataset "$DATASET" \
        --num_clients "$NUM_CLIENTS" \
        --global_rounds "$GLOBAL_ROUNDS" \
        --ssl_rounds "$SSL_ROUNDS" \
        --batch_size "$BATCH_SIZE" \
        --local_learning_rate "$LEARNING_RATE" \
        ${gpu_ids:+--gpu_ids "$gpu_ids"}
}

# Function to run multi-process DDP
run_multi() {
    local nproc=$1
    local gpu_ids=$2
    
    if [ -z "$nproc" ]; then
        nproc=2
    fi
    
    echo "Running FedGFE with DDP using $nproc processes..."
    
    # Set GPU visibility if specified
    if [ -n "$gpu_ids" ]; then
        echo "Using GPUs: $gpu_ids"
        export CUDA_VISIBLE_DEVICES="$gpu_ids"
        
        # Count available GPUs after setting CUDA_VISIBLE_DEVICES
        IFS=',' read -ra GPU_ARRAY <<< "$gpu_ids"
        available_gpu_count=${#GPU_ARRAY[@]}
        echo "Available GPUs after selection: $available_gpu_count"
        
        if [ "$nproc" -gt "$available_gpu_count" ]; then
            echo "Warning: Requested $nproc processes but only $available_gpu_count GPUs selected"
            echo "Consider reducing processes or selecting more GPUs"
        fi
    else
        # Check if GPUs are available
        if ! command -v nvidia-smi &> /dev/null; then
            echo "Warning: nvidia-smi not found, using CPU backend"
            export CUDA_VISIBLE_DEVICES=""
        else
            gpu_count=$(nvidia-smi --list-gpus | wc -l)
            echo "Available GPUs: $gpu_count"
            
            if [ "$nproc" -gt "$gpu_count" ]; then
                echo "Warning: Requested $nproc processes but only $gpu_count GPUs available"
            fi
        fi
    fi
    
    cd "$PROJECT_DIR"
    torchrun \
        --nproc_per_node="$nproc" \
        --nnodes=1 \
        --node_rank=0 \
        --master_addr=localhost \
        --master_port=29500 \
        "$EXAMPLE_SCRIPT" \
        --dataset "$DATASET" \
        --num_clients "$NUM_CLIENTS" \
        --global_rounds "$GLOBAL_ROUNDS" \
        --ssl_rounds "$SSL_ROUNDS" \
        --batch_size "$BATCH_SIZE" \
        --local_learning_rate "$LEARNING_RATE" \
        ${gpu_ids:+--gpu_ids "$gpu_ids"}
}

# Function to show usage
show_usage() {
    echo "Usage: $0 <mode> [num_processes] [gpu_ids]"
    echo ""
    echo "Modes:"
    echo "  single [gpu_ids]              Run in single process mode"
    echo "  multi num_procs [gpu_ids]     Run with DDP using num_procs processes"
    echo ""
    echo "GPU Selection (gpu_ids):"
    echo "  \"0,2,3\"      Use specific GPUs 0, 2, and 3"
    echo "  \"0-3\"        Use GPU range 0 to 3 (0,1,2,3)"
    echo "  \"all\"        Use all available GPUs (default)"
    echo "  (empty)       Use all available GPUs"
    echo ""
    echo "Examples:"
    echo "  $0 single                    # Single process, all GPUs"
    echo "  $0 single \"0,2\"             # Single process, GPUs 0,2"
    echo "  $0 multi 2                  # DDP with 2 processes, all GPUs"
    echo "  $0 multi 2 \"0,1\"            # DDP with 2 processes, GPUs 0,1"
    echo "  $0 multi 4 \"0-3\"            # DDP with 4 processes, GPUs 0,1,2,3"
    echo ""
    echo "Environment variables:"
    echo "  DATASET=$DATASET"
    echo "  NUM_CLIENTS=$NUM_CLIENTS"
    echo "  GLOBAL_ROUNDS=$GLOBAL_ROUNDS"
    echo "  SSL_ROUNDS=$SSL_ROUNDS"
    echo "  BATCH_SIZE=$BATCH_SIZE"
    echo "  LEARNING_RATE=$LEARNING_RATE"
}

# Check if torchrun is available
check_torchrun() {
    if ! command -v torchrun &> /dev/null; then
        echo "Error: torchrun not found. Please install PyTorch with distributed support."
        echo "You can install it with: pip install torch torchvision torchaudio"
        exit 1
    fi
}

# Function to list available GPUs
list_gpus() {
    echo "Available GPUs:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --list-gpus | nl -v0 | sed 's/^[ \t]*/  /'
    else
        echo "  No NVIDIA GPUs detected or nvidia-smi not available"
    fi
}

# Main script logic
case "$1" in
    "single")
        run_single "$2"
        ;;
    "multi")
        check_torchrun
        run_multi "$2" "$3"
        ;;
    "list-gpus")
        list_gpus
        ;;
    *)
        show_usage
        echo ""
        list_gpus
        exit 1
        ;;
esac