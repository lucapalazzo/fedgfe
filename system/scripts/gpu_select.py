#!/usr/bin/env python3
"""
Interactive GPU selection tool for FedGFE DDP.

This script helps users select optimal GPUs for distributed training
based on current GPU status and resource requirements.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.gpu_utils import get_gpu_info, print_gpu_status, recommend_gpu_configuration, select_optimal_gpus
from utils.ddp_utils import parse_gpu_selection
import argparse


def interactive_gpu_selection():
    """Interactive GPU selection with recommendations."""
    print("=== FedGFE DDP GPU Selection Tool ===\n")
    
    # Show current GPU status
    print_gpu_status()
    
    # Get user input for number of processes
    try:
        num_processes = int(input("Number of DDP processes to run: "))
        if num_processes < 1:
            print("Number of processes must be >= 1")
            return None
    except ValueError:
        print("Invalid input for number of processes")
        return None
    
    # Get recommendations
    config = recommend_gpu_configuration(num_processes)
    
    if config['recommended_gpus']:
        print(f"\nRecommended configuration for {num_processes} processes:")
        print(f"  - GPUs: {config['recommended_gpus']}")
        print(f"  - Processes per GPU: {config['processes_per_gpu']}")
        if config['warning']:
            print(f"  - Warning: {config['warning']}")
        
        # Ask user if they want to use recommendations
        use_recommended = input(f"\nUse recommended GPUs {config['recommended_gpus']}? [Y/n]: ").lower()
        if use_recommended in ['', 'y', 'yes']:
            return config['recommended_gpus']
    
    # Manual selection
    print("\nManual GPU selection:")
    print("Enter GPU IDs (examples: '0,2,3' or '0-3' or 'all'):")
    
    while True:
        gpu_input = input("GPU IDs: ").strip()
        if not gpu_input:
            print("No GPUs selected")
            return None
        
        try:
            selected_gpus = parse_gpu_selection(gpu_input)
            if not selected_gpus:
                print("No valid GPUs found in selection")
                continue
            
            print(f"Selected GPUs: {selected_gpus}")
            
            # Validate selection
            gpu_info = get_gpu_info()
            available_gpu_ids = [gpu['id'] for gpu in gpu_info]
            
            invalid_gpus = [gpu_id for gpu_id in selected_gpus if gpu_id not in available_gpu_ids]
            if invalid_gpus:
                print(f"Invalid GPU IDs: {invalid_gpus}")
                print(f"Available GPU IDs: {available_gpu_ids}")
                continue
            
            return selected_gpus
            
        except ValueError as e:
            print(f"Invalid GPU selection: {e}")
            continue


def generate_command_examples(selected_gpus, num_processes):
    """Generate example commands for the selected configuration."""
    if not selected_gpus:
        return
    
    gpu_str = ','.join(map(str, selected_gpus))
    
    print(f"\n=== Command Examples ===")
    print(f"Selected GPUs: {selected_gpus}")
    print(f"Number of processes: {num_processes}")
    
    print(f"\n1. Using run_ddp.sh script:")
    print(f"   ./scripts/run_ddp.sh multi {num_processes} \"{gpu_str}\"")
    
    print(f"\n2. Using torchrun directly:")
    print(f"   CUDA_VISIBLE_DEVICES={gpu_str} torchrun --nproc_per_node={num_processes} your_script.py")
    
    print(f"\n3. Using environment variables:")
    print(f"   export CUDA_VISIBLE_DEVICES={gpu_str}")
    print(f"   export WORLD_SIZE={num_processes}")
    print(f"   # Then run your script")
    
    print(f"\n4. Programmatic usage:")
    print(f"   from utils.ddp_utils import run_federated_ddp")
    print(f"   run_federated_ddp({num_processes}, your_function, args, visible_gpus={selected_gpus})")


def main():
    parser = argparse.ArgumentParser(description='GPU Selection Tool for FedGFE DDP')
    parser.add_argument('--processes', type=int, help='Number of DDP processes')
    parser.add_argument('--auto', action='store_true', help='Automatic selection without interaction')
    parser.add_argument('--min-memory', type=int, default=4000, help='Minimum free memory (MB)')
    parser.add_argument('--max-utilization', type=int, default=50, help='Maximum utilization (%)')
    
    args = parser.parse_args()
    
    if args.auto:
        # Automatic mode
        if not args.processes:
            print("--processes required in automatic mode")
            sys.exit(1)
        
        print(f"=== Automatic GPU Selection for {args.processes} processes ===")
        print_gpu_status()
        
        selected_gpus = select_optimal_gpus(
            args.processes, 
            min_memory_mb=args.min_memory,
            max_utilization=args.max_utilization
        )
        
        if selected_gpus:
            print(f"Selected GPUs: {selected_gpus}")
            generate_command_examples(selected_gpus, args.processes)
        else:
            print("No suitable GPUs found with current criteria")
            print(f"Criteria: min_memory={args.min_memory}MB, max_utilization={args.max_utilization}%")
    else:
        # Interactive mode
        if args.processes:
            num_processes = args.processes
            config = recommend_gpu_configuration(num_processes)
            if config['recommended_gpus']:
                selected_gpus = config['recommended_gpus']
                generate_command_examples(selected_gpus, num_processes)
            else:
                print("No suitable GPUs found for automatic recommendation")
        else:
            selected_gpus = interactive_gpu_selection()
            if selected_gpus:
                num_processes = int(input(f"\nNumber of processes for {len(selected_gpus)} selected GPUs: "))
                generate_command_examples(selected_gpus, num_processes)


if __name__ == "__main__":
    main()