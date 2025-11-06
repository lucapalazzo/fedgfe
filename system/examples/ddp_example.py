#!/usr/bin/env python3
"""
Example script showing how to use FedGFE with DDP support.

Usage examples:

1. Single process (no DDP):
   python ddp_example.py --dataset cifar10 --num_clients 4

2. Multi-process DDP (2 GPUs):
   torchrun --nproc_per_node=2 ddp_example.py --dataset cifar10 --num_clients 4

3. Multi-node DDP (2 nodes, 2 GPUs each):
   # On node 0:
   torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=192.168.1.100 ddp_example.py --dataset cifar10 --num_clients 4
   
   # On node 1:
   torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=192.168.1.100 ddp_example.py --dataset cifar10 --num_clients 4
"""

import os
import sys
import argparse

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flcore.servers.servergfe_ddp import FedGFEDDP
from utils.ddp_utils import initialize_ddp_args, verify_ddp_setup, DDPErrorHandler, parse_gpu_selection
from utils.gpu_utils import print_gpu_status, recommend_gpu_configuration, validate_gpu_selection
import torch


def create_args():
    """Create sample arguments for testing DDP functionality."""
    parser = argparse.ArgumentParser(description='FedGFE with DDP Example')
    
    # Basic federated learning arguments
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num_clients', type=int, default=4)
    parser.add_argument('--global_rounds', type=int, default=10)
    parser.add_argument('--ssl_rounds', type=int, default=5)
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--local_learning_rate', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # DDP GPU selection arguments
    parser.add_argument('--gpu_ids', type=str, default=None, 
                       help='GPU IDs to use (e.g., "0,2,3" or "0-3" or "all")')
    parser.add_argument('--processes_per_gpu', type=int, default=1,
                       help='Number of processes per GPU (default: 1)')
    
    # Model arguments
    parser.add_argument('--nodes_backbone_model', type=str, default='vit_base_patch16_224')
    parser.add_argument('--nodes_pretext_tasks', type=str, default='patch_ordering')
    parser.add_argument('--nodes_downstream_tasks', type=str, default='classification')
    parser.add_argument('--nodes_training_sequence', type=str, default='both')
    
    # Dataset arguments
    parser.add_argument('--dataset_image_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--num_classes', type=int, default=10)
    
    # Training arguments
    parser.add_argument('--model_optimizer', type=str, default='AdamW')
    parser.add_argument('--model_optimizer_weight_decay', type=float, default=1e-4)
    parser.add_argument('--model_optimizer_momentum', type=float, default=0.9)
    parser.add_argument('--learning_rate_schedule', type=str, default='none')
    
    # Other arguments with defaults
    parser.add_argument('--no_wandb', action='store_true', default=True)
    parser.add_argument('--cls_token_only', action='store_true', default=False)
    parser.add_argument('--downstream_loss_operation', type=str, default='crossentropy')
    parser.add_argument('--no_downstream_tasks', action='store_true', default=False)
    parser.add_argument('--loss_weighted', action='store_true', default=False)
    parser.add_argument('--dataset_limit', type=int, default=0)
    parser.add_argument('--privacy', type=str, default='none')
    parser.add_argument('--dp_sigma', type=float, default=0.0)
    
    # Server arguments
    parser.add_argument('--save_folder_name', type=str, default='temp_results')
    parser.add_argument('--model_aggregation', type=str, default='fedavg')
    parser.add_argument('--model_aggregation_weighted', action='store_true', default=False)
    parser.add_argument('--model_aggregation_random', action='store_true', default=False)
    parser.add_argument('--federation_grid_metrics', action='store_true', default=False)
    
    # Checkpoint arguments
    parser.add_argument('--model_backbone_load_checkpoint', action='store_true', default=False)
    parser.add_argument('--model_backbone_save_checkpoint', action='store_true', default=False)
    parser.add_argument('--model_backbone_checkpoint', type=str, default='')
    
    args = parser.parse_args()
    
    # Initialize DDP-specific arguments
    args = initialize_ddp_args(args)
    
    return args


def run_federated_learning(args):
    """Run federated learning with DDP support."""
    
    # Print GPU status
    print_gpu_status()
    
    # Handle GPU selection
    if hasattr(args, 'gpu_ids') and args.gpu_ids:
        selected_gpus = parse_gpu_selection(args.gpu_ids)
        if selected_gpus and not validate_gpu_selection(selected_gpus):
            print("Invalid GPU selection, falling back to automatic selection")
            selected_gpus = None
        args.ddp_visible_gpus = selected_gpus
    
    # Get recommendations if DDP is enabled
    if args.ddp_enabled:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        if world_size > 1:
            config = recommend_gpu_configuration(world_size)
            print(f"\nRecommended GPU configuration for {world_size} processes:")
            print(f"  - GPUs: {config['recommended_gpus']}")
            print(f"  - Processes per GPU: {config['processes_per_gpu']}")
            if config['warning']:
                print(f"  - Warning: {config['warning']}")
    
    # Verify DDP setup
    if args.ddp_enabled:
        if not verify_ddp_setup():
            print("DDP verification failed, falling back to single process mode")
            args.ddp_enabled = False
    
    # Initialize server with DDP support
    print("\nInitializing FedGFE server with DDP support...")
    
    with DDPErrorHandler():
        # Create server
        server = FedGFEDDP(args, times=[])
        
        print(f"Server initialized with {len(server.clients)} clients")
        
        try:
            # Run complete training (handles SSL and downstream automatically)
            server.train()
            
            print("Training completed successfully!")
            
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        except Exception as e:
            print(f"Training failed: {e}")
            raise
        finally:
            # Cleanup DDP resources
            if hasattr(server, 'cleanup_ddp'):
                server.cleanup_ddp()


def main():
    """Main function for the example."""
    args = create_args()
    
    print("=== FedGFE with DDP Example ===")
    print(f"DDP Enabled: {args.ddp_enabled}")
    
    if args.ddp_enabled:
        ddp_info = {
            'rank': int(os.environ.get('RANK', 0)),
            'local_rank': int(os.environ.get('LOCAL_RANK', 0)),
            'world_size': int(os.environ.get('WORLD_SIZE', 1)),
            'master_addr': os.environ.get('MASTER_ADDR', 'localhost'),
            'master_port': os.environ.get('MASTER_PORT', '12355')
        }
        print(f"DDP Config: {ddp_info}")
    
    # Run federated learning
    run_federated_learning(args)


if __name__ == "__main__":
    main()