#!/usr/bin/env python
"""
Test complete integration of all JSON configuration sections.
"""

import argparse
import sys
sys.path.append('/home/lpala/fedgfe/system')

from utils.config_loader import load_config_to_args

def test_complete_integration():
    """Test complete integration of all JSON sections with the updated config."""

    # Create parser with all relevant arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)

    # Basic arguments
    parser.add_argument('--algorithm', type=str, default='FedAvg')
    parser.add_argument('--num_clients', type=int, default=2)
    parser.add_argument('--global_rounds', type=int, default=100)
    parser.add_argument('--local_epochs', type=int, default=1)
    parser.add_argument('--local_learning_rate', type=float, default=0.005)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--model_optimizer', type=str, default='SGD')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--no_wandb', type=bool, default=False)

    # FedGFE specific
    parser.add_argument('--nodes_datasets', type=str, default='cifar10')
    parser.add_argument('--nodes_pretext_tasks', type=str, default='')
    parser.add_argument('--nodes_downstream_tasks', type=str, default='none')
    parser.add_argument('--nodes_training_sequence', type=str, default='both')

    # Parse with config file
    args = parser.parse_args(['--config', 'configs/jsrt_simple_classification.json'])

    print("=== BEFORE CONFIG LOADING ===")
    print(f"Algorithm: {args.algorithm}")
    print(f"Num Clients: {args.num_clients}")
    print(f"Global Rounds: {args.global_rounds}")
    print(f"Learning Rate: {args.local_learning_rate}")
    print(f"Optimizer: {args.model_optimizer}")
    print(f"Dataset: {args.dataset}")
    print(f"No WandB: {args.no_wandb}")
    print(f"Nodes Datasets: {args.nodes_datasets}")
    print(f"Pretext Tasks: {args.nodes_pretext_tasks}")
    print(f"Downstream Tasks: {args.nodes_downstream_tasks}")
    print(f"Training Sequence: {args.nodes_training_sequence}")

    # Load config
    try:
        args = load_config_to_args(args.config, args)

        print("\n=== AFTER CONFIG LOADING ===")
        print(f"Algorithm: {args.algorithm}")
        print(f"Num Clients: {args.num_clients}")
        print(f"Global Rounds: {args.global_rounds}")
        print(f"Learning Rate: {args.local_learning_rate}")
        print(f"Optimizer: {args.model_optimizer}")
        print(f"Dataset: {args.dataset}")
        print(f"No WandB: {args.no_wandb}")
        print(f"Nodes Datasets: {args.nodes_datasets}")
        print(f"Pretext Tasks: {args.nodes_pretext_tasks}")
        print(f"Downstream Tasks: {args.nodes_downstream_tasks}")
        print(f"Training Sequence: {args.nodes_training_sequence}")

        print("\n✅ Complete integration test successful!")
        print("\n📋 Configuration Summary:")
        print(f"  - {args.num_clients} clients using {args.dataset}")
        print(f"  - Training for {args.global_rounds} rounds with {args.local_epochs} local epochs")
        print(f"  - Using {args.model_optimizer} optimizer with LR {args.local_learning_rate}")
        print(f"  - WandB {'disabled' if args.no_wandb else 'enabled'}")
        print(f"  - Pretext tasks: {args.nodes_pretext_tasks or 'None'}")
        print(f"  - Training sequence: {args.nodes_training_sequence}")

        return True

    except Exception as e:
        print(f"\n❌ Error during config loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_complete_integration()