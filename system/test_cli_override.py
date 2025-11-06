#!/usr/bin/env python
"""
Test CLI argument override functionality - CLI should take precedence over JSON.
"""

import argparse
import sys
sys.path.append('/home/lpala/fedgfe/system')

from utils.config_loader import load_config_to_args

def test_cli_override():
    """Test that CLI arguments override JSON values correctly."""

    # Create parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--algorithm', type=str, default='FedAvg')
    parser.add_argument('--num_clients', type=int, default=2)
    parser.add_argument('--global_rounds', type=int, default=100)
    parser.add_argument('--local_learning_rate', type=float, default=0.005)
    parser.add_argument('--no_wandb', action='store_true', default=False)

    print("=== TEST 1: JSON values only ===")
    # Parse with config file only
    args1 = parser.parse_args(['--config', 'configs/jsrt_simple_classification.json'])

    print(f"Before: algorithm={args1.algorithm}, rounds={args1.global_rounds}, no_wandb={args1.no_wandb}")
    args1 = load_config_to_args(args1.config, args1)
    print(f"After:  algorithm={args1.algorithm}, rounds={args1.global_rounds}, no_wandb={args1.no_wandb}")

    print("\n=== TEST 2: CLI overrides JSON ===")
    # Parse with config file + CLI overrides
    args2 = parser.parse_args([
        '--config', 'configs/jsrt_simple_classification.json',
        '--algorithm', 'FedProx',
        '--global_rounds', '200',
        '--no_wandb'  # This should remain True even if JSON says False
    ])

    print(f"Before: algorithm={args2.algorithm}, rounds={args2.global_rounds}, no_wandb={args2.no_wandb}")
    args2 = load_config_to_args(args2.config, args2)
    print(f"After:  algorithm={args2.algorithm}, rounds={args2.global_rounds}, no_wandb={args2.no_wandb}")

    # Verify CLI took precedence
    success = True
    if args2.algorithm != 'FedProx':
        print("❌ CLI algorithm override failed")
        success = False
    if args2.global_rounds != 200:
        print("❌ CLI global_rounds override failed")
        success = False
    if args2.no_wandb != True:
        print("❌ CLI no_wandb override failed")
        success = False

    if success:
        print("✅ CLI override functionality working correctly!")
    else:
        print("❌ CLI override functionality failed!")

    return success

if __name__ == "__main__":
    test_cli_override()