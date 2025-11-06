#!/usr/bin/env python
"""
Quick test script to verify JSON configuration loading works correctly.
"""

import argparse
import sys
import os

# Add system path to import config_loader
sys.path.append('/home/lpala/fedgfe/system')

from utils.config_loader import load_config_to_args

def test_config_loading():
    """Test that JSON config values are properly loaded into args."""

    # Create minimal parser (like main.py)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config", type=str, default=None)
    parser.add_argument("--algorithm", type=str, default="FedAvg")
    parser.add_argument("--num_clients", type=int, default=2)
    parser.add_argument("--global_rounds", type=int, default=100)
    parser.add_argument("--local_learning_rate", type=float, default=0.005)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="mnist")

    # Parse with config file
    args = parser.parse_args(['--config', 'configs/example_jsrt_8c.json'])

    print("=== BEFORE CONFIG LOADING ===")
    print(f"Algorithm: {args.algorithm}")
    print(f"Num Clients: {args.num_clients}")
    print(f"Global Rounds: {args.global_rounds}")
    print(f"Learning Rate: {args.local_learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Dataset: {args.dataset}")

    # Load config
    args = load_config_to_args(args.config, args)

    print("\n=== AFTER CONFIG LOADING ===")
    print(f"Algorithm: {args.algorithm}")
    print(f"Num Clients: {args.num_clients}")
    print(f"Global Rounds: {args.global_rounds}")
    print(f"Learning Rate: {args.local_learning_rate}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Dataset: {args.dataset}")

    # Test CLI override
    print("\n=== TESTING CLI OVERRIDE ===")
    args2 = parser.parse_args(['--config', 'configs/example_jsrt_8c.json', '--algorithm', 'FedProx'])
    print(f"Before: Algorithm = {args2.algorithm}")
    args2 = load_config_to_args(args2.config, args2)
    print(f"After: Algorithm = {args2.algorithm} (should be FedProx, not FedGFE)")

if __name__ == "__main__":
    test_config_loading()