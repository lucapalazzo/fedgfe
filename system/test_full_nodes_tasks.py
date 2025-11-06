#!/usr/bin/env python
"""
Test full integration of nodes_tasks with config_loader.
"""

import argparse
import sys
sys.path.append('/home/lpala/fedgfe/system')

from utils.config_loader import load_config_to_args

def test_full_integration():
    """Test full integration of nodes_tasks with argparse."""

    # Create minimal parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--num_clients', type=int, default=2)
    parser.add_argument('--nodes_datasets', type=str, default='cifar10')
    parser.add_argument('--nodes_pretext_tasks', type=str, default='')
    parser.add_argument('--nodes_downstream_tasks', type=str, default='none')

    # Parse with config file
    args = parser.parse_args(['--config', 'configs/jsrt_simple_classification.json'])

    print("=== BEFORE CONFIG LOADING ===")
    print(f"num_clients: {args.num_clients}")
    print(f"nodes_datasets: {args.nodes_datasets}")
    print(f"nodes_pretext_tasks: {args.nodes_pretext_tasks}")
    print(f"nodes_downstream_tasks: {args.nodes_downstream_tasks}")

    # Load config
    try:
        args = load_config_to_args(args.config, args)

        print("\n=== AFTER CONFIG LOADING ===")
        print(f"num_clients: {args.num_clients}")
        print(f"nodes_datasets: {args.nodes_datasets}")
        print(f"nodes_pretext_tasks: {args.nodes_pretext_tasks}")
        print(f"nodes_downstream_tasks: {args.nodes_downstream_tasks}")

        print("\n✅ Integration test successful!")
        return True

    except Exception as e:
        print(f"\n❌ Error during config loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_full_integration()