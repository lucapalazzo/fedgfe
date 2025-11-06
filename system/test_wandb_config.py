#!/usr/bin/env python
"""
Test script to verify wandb.disabled JSON mapping works correctly.
"""

import argparse
import sys
sys.path.append('/home/lpala/fedgfe/system')

from utils.config_loader import load_config_to_args

def test_wandb_config():
    """Test that wandb.disabled is properly mapped to args.no_wandb."""

    # Create minimal parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--no_wandb', type=bool, default=False)

    # Parse with config file
    args = parser.parse_args(['--config', 'configs/jsrt_simple_classification.json'])

    print("=== BEFORE CONFIG LOADING ===")
    print(f"no_wandb: {args.no_wandb}")

    # Load config
    try:
        args = load_config_to_args(args.config, args)

        print("\n=== AFTER CONFIG LOADING ===")
        print(f"no_wandb: {args.no_wandb}")

        # Verify it was set correctly
        if args.no_wandb == True:
            print("✅ wandb.disabled correctly mapped to args.no_wandb=True")
        else:
            print("❌ wandb.disabled mapping failed")

        return True

    except Exception as e:
        print(f"\n❌ Error during config loading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_wandb_config()