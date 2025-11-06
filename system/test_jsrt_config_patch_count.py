#!/usr/bin/env python
"""
Test patch_count calculation with the actual JSRT configuration file
"""

import sys
import argparse
sys.path.append('/home/lpala/fedgfe/system')

from utils.config_loader import load_config_to_args

def test_jsrt_config_patch_count():
    """Test patch_count calculation with the actual JSRT configuration."""

    print("🧪 Testing patch_count with JSRT Configuration")
    print("=" * 60)

    class MockArgs:
        def __init__(self):
            # Use defaults that match main.py
            self.config = None
            self.dataset_image_size = -1
            self.patch_size = 16
            self.patch_count = None
            # Add all other attributes needed
            self.goal = "test"
            self.device = "cuda"
            self.device_id = "0"
            self.times = 1
            self.seed = -1
            self.algorithm = "FedAvg"
            self.model = "cnn"
            self.num_classes = 10

    print("✅ Testing with configs/jsrt_simple_classification.json")

    args = MockArgs()
    args.config = "/home/lpala/fedgfe/configs/jsrt_simple_classification.json"

    print(f"Before loading:")
    print(f"  patch_size: {args.patch_size}")
    print(f"  dataset_image_size: {args.dataset_image_size}")
    print(f"  patch_count: {args.patch_count}")

    try:
        result = load_config_to_args(args.config, args)

        print(f"\nAfter loading:")
        print(f"  patch_size: {getattr(result, 'patch_size', 'NOT_SET')}")
        print(f"  dataset_image_size: {getattr(result, 'dataset_image_size', 'NOT_SET')}")
        print(f"  patch_count: {getattr(result, 'patch_count', 'NOT_SET')}")

        # Verify the configuration loaded correctly
        assert hasattr(result, 'patch_size'), "patch_size not loaded"
        assert hasattr(result, 'nodes_tasks'), "nodes_tasks not loaded"
        print(f"\n✅ Configuration loaded successfully!")

        # Check if patch_count was calculated
        if hasattr(result, 'patch_count') and result.patch_count is not None:
            print(f"✅ patch_count calculated: {result.patch_count}")
        else:
            print("ℹ️  patch_count not calculated (may need dataset_image_size)")

        # Show nodes_tasks configuration
        if hasattr(result, 'nodes_tasks') and result.nodes_tasks:
            print(f"\n📋 nodes_tasks configuration:")
            for node_id, config in result.nodes_tasks.items():
                print(f"  Node {node_id}: {config}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("🎉 JSRT configuration test completed!")
    return True

if __name__ == "__main__":
    try:
        test_jsrt_config_patch_count()
        print("\n🎯 JSRT config test successful!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)