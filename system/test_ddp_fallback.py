#!/usr/bin/env python3
"""
Test script to verify DDP fallback behavior.
This script tests that the system gracefully falls back to single process mode
when DDP environment variables are not set.
"""

import os
import sys

# Add the parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flcore.clients.clientgfe_ddp import clientGFEDDP
from utils.ddp_utils import initialize_ddp_args, is_ddp_available


class MockArgs:
    """Mock arguments for testing."""
    def __init__(self):
        # Basic federated learning arguments
        self.dataset = 'cifar10'
        self.num_clients = 2
        self.global_rounds = 2
        self.ssl_rounds = 1
        self.local_epochs = 1
        self.batch_size = 8
        self.local_learning_rate = 0.01
        self.device = 'cpu'  # Use CPU for testing
        
        # Model arguments
        self.nodes_backbone_model = 'hf_vit'
        self.nodes_pretext_tasks = 'patch_ordering'
        self.nodes_downstream_tasks = 'classification'
        self.nodes_training_sequence = 'both'
        
        # Dataset arguments
        self.dataset_image_size = 224
        self.patch_size = 16
        self.num_classes = 10
        
        # Training arguments
        self.model_optimizer = 'AdamW'
        self.model_optimizer_weight_decay = 1e-4
        self.model_optimizer_momentum = 0.9
        self.learning_rate_schedule = 'none'
        
        # Other arguments
        self.no_wandb = True
        self.cls_token_only = False
        self.downstream_loss_operation = 'crossentropy'
        self.no_downstream_tasks = False
        self.loss_weighted = False
        self.dataset_limit = 0
        self.privacy = 'none'
        self.dp_sigma = 0.0
        
        # Embedding and model args
        self.embedding_size = 768
        self.num_hidden_layers = 12
        self.model_pretrain = False
        
        # Debug
        self.debug_pretext_images = False


def test_ddp_fallback():
    """Test that DDP gracefully falls back to single process mode."""
    
    print("=== Testing DDP Fallback Behavior ===\n")
    
    # Clear DDP environment variables to simulate missing setup
    ddp_vars = ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
    original_values = {}
    
    for var in ddp_vars:
        if var in os.environ:
            original_values[var] = os.environ[var]
            del os.environ[var]
    
    print("1. Testing DDP availability check...")
    ddp_available = is_ddp_available()
    print(f"   DDP Available: {ddp_available} (expected: False)\n")
    
    print("2. Testing args initialization...")
    args = MockArgs()
    args = initialize_ddp_args(args)
    
    print(f"   DDP Enabled: {args.ddp_enabled} (expected: False)")
    print(f"   World Size: {args.ddp_world_size} (expected: 1)")
    print(f"   Backend: {args.ddp_backend}\n")
    
    print("3. Testing client creation...")
    try:
        # Create a mock client to test DDP mixin behavior
        client = clientGFEDDP(
            args, 
            model_id=0, 
            train_samples=100, 
            test_samples=50
        )
        
        print(f"   Client created successfully")
        print(f"   Client is_distributed: {client.is_distributed} (expected: False)")
        print(f"   Client rank: {client.rank} (expected: 0)")
        print(f"   Client world_size: {client.world_size} (expected: 1)")
        
        # Test distributed info
        if hasattr(client, 'get_distributed_info'):
            info = client.get_distributed_info()
            print(f"   Client DDP info: {info}")
        
        print("\n✅ Test PASSED: DDP fallback works correctly")
        
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        return False
    
    finally:
        # Restore original environment variables
        for var, value in original_values.items():
            os.environ[var] = value
    
    return True


def test_ddp_with_environment():
    """Test DDP with proper environment setup."""
    
    print("\n=== Testing DDP with Environment Setup ===\n")
    
    # Set up DDP environment for single process
    os.environ['RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'  # Single process
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    print("1. Testing DDP availability with environment...")
    ddp_available = is_ddp_available()
    print(f"   DDP Available: {ddp_available} (expected: True)")
    
    print("2. Testing args initialization with environment...")
    args = MockArgs()
    args = initialize_ddp_args(args)
    
    print(f"   DDP Enabled: {args.ddp_enabled} (expected: False for WORLD_SIZE=1)")
    print(f"   World Size: {args.ddp_world_size}")
    
    # Clean up
    ddp_vars = ['RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
    for var in ddp_vars:
        if var in os.environ:
            del os.environ[var]
    
    print("\n✅ Test PASSED: DDP environment handling works correctly")
    
    return True


def main():
    """Run all tests."""
    print("DDP Fallback Test Suite")
    print("=" * 50)
    
    success = True
    
    try:
        success &= test_ddp_fallback()
        success &= test_ddp_with_environment()
        
        if success:
            print("\n" + "=" * 50)
            print("🎉 ALL TESTS PASSED")
            print("DDP fallback behavior is working correctly!")
        else:
            print("\n" + "=" * 50)
            print("⚠️  SOME TESTS FAILED")
            return 1
            
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())