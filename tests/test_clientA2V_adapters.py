#!/usr/bin/env python
"""
Test script to verify the local adapter mechanism in clientA2V.

This script verifies:
1. Local adapters are created correctly
2. Optimizer tracks only local adapter parameters
3. Training updates only local adapters
4. sync_local_to_global() propagates changes correctly
"""

import sys
sys.path.append('/home/lpala/fedgfe/system')

import torch
import torch.nn as nn
from collections import OrderedDict


class MockArgs:
    def __init__(self):
        self.local_learning_rate = 0.001
        self.no_wandb = True
        self.dataset = "mock_dataset"
        self.model_optimizer = "adam"
        self.device = "cpu"
        self.global_rounds = 10

        # A2V specific
        self.diffusion_type = 'sd'
        self.use_act_loss = True
        self.audio_model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
        self.img_pipe_name = "runwayml/stable-diffusion-v1-5"
        self.img_lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"


class MockAdapter(nn.Module):
    """Mock adapter for testing"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)


class MockAudio2Image:
    """Mock Audio2Image model for testing"""
    def __init__(self):
        self.clip_adapter = MockAdapter(768, 768)
        self.clip_projection = MockAdapter(768, 768)
        # Initialize with specific values for testing
        nn.init.constant_(self.clip_adapter.fc.weight, 1.0)
        nn.init.constant_(self.clip_adapter.fc.bias, 0.0)
        nn.init.constant_(self.clip_projection.fc.weight, 2.0)
        nn.init.constant_(self.clip_projection.fc.bias, 0.0)


class MockGlobalModel:
    """Mock global model for testing"""
    def __init__(self):
        self.audio2image = MockAudio2Image()

    def get_audio2image_model(self):
        return self.audio2image


class MockNodeConfig:
    """Mock node configuration"""
    def __init__(self):
        self.dataset = None


def test_local_adapter_creation():
    """Test 1: Verify local adapters are created as deep copies"""
    print("\n" + "="*60)
    print("TEST 1: Local Adapter Creation")
    print("="*60)

    from flcore.clients.clientA2V import clientA2V

    args = MockArgs()
    global_model = MockGlobalModel()
    node_config = MockNodeConfig()

    # Create client
    client = clientA2V(args, node_id=0, node_config=node_config, global_model=global_model)

    # Verify local adapters exist
    assert hasattr(client, 'local_clip_adapter'), "❌ local_clip_adapter not created"
    assert hasattr(client, 'local_clip_projection'), "❌ local_clip_projection not created"
    print("✓ Local adapters created")

    # Verify they are different objects
    assert id(client.local_clip_adapter) != id(global_model.audio2image.clip_adapter), \
        "❌ local_clip_adapter is the same object as global"
    print("✓ Local adapters are separate objects from global")

    # Verify they have the same initial values
    local_weight = client.local_clip_adapter.fc.weight.data
    global_weight = global_model.audio2image.clip_adapter.fc.weight.data
    assert torch.allclose(local_weight, global_weight), "❌ Initial values don't match"
    print("✓ Local adapters initialized with global values")

    print("\n✅ TEST 1 PASSED")
    return client, global_model


def test_optimizer_tracking(client):
    """Test 2: Verify optimizer tracks only local adapter parameters"""
    print("\n" + "="*60)
    print("TEST 2: Optimizer Tracking")
    print("="*60)

    # Setup optimizer
    client.setup_optimizer()

    # Verify optimizer exists
    assert client.train_optimizer is not None, "❌ Optimizer not created"
    print("✓ Optimizer created")

    # Get optimizer parameter IDs
    optimizer_param_ids = set()
    for param_group in client.train_optimizer.param_groups:
        for param in param_group['params']:
            optimizer_param_ids.add(id(param))

    # Get local adapter parameter IDs
    local_param_ids = set()
    for param in client.local_clip_adapter.parameters():
        local_param_ids.add(id(param))
    for param in client.local_clip_projection.parameters():
        local_param_ids.add(id(param))

    # Verify all local parameters are in optimizer
    assert local_param_ids.issubset(optimizer_param_ids), "❌ Not all local parameters in optimizer"
    print(f"✓ All {len(local_param_ids)} local parameters tracked by optimizer")

    # Verify optimizer tracking
    is_valid = client._verify_optimizer_tracking()
    assert is_valid, "❌ Optimizer verification failed"

    print("\n✅ TEST 2 PASSED")


def test_training_updates_local_only(client, global_model):
    """Test 3: Verify training updates only local adapters"""
    print("\n" + "="*60)
    print("TEST 3: Training Updates Local Adapters Only")
    print("="*60)

    # Store initial global values
    initial_global_clip_weight = global_model.audio2image.clip_adapter.fc.weight.data.clone()
    initial_local_clip_weight = client.local_clip_adapter.fc.weight.data.clone()

    # Simulate training update (manual gradient step)
    client.train_optimizer.zero_grad()

    # Create fake loss using local adapter
    fake_input = torch.randn(2, 768)
    output = client.local_clip_adapter(fake_input)
    loss = output.sum()
    loss.backward()

    # Step optimizer
    client.train_optimizer.step()

    # Verify local adapter changed
    local_changed = not torch.allclose(client.local_clip_adapter.fc.weight.data, initial_local_clip_weight)
    assert local_changed, "❌ Local adapter not updated by optimizer"
    print("✓ Local adapter updated by training")

    # Verify global adapter unchanged (before sync)
    global_unchanged = torch.allclose(global_model.audio2image.clip_adapter.fc.weight.data,
                                      initial_global_clip_weight)
    assert global_unchanged, "❌ Global adapter changed without sync"
    print("✓ Global adapter unchanged before sync")

    print("\n✅ TEST 3 PASSED")


def test_sync_local_to_global(client, global_model):
    """Test 4: Verify sync_local_to_global() propagates changes"""
    print("\n" + "="*60)
    print("TEST 4: Sync Local to Global")
    print("="*60)

    # Get current local state
    local_clip_weight = client.local_clip_adapter.fc.weight.data.clone()

    # Sync to global
    client.sync_local_to_global()

    # Verify global now matches local
    global_clip_weight = global_model.audio2image.clip_adapter.fc.weight.data
    assert torch.allclose(global_clip_weight, local_clip_weight), \
        "❌ Global adapter not synced with local"
    print("✓ Global adapter synced with local adapter")

    # Verify optimizer still tracks local
    is_valid = client._verify_optimizer_tracking()
    assert is_valid, "❌ Optimizer tracking broken after sync"
    print("✓ Optimizer still tracks local parameters after sync")

    print("\n✅ TEST 4 PASSED")


def test_set_parameters(client, global_model):
    """Test 5: Verify set_parameters updates local adapters"""
    print("\n" + "="*60)
    print("TEST 5: Set Parameters")
    print("="*60)

    # Modify global model (simulating aggregated update from server)
    new_weight = torch.ones_like(global_model.audio2image.clip_adapter.fc.weight) * 5.0
    global_model.audio2image.clip_adapter.fc.weight.data.copy_(new_weight)

    # Set parameters
    client.set_parameters(global_model)

    # Verify local adapter updated
    assert torch.allclose(client.local_clip_adapter.fc.weight.data, new_weight), \
        "❌ Local adapter not updated from global"
    print("✓ Local adapter updated from global model")

    # Verify optimizer still tracks local
    is_valid = client._verify_optimizer_tracking()
    assert is_valid, "❌ Optimizer tracking broken after set_parameters"
    print("✓ Optimizer still tracks local parameters after update")

    print("\n✅ TEST 5 PASSED")


def main():
    print("\n" + "="*60)
    print("CLIENT A2V LOCAL ADAPTER MECHANISM TEST")
    print("="*60)

    try:
        # Test 1: Create client and verify local adapters
        client, global_model = test_local_adapter_creation()

        # Test 2: Verify optimizer tracks local adapters
        test_optimizer_tracking(client)

        # Test 3: Verify training updates local only
        test_training_updates_local_only(client, global_model)

        # Test 4: Verify sync propagates changes
        test_sync_local_to_global(client, global_model)

        # Test 5: Verify set_parameters updates local
        test_set_parameters(client, global_model)

        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*60)
        print("\nSummary:")
        print("✅ Local adapters created correctly")
        print("✅ Optimizer tracks only local parameters")
        print("✅ Training updates only local adapters")
        print("✅ sync_local_to_global() works correctly")
        print("✅ set_parameters() updates local adapters")
        print("✅ Optimizer remains valid throughout")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
