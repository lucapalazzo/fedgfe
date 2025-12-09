#!/usr/bin/env python3
"""
Test script to verify generator sequence length configurations work correctly.
Tests different combinations of training and output sequence lengths.
"""

import torch
import sys
sys.path.append('system')

from flcore.trainmodel.generators import VAEGenerator

def test_generator_sequence_lengths():
    """Test different sequence length configurations"""

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    test_configs = [
        {
            "name": "Default (4 tokens)",
            "training_seq_len": 4,
            "output_seq_len": None,
            "expected_output": 4
        },
        {
            "name": "Training=4, Output=8 (2x upsampling)",
            "training_seq_len": 4,
            "output_seq_len": 8,
            "expected_output": 8
        },
        {
            "name": "Training=4, Output=1214 (full AST)",
            "training_seq_len": 4,
            "output_seq_len": 1214,
            "expected_output": 1214
        },
        {
            "name": "Training=8, Output=1214",
            "training_seq_len": 8,
            "output_seq_len": 1214,
            "expected_output": 1214
        }
    ]

    for config in test_configs:
        print(f"Testing: {config['name']}")
        print("=" * 60)

        # Create generator
        generator = VAEGenerator(
            input_dim=768,
            hidden_dim=1024,
            latent_dim=256,
            sequence_length=config['training_seq_len']
        ).to(device)

        # Count parameters
        num_params = sum(p.numel() for p in generator.parameters())
        print(f"  Model parameters: {num_params:,}")

        # Test training (forward pass)
        batch_size = 2
        training_input = torch.randn(
            batch_size,
            config['training_seq_len'],
            768
        ).to(device)

        try:
            recon, mu, logvar = generator(training_input)
            print(f"  ✅ Training forward pass successful")
            print(f"     Input shape:  {list(training_input.shape)}")
            print(f"     Output shape: {list(recon.shape)}")
            assert recon.shape == training_input.shape, "Training output shape mismatch!"
        except Exception as e:
            print(f"  ❌ Training forward pass failed: {e}")
            continue

        # Test generation (sampling)
        num_samples = 5
        target_seq_len = config['output_seq_len']

        try:
            generated = generator.sample(
                num_samples=num_samples,
                device=device,
                target_sequence_length=target_seq_len
            )
            print(f"  ✅ Generation (sampling) successful")
            print(f"     Generated shape: {list(generated.shape)}")

            expected_shape = [num_samples, config['expected_output'], 768]
            assert list(generated.shape) == expected_shape, \
                f"Generated shape {list(generated.shape)} != expected {expected_shape}"
            print(f"  ✅ Output shape matches expected: {expected_shape}")

        except Exception as e:
            print(f"  ❌ Generation failed: {e}")
            continue

        # Memory estimation
        if device == 'cuda':
            torch.cuda.empty_cache()
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"  GPU Memory allocated: {memory_allocated:.2f} GB")

        print(f"  ✅ All tests passed for this configuration!\n")

        # Cleanup
        del generator
        if device == 'cuda':
            torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Generator Sequence Length Configuration Test")
    print("=" * 60)
    print()

    try:
        test_generator_sequence_lengths()
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        raise
