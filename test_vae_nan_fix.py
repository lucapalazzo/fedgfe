#!/usr/bin/env python3
"""
Test script to verify VAE Generator stability fixes for NaN issues.
"""

import torch
import sys
sys.path.append('/home/lpala/fedgfe')

from system.flcore.trainmodel.generators import VAEGenerator, VAELoss

def test_vae_stability():
    """Test VAE stability with extreme inputs."""
    print("Testing VAE Generator stability fixes...\n")

    # Create VAE
    input_dim = 768
    hidden_dim = 512
    latent_dim = 256
    sequence_length = 4
    batch_size = 8

    vae = VAEGenerator(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        sequence_length=sequence_length
    ).cuda()

    loss_fn = VAELoss(total_epochs=100, beta_warmup_ratio=0.5)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    print(f"VAE Parameters:")
    print(f"  Input dim: {input_dim}")
    print(f"  Hidden dim: {hidden_dim}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Batch size: {batch_size}\n")

    # Test 1: Normal input
    print("Test 1: Normal input (random)")
    x_normal = torch.randn(batch_size, sequence_length, input_dim).cuda()
    test_forward_pass(vae, loss_fn, optimizer, x_normal, epoch=0, test_name="Normal")

    # Test 2: Large magnitude input
    print("\nTest 2: Large magnitude input (x100)")
    x_large = torch.randn(batch_size, sequence_length, input_dim).cuda() * 100
    test_forward_pass(vae, loss_fn, optimizer, x_large, epoch=0, test_name="Large")

    # Test 3: Normalized input (should work best)
    print("\nTest 3: Normalized input")
    x_normalized = torch.nn.functional.normalize(
        torch.randn(batch_size, sequence_length, input_dim).cuda(),
        p=2, dim=-1
    )
    test_forward_pass(vae, loss_fn, optimizer, x_normalized, epoch=0, test_name="Normalized")

    # Test 4: Multiple batches simulation
    print("\nTest 4: Multiple batches (10 iterations)")
    success_count = 0
    for i in range(10):
        x = torch.nn.functional.normalize(
            torch.randn(batch_size, sequence_length, input_dim).cuda(),
            p=2, dim=-1
        )
        success = test_forward_pass(vae, loss_fn, optimizer, x, epoch=i, test_name=f"Batch {i+1}", verbose=False)
        if success:
            success_count += 1

    print(f"\nBatch test results: {success_count}/10 batches succeeded")

    print("\n" + "="*60)
    if success_count == 10:
        print("✓ All tests passed! VAE is stable.")
    else:
        print("✗ Some tests failed. VAE may still have stability issues.")
    print("="*60)

def test_forward_pass(vae, loss_fn, optimizer, x, epoch, test_name, verbose=True):
    """Test a single forward and backward pass."""
    try:
        # Forward pass
        recon_x, mu, logvar = vae(x)

        # Check outputs
        if torch.isnan(recon_x).any():
            if verbose:
                print(f"  ✗ {test_name}: NaN in reconstruction")
            return False
        if torch.isnan(mu).any():
            if verbose:
                print(f"  ✗ {test_name}: NaN in mu")
            return False
        if torch.isnan(logvar).any():
            if verbose:
                print(f"  ✗ {test_name}: NaN in logvar")
            return False

        # Compute loss
        total_loss, recon_loss, kl_loss, sim_loss = loss_fn(recon_x, x, mu, logvar, epoch)

        if torch.isnan(total_loss) or torch.isinf(total_loss):
            if verbose:
                print(f"  ✗ {test_name}: NaN/Inf in loss")
                print(f"     recon_loss: {recon_loss.item():.4f}")
                print(f"     kl_loss: {kl_loss.item():.4f}")
                print(f"     sim_loss: {sim_loss.item():.4f}")
            return False

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Check gradients
        has_nan_grad = False
        for name, param in vae.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                if verbose:
                    print(f"  ✗ {test_name}: NaN gradient in {name}")
                has_nan_grad = True
                break

        if has_nan_grad:
            return False

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(vae.parameters(), max_norm=5.0)
        optimizer.step()

        if verbose:
            print(f"  ✓ {test_name}: Success")
            print(f"     loss: {total_loss.item():.4f}")
            print(f"     mu range: [{mu.min().item():.2f}, {mu.max().item():.2f}]")
            print(f"     logvar range: [{logvar.min().item():.2f}, {logvar.max().item():.2f}]")

        return True

    except Exception as e:
        if verbose:
            print(f"  ✗ {test_name}: Exception - {str(e)}")
        return False

if __name__ == "__main__":
    test_vae_stability()
