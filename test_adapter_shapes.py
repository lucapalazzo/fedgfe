#!/usr/bin/env python3
"""
Test script to verify adapter output shapes with different batch sizes
"""
import torch
import torch.nn as nn
import sys
sys.path.append('system')

from flcore.trainmodel.Audio2Visual_NoData.src.models.projection import ASTAdapter
from flcore.trainmodel.Audio2Visual_NoData.src.models.multi_head_attention import MAPBlock

def test_clip_adapter_shapes():
    """Test CLIP adapter with different batch sizes"""
    print("=" * 60)
    print("Testing CLIP Adapter Shapes")
    print("=" * 60)

    # CLIP adapter setup (as in downstreamsinestesiaadapters.py line 89-95)
    clip_adapter = torch.nn.Sequential()
    clip_adapter.add_module("adapter_clip", ASTAdapter(embed_dim=768, n_latents=64, n_heads=8))
    clip_adapter.add_module("projection_clip", MAPBlock(n_latents=1, embed_dim=768, n_heads=8))

    # Test with different batch sizes
    for batch_size in [1, 2, 4, 8]:
        print(f"\n--- Batch Size: {batch_size} ---")

        # Simulate AST output: (batch, 1214, 768)
        ast_output = torch.randn(batch_size, 1214, 768)
        print(f"AST output shape: {ast_output.shape}")

        # Pass through CLIP adapter
        clip_output = clip_adapter(ast_output)
        print(f"CLIP adapter output shape: {clip_output.shape}")

        # Simulate target pooled prompt embeds (should be (batch, 768))
        target_pooled = torch.randn(batch_size, 768)
        print(f"Target pooled embeds shape: {target_pooled.shape}")

        # Test MSE loss
        try:
            mse_loss = nn.MSELoss()
            loss = mse_loss(clip_output, target_pooled)
            print(f"✓ MSE Loss computed successfully: {loss.item():.6f}")
        except Exception as e:
            print(f"✗ MSE Loss failed: {e}")

def test_t5_adapter_shapes():
    """Test T5 adapter with different batch sizes"""
    print("\n" + "=" * 60)
    print("Testing T5 Adapter Shapes")
    print("=" * 60)

    # T5 adapter setup (as in downstreamsinestesiaadapters.py line 98-104)
    t5_adapter = torch.nn.Sequential()
    t5_adapter.add_module("adapter_t5", ASTAdapter(embed_dim=768, n_latents=64, n_heads=8, output_dim=4096))
    t5_adapter.add_module("projection_t5", MAPBlock(n_latents=17, embed_dim=4096, n_heads=8))

    # Test with different batch sizes
    for batch_size in [1, 2, 4, 8]:
        print(f"\n--- Batch Size: {batch_size} ---")

        # Simulate AST output: (batch, 1214, 768)
        ast_output = torch.randn(batch_size, 1214, 768)
        print(f"AST output shape: {ast_output.shape}")

        # Pass through T5 adapter
        t5_output = t5_adapter(ast_output)
        print(f"T5 adapter output shape: {t5_output.shape}")

        # Simulate target prompt embeds (should be (batch, 17, 4096))
        target_embeds = torch.randn(batch_size, 17, 4096)
        print(f"Target prompt embeds shape: {target_embeds.shape}")

        # Test MSE loss
        try:
            mse_loss = nn.MSELoss()
            loss = mse_loss(t5_output, target_embeds)
            print(f"✓ MSE Loss computed successfully: {loss.item():.6f}")
        except Exception as e:
            print(f"✗ MSE Loss failed: {e}")

if __name__ == "__main__":
    test_clip_adapter_shapes()
    test_t5_adapter_shapes()
    print("\n" + "=" * 60)
    print("Shape verification complete!")
    print("=" * 60)
