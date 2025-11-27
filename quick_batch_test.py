"""Quick test for batch size issues"""
import sys
sys.path.insert(0, '/home/lpala/fedgfe/system')
sys.path.insert(0, '/home/lpala/Audio2Visual_NoData/src/models')

import torch
import torch.nn as nn

print("Testing MAPBlock.squeeze() issue...")

# Simula MAPBlock output
class TestMAPBlock:
    def forward_n1(self, batch_size, n_latents):
        """Simula output con n_latents=1 (CLIP)"""
        latents = torch.randn(batch_size, n_latents, 768)
        return latents.squeeze(dim=1)

    def forward_n17(self, batch_size, n_latents):
        """Simula output con n_latents=17 (T5)"""
        latents = torch.randn(batch_size, n_latents, 4096)
        return latents.squeeze(dim=1)

test = TestMAPBlock()

print("\nCLIP (n_latents=1):")
for bs in [1, 2, 8, 16]:
    out = test.forward_n1(bs, 1)
    print(f"  batch_size={bs:2d}: input=(bs,1,768) → output={out.shape}")
    if bs == 1:
        if len(out.shape) != 2:
            print(f"    ⚠️  WARNING: Expected (1,768), got {out.shape}")

print("\nT5 (n_latents=17):")
for bs in [1, 2, 8, 16]:
    out = test.forward_n17(bs, 17)
    print(f"  batch_size={bs:2d}: input=(bs,17,4096) → output={out.shape}")
    if bs == 1:
        if len(out.shape) != 3:
            print(f"    ⚠️  WARNING: Expected (1,17,4096), got {out.shape}")

# Test MSE loss con batch eterogeneo
print("\n" + "="*60)
print("Testing MSELoss with heterogeneous batches...")
print("="*60)

mse = nn.MSELoss()

# Caso 1: Tutti samples dalla stessa classe
print("\nCase 1: Homogeneous batch (same class)")
output_same = torch.randn(8, 768)
target_same = torch.randn(1, 768).expand(8, -1) + torch.randn(8, 768) * 0.1
loss_same = mse(output_same, target_same)
print(f"  Loss: {loss_same.item():.4f}")

# Caso 2: Samples da classi diverse
print("\nCase 2: Heterogeneous batch (different classes)")
output_diff = torch.randn(8, 768)
targets_diff = []
for i in range(8):
    # Ogni sample ha target molto diverso
    targets_diff.append(torch.randn(1, 768) * (i+1))
target_diff = torch.cat(targets_diff, dim=0)
loss_diff = mse(output_diff, target_diff)
print(f"  Loss: {loss_diff.item():.4f}")

print(f"\nLoss difference: {abs(loss_same.item() - loss_diff.item()):.4f}")

# Test GroupNorm vs LayerNorm
print("\n" + "="*60)
print("Testing GroupNorm vs LayerNorm stability...")
print("="*60)

from projection_improved import Adapter as ImprovedAdapter

# Test con batch size diversi
print("\nImprovedAdapter (with GroupNorm):")
adapter_improved = ImprovedAdapter(768, [1024, 2048], 768, dropout=0.0)
adapter_improved.eval()

for bs in [1, 2, 8, 16]:
    x = torch.randn(bs, 1214, 768)
    with torch.no_grad():
        out = adapter_improved(x)
    print(f"  batch_size={bs:2d}: mean={out.mean().item():+.4f}, std={out.std().item():.4f}")

print("\n✓ Test completed!")
