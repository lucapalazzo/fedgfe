"""
Script di debug per identificare perché l'adapter CLIP non converge
"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/lpala/fedgfe/system')

from flcore.trainmodel.Audio2Visual_NoData.src.models.projection import Adapter
from flcore.trainmodel.Audio2Visual_NoData.src.models.multi_head_attention import MAPBlock

print("=" * 80)
print("DEBUG: Analisi Adapter CLIP")
print("=" * 80)

# Configurazione
batch_size = 8
seq_len = 1214
embed_dim = 768

# Simula audio embeddings da AST
audio_emb = torch.randn(batch_size, seq_len, embed_dim, requires_grad=False)
print(f"\n1. Audio embeddings shape: {audio_emb.shape}")
print(f"   Audio embeddings stats: mean={audio_emb.mean():.4f}, std={audio_emb.std():.4f}")

# Target CLIP pooled embeddings (da text encoder)
target_pooled = torch.randn(batch_size, embed_dim)
print(f"\n2. Target pooled embeddings shape: {target_pooled.shape}")
print(f"   Target stats: mean={target_pooled.mean():.4f}, std={target_pooled.std():.4f}")

# Crea adapter e projection
print("\n3. Creazione Adapter e Projection...")
adapter_clip = Adapter(input_dim=768, hidden_dims=[1024], output_dim=768, dropout=0.1)
projection_clip = MAPBlock(n_latents=1, embed_dim=768, n_heads=8)

# Count parameters
adapter_params = sum(p.numel() for p in adapter_clip.parameters() if p.requires_grad)
projection_params = sum(p.numel() for p in projection_clip.parameters() if p.requires_grad)
print(f"   Adapter parameters: {adapter_params:,}")
print(f"   Projection parameters: {projection_params:,}")
print(f"   Total parameters: {adapter_params + projection_params:,}")

# Forward pass
print("\n4. Forward pass...")
adapter_clip.train()
projection_clip.train()

out_adapter = adapter_clip(audio_emb)
print(f"   After adapter: shape={out_adapter.shape}, mean={out_adapter.mean():.4f}, std={out_adapter.std():.4f}")

out_projection = projection_clip(out_adapter)
print(f"   After projection: shape={out_projection.shape}, mean={out_projection.mean():.4f}, std={out_projection.std():.4f}")

# Calcola loss
print("\n5. Calcolo loss...")
loss_fn = nn.MSELoss()
loss = loss_fn(out_projection, target_pooled)
print(f"   MSE Loss: {loss.item():.6f}")

# Test gradient flow
print("\n6. Test gradient flow...")
optimizer = torch.optim.AdamW(
    list(adapter_clip.parameters()) + list(projection_clip.parameters()),
    lr=1e-4,
    weight_decay=0.01
)

optimizer.zero_grad()
loss.backward()

# Verifica gradienti
print("\n   Gradienti Adapter:")
for name, param in adapter_clip.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_mean = param.grad.mean().item()
        grad_max = param.grad.abs().max().item()
        print(f"     {name}: norm={grad_norm:.6f}, mean={grad_mean:.8f}, max={grad_max:.6f}")
    else:
        print(f"     {name}: NO GRADIENT!")

print("\n   Gradienti Projection:")
for name, param in projection_clip.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_mean = param.grad.mean().item()
        grad_max = param.grad.abs().max().item()
        print(f"     {name}: norm={grad_norm:.6f}, mean={grad_mean:.8f}, max={grad_max:.6f}")
    else:
        print(f"     {name}: NO GRADIENT!")

optimizer.step()

# Test multiple steps
print("\n7. Test convergenza (10 steps)...")
losses = []
for step in range(10):
    optimizer.zero_grad()

    out_adapter = adapter_clip(audio_emb)
    out_projection = projection_clip(out_adapter)
    loss = loss_fn(out_projection, target_pooled)

    loss.backward()

    # Check gradient norm
    total_grad_norm = 0
    for p in list(adapter_clip.parameters()) + list(projection_clip.parameters()):
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5

    optimizer.step()

    losses.append(loss.item())
    print(f"   Step {step+1}: loss={loss.item():.6f}, grad_norm={total_grad_norm:.6f}")

print("\n8. Analisi convergenza:")
if losses[-1] < losses[0]:
    print(f"   ✓ Loss diminuita: {losses[0]:.6f} -> {losses[-1]:.6f} (delta={losses[0]-losses[-1]:.6f})")
else:
    print(f"   ✗ Loss NON diminuita: {losses[0]:.6f} -> {losses[-1]:.6f}")

# Test con audio embeddings detached (come nel codice reale)
print("\n9. Test con audio embeddings DETACHED (come nel codice reale)...")
audio_emb_detached = audio_emb.detach()
losses_detached = []

optimizer_detached = torch.optim.AdamW(
    list(adapter_clip.parameters()) + list(projection_clip.parameters()),
    lr=1e-4,
    weight_decay=0.01
)

for step in range(10):
    optimizer_detached.zero_grad()

    out_adapter = adapter_clip(audio_emb_detached)
    out_projection = projection_clip(out_adapter)
    loss = loss_fn(out_projection, target_pooled)

    loss.backward()
    optimizer_detached.step()

    losses_detached.append(loss.item())
    if step == 0 or step == 9:
        print(f"   Step {step+1}: loss={loss.item():.6f}")

print(f"\n10. Confronto:")
print(f"    Con gradients: {losses[0]:.6f} -> {losses[-1]:.6f}")
print(f"    Con detach:    {losses_detached[0]:.6f} -> {losses_detached[-1]:.6f}")

print("\n" + "=" * 80)
print("Fine debug")
print("=" * 80)
