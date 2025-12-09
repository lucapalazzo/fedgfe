"""
Test diverse architetture di adapter per batch eterogenei
"""
import sys
sys.path.insert(0, '/home/lpala/fedgfe/system')

import torch
import torch.nn as nn
from flcore.trainmodel.Audio2Visual_NoData.src.models.projection_improved import Adapter
from flcore.trainmodel.Audio2Visual_NoData.src.models.multi_head_attention import MAPBlock

print("="*80)
print("TEST: Diverse architetture adapter per batch eterogenei")
print("="*80)

# Carica target reali
text_embs = torch.load('/home/lpala/fedgfe/dataset/Audio/esc50_text_embs_dict.pt', map_location='cpu')
classes = ['dog', 'cat', 'frog', 'cow']
targets = []
for cls in classes:
    if cls in text_embs:
        pooled = text_embs[cls]['flux']['pooled_prompt_embeds'].squeeze()
        targets.append(pooled)
targets = torch.stack(targets)  # (4, 768)

# Configurazioni da testare
configs = [
    {"name": "Original [1024]", "hidden_dims": [1024]},
    {"name": "Wider [2048]", "hidden_dims": [2048]},
    {"name": "Very Wide [3072]", "hidden_dims": [3072]},
    {"name": "Two layers [1536,1536]", "hidden_dims": [1536, 1536]},
    {"name": "Bottleneck [2048,512,2048]", "hidden_dims": [2048, 512, 2048]},
    {"name": "Deep [1024,1024,1024]", "hidden_dims": [1024, 1024, 1024]},
]

loss_fn = nn.MSELoss()
n_steps = 50

results = []

for config in configs:
    print(f"\n{'='*80}")
    print(f"Testing: {config['name']}")
    print(f"Architecture: 768 → {config['hidden_dims']} → 768")
    print("="*80)

    # Crea modello
    adapter = Adapter(768, config['hidden_dims'], 768, dropout=0.0)
    projection = MAPBlock(n_latents=1, embed_dim=768, n_heads=8)
    adapter.train()
    projection.train()

    # Count params
    total_params = sum(p.numel() for p in adapter.parameters()) + sum(p.numel() for p in projection.parameters())

    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(projection.parameters()),
        lr=3e-4,
        weight_decay=1e-4
    )

    losses = []
    for step in range(n_steps):
        # Batch eterogeneo con ordine random
        indices = torch.randperm(4)
        audio = torch.randn(4, 1214, 768)
        target = targets[indices]

        optimizer.zero_grad()
        out = projection(adapter(audio))
        loss = loss_fn(out, target)
        loss.backward()

        # Check gradient norm
        grad_norm = 0
        for p in list(adapter.parameters()) + list(projection.parameters()):
            if p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5

        optimizer.step()
        losses.append(loss.item())

        if step % 10 == 0 or step == n_steps-1:
            print(f"  Step {step:2d}: loss={loss.item():.4f}, grad_norm={grad_norm:.4f}")

    # Risultati
    initial_loss = losses[0]
    final_loss = losses[-1]
    reduction = initial_loss - final_loss
    reduction_pct = (reduction / initial_loss) * 100

    results.append({
        'name': config['name'],
        'hidden_dims': config['hidden_dims'],
        'params': total_params,
        'initial': initial_loss,
        'final': final_loss,
        'reduction': reduction,
        'reduction_pct': reduction_pct
    })

    print(f"\n  Params: {total_params:,}")
    print(f"  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Reduction: {reduction:.4f} ({reduction_pct:.1f}%)")

# Confronto finale
print(f"\n{'='*80}")
print("CONFRONTO FINALE")
print("="*80)

print(f"\n{'Architecture':<30} {'Params':>12} {'Initial':>10} {'Final':>10} {'Reduction':>10} {'%':>8}")
print("-"*80)

# Ordina per final loss (migliore prima)
results_sorted = sorted(results, key=lambda x: x['final'])

for r in results_sorted:
    print(f"{r['name']:<30} {r['params']:>12,} {r['initial']:>10.4f} {r['final']:>10.4f} {r['reduction']:>10.4f} {r['reduction_pct']:>7.1f}%")

print(f"\n{'='*80}")
print(f"🏆 MIGLIORE: {results_sorted[0]['name']} (final loss: {results_sorted[0]['final']:.4f})")
print("="*80)
