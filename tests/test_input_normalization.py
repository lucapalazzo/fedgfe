"""
Test: Normalizzazione del batch all'INPUT dell'adapter
"""
import sys
sys.path.insert(0, '/home/lpala/fedgfe/system')

import torch
import torch.nn as nn
from flcore.trainmodel.Audio2Visual_NoData.src.models.projection_improved import Adapter
from flcore.trainmodel.Audio2Visual_NoData.src.models.multi_head_attention import MAPBlock

print("="*80)
print("TEST: Input Normalization Strategies")
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


class AdapterWithInputNorm(nn.Module):
    """Adapter con normalizzazione all'input"""
    def __init__(self, input_dim, hidden_dims, output_dim, norm_type='batch'):
        super().__init__()

        # Input normalization
        if norm_type == 'batch':
            self.input_norm = nn.BatchNorm1d(input_dim)
        elif norm_type == 'layer':
            self.input_norm = nn.LayerNorm(input_dim)
        elif norm_type == 'instance':
            self.input_norm = nn.InstanceNorm1d(input_dim, affine=True)
        elif norm_type == 'group':
            self.input_norm = nn.GroupNorm(16, input_dim)
        elif norm_type == 'rms':
            # RMSNorm custom
            class RMSNorm(nn.Module):
                def __init__(self, dim):
                    super().__init__()
                    self.scale = nn.Parameter(torch.ones(dim))
                    self.eps = 1e-6
                def forward(self, x):
                    norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
                    return x * norm * self.scale
            self.input_norm = RMSNorm(input_dim)
        else:
            self.input_norm = nn.Identity()

        # Main adapter (usa Adapter migliorato)
        self.adapter = Adapter(input_dim, hidden_dims, output_dim, dropout=0.0)

    def forward(self, x):
        # x: (batch, seq_len, features)
        batch, seq_len, feat = x.shape

        if isinstance(self.input_norm, (nn.BatchNorm1d, nn.GroupNorm, nn.InstanceNorm1d)):
            # Transpose per batch/group/instance norm: (batch, features, seq_len)
            x = x.transpose(1, 2)
            x = self.input_norm(x)
            x = x.transpose(1, 2)
        else:
            # LayerNorm, RMSNorm operano sull'ultima dim
            x = self.input_norm(x)

        # Main processing
        x = self.adapter(x)

        return x


# Configurazioni da testare
configs = [
    {"name": "No Normalization (baseline)", "norm": None},
    {"name": "BatchNorm1d INPUT", "norm": "batch"},
    {"name": "LayerNorm INPUT", "norm": "layer"},
    {"name": "InstanceNorm1d INPUT", "norm": "instance"},
    {"name": "GroupNorm INPUT", "norm": "group"},
    {"name": "RMSNorm INPUT", "norm": "rms"},
]

loss_fn = nn.MSELoss()
n_steps = 50

results = []

for config in configs:
    print(f"\n{'='*80}")
    print(f"Testing: {config['name']}")
    print("="*80)

    # Crea modello
    if config['norm'] is None:
        # Baseline senza normalizzazione input
        adapter = Adapter(768, [1024], 768, dropout=0.0)
    else:
        adapter = AdapterWithInputNorm(768, [1024], 768, norm_type=config['norm'])

    projection = MAPBlock(n_latents=1, embed_dim=768, n_heads=8)
    adapter.train()
    projection.train()

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
        optimizer.step()

        losses.append(loss.item())

        if step % 10 == 0 or step == n_steps-1:
            print(f"  Step {step:2d}: loss={loss.item():.4f}")

    # Risultati
    initial_loss = losses[0]
    final_loss = losses[-1]
    min_loss = min(losses)
    reduction = initial_loss - final_loss
    reduction_pct = (reduction / initial_loss) * 100

    results.append({
        'name': config['name'],
        'initial': initial_loss,
        'final': final_loss,
        'min': min_loss,
        'reduction': reduction,
        'reduction_pct': reduction_pct
    })

    print(f"\n  Initial: {initial_loss:.4f}")
    print(f"  Final:   {final_loss:.4f}")
    print(f"  Min:     {min_loss:.4f}")
    print(f"  Reduction: {reduction:.4f} ({reduction_pct:.1f}%)")

# Test aggiuntivo: Doppia normalizzazione (INPUT + durante)
print(f"\n{'='*80}")
print("BONUS TEST: BatchNorm INPUT + GroupNorm INTERNO")
print("="*80)

class DoubleNormAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        # Input normalization
        self.input_norm = nn.BatchNorm1d(768)
        # Adapter con GroupNorm interno (già presente in Adapter)
        self.adapter = Adapter(768, [1024], 768, dropout=0.0)

    def forward(self, x):
        # BatchNorm sull'input
        x = x.transpose(1, 2)
        x = self.input_norm(x)
        x = x.transpose(1, 2)
        # Adapter (già ha GroupNorm interno)
        return self.adapter(x)

adapter_double = DoubleNormAdapter()
projection_double = MAPBlock(n_latents=1, embed_dim=768, n_heads=8)
adapter_double.train()
projection_double.train()

optimizer_double = torch.optim.AdamW(
    list(adapter_double.parameters()) + list(projection_double.parameters()),
    lr=3e-4,
    weight_decay=1e-4
)

losses_double = []
for step in range(n_steps):
    indices = torch.randperm(4)
    audio = torch.randn(4, 1214, 768)
    target = targets[indices]

    optimizer_double.zero_grad()
    out = projection_double(adapter_double(audio))
    loss = loss_fn(out, target)
    loss.backward()
    optimizer_double.step()

    losses_double.append(loss.item())

    if step % 10 == 0 or step == n_steps-1:
        print(f"  Step {step:2d}: loss={loss.item():.4f}")

results.append({
    'name': 'Double Norm (Input+Internal)',
    'initial': losses_double[0],
    'final': losses_double[-1],
    'min': min(losses_double),
    'reduction': losses_double[0] - losses_double[-1],
    'reduction_pct': ((losses_double[0] - losses_double[-1]) / losses_double[0]) * 100
})

# Confronto finale
print(f"\n{'='*80}")
print("CONFRONTO FINALE")
print("="*80)

print(f"\n{'Strategy':<35} {'Initial':>10} {'Final':>10} {'Min':>10} {'Reduction':>10} {'%':>8}")
print("-"*90)

# Ordina per final loss (migliore prima)
results_sorted = sorted(results, key=lambda x: x['final'])

for r in results_sorted:
    print(f"{r['name']:<35} {r['initial']:>10.4f} {r['final']:>10.4f} {r['min']:>10.4f} {r['reduction']:>10.4f} {r['reduction_pct']:>7.1f}%")

print(f"\n{'='*80}")
print(f"🏆 MIGLIORE: {results_sorted[0]['name']}")
print(f"   Final loss: {results_sorted[0]['final']:.4f} (vs baseline ~0.135)")
if results_sorted[0]['final'] < 0.10:
    print(f"   ✅ SUCCESS! Riduzione significativa con batch eterogenei!")
elif results_sorted[0]['final'] < 0.12:
    print(f"   ✓ Miglioramento moderato")
else:
    print(f"   ⚠️  Miglioramento limitato")
print("="*80)
