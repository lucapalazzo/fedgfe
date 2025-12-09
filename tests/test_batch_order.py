"""
Test per verificare se l'ordine dei batch influenza la convergenza
"""
import sys
sys.path.insert(0, '/home/lpala/fedgfe/system')

import torch
import torch.nn as nn
from flcore.trainmodel.Audio2Visual_NoData.src.models.projection_improved import Adapter
from flcore.trainmodel.Audio2Visual_NoData.src.models.multi_head_attention import MAPBlock

print("="*80)
print("TEST: Effetto dell'ordine dei batch sulla convergenza")
print("="*80)

# Simula text embeddings per diverse classi
torch.manual_seed(42)
text_embs = torch.load('/home/lpala/fedgfe/dataset/Audio/esc50_text_embs_dict.pt', map_location='cpu')

# Prendi 4 classi
classes = ['dog', 'cat', 'frog', 'cow']
targets = []
for cls in classes:
    if cls in text_embs:
        pooled = text_embs[cls]['flux']['pooled_prompt_embeds'].squeeze()  # (768,)
        targets.append(pooled)

targets = torch.stack(targets)  # (4, 768)

# Crea adapter e projection
adapter = Adapter(768, [1024], 768, dropout=0.0)
projection = MAPBlock(n_latents=1, embed_dim=768, n_heads=8)
adapter.train()
projection.train()

optimizer = torch.optim.AdamW(
    list(adapter.parameters()) + list(projection.parameters()),
    lr=3e-4,
    weight_decay=1e-4
)

loss_fn = nn.MSELoss()

print(f"\nTarget distances:")
for i in range(len(classes)):
    for j in range(i+1, len(classes)):
        dist = torch.dist(targets[i], targets[j])
        print(f"  {classes[i]}-{classes[j]}: {dist.item():.3f}")

# Test 1: Batch omogeneo (stesso target)
print(f"\n{'='*80}")
print("TEST 1: Batch omogeneo (4 samples, stesso target 'dog')")
print("="*80)

adapter_homo = Adapter(768, [1024], 768, dropout=0.0)
projection_homo = MAPBlock(n_latents=1, embed_dim=768, n_heads=8)
adapter_homo.train()
projection_homo.train()

optimizer_homo = torch.optim.AdamW(
    list(adapter_homo.parameters()) + list(projection_homo.parameters()),
    lr=3e-4,
    weight_decay=1e-4
)

losses_homo = []
for step in range(20):
    # 4 samples, tutti 'dog'
    audio = torch.randn(4, 1214, 768)
    target = targets[0:1].expand(4, -1)  # tutti 'dog'

    optimizer_homo.zero_grad()
    out = projection_homo(adapter_homo(audio))
    loss = loss_fn(out, target)
    loss.backward()
    optimizer_homo.step()

    losses_homo.append(loss.item())
    if step % 5 == 0 or step == 19:
        print(f"  Step {step:2d}: loss={loss.item():.4f}")

# Test 2: Batch eterogeneo ma ordinato (sempre stesso ordine)
print(f"\n{'='*80}")
print("TEST 2: Batch eterogeneo ORDINATO (4 samples: dog,cat,frog,cow - sempre stesso ordine)")
print("="*80)

adapter_hetero_ord = Adapter(768, [1024], 768, dropout=0.0)
projection_hetero_ord = MAPBlock(n_latents=1, embed_dim=768, n_heads=8)
adapter_hetero_ord.train()
projection_hetero_ord.train()

optimizer_hetero_ord = torch.optim.AdamW(
    list(adapter_hetero_ord.parameters()) + list(projection_hetero_ord.parameters()),
    lr=3e-4,
    weight_decay=1e-4
)

losses_hetero_ord = []
for step in range(20):
    # 4 samples, classi diverse MA sempre nello stesso ordine
    audio = torch.randn(4, 1214, 768)
    target = targets  # [dog, cat, frog, cow] - sempre lo stesso ordine!

    optimizer_hetero_ord.zero_grad()
    out = projection_hetero_ord(adapter_hetero_ord(audio))
    loss = loss_fn(out, target)
    loss.backward()
    optimizer_hetero_ord.step()

    losses_hetero_ord.append(loss.item())
    if step % 5 == 0 or step == 19:
        print(f"  Step {step:2d}: loss={loss.item():.4f}")

# Test 3: Batch eterogeneo RANDOM (ordine cambia ogni volta)
print(f"\n{'='*80}")
print("TEST 3: Batch eterogeneo RANDOM (4 samples: ordine casuale ogni step)")
print("="*80)

adapter_hetero_rand = Adapter(768, [1024], 768, dropout=0.0)
projection_hetero_rand = MAPBlock(n_latents=1, embed_dim=768, n_heads=8)
adapter_hetero_rand.train()
projection_hetero_rand.train()

optimizer_hetero_rand = torch.optim.AdamW(
    list(adapter_hetero_rand.parameters()) + list(projection_hetero_rand.parameters()),
    lr=3e-4,
    weight_decay=1e-4
)

losses_hetero_rand = []
for step in range(20):
    # 4 samples, classi diverse E ordine randomico
    indices = torch.randperm(4)
    audio = torch.randn(4, 1214, 768)
    target = targets[indices]  # Ordine random!

    optimizer_hetero_rand.zero_grad()
    out = projection_hetero_rand(adapter_hetero_rand(audio))
    loss = loss_fn(out, target)
    loss.backward()
    optimizer_hetero_rand.step()

    losses_hetero_rand.append(loss.item())
    if step % 5 == 0 or step == 19:
        print(f"  Step {step:2d}: loss={loss.item():.4f}")

# Confronto
print(f"\n{'='*80}")
print("CONFRONTO")
print("="*80)

print(f"\nLoss iniziale vs finale:")
print(f"  Omogeneo:          {losses_homo[0]:.4f} → {losses_homo[-1]:.4f} (riduzione: {losses_homo[0]-losses_homo[-1]:.4f}, {(1-losses_homo[-1]/losses_homo[0])*100:.1f}%)")
print(f"  Eterogeneo ord:    {losses_hetero_ord[0]:.4f} → {losses_hetero_ord[-1]:.4f} (riduzione: {losses_hetero_ord[0]-losses_hetero_ord[-1]:.4f}, {(1-losses_hetero_ord[-1]/losses_hetero_ord[0])*100:.1f}%)")
print(f"  Eterogeneo random: {losses_hetero_rand[0]:.4f} → {losses_hetero_rand[-1]:.4f} (riduzione: {losses_hetero_rand[0]-losses_hetero_rand[-1]:.4f}, {(1-losses_hetero_rand[-1]/losses_hetero_rand[0])*100:.1f}%)")

print(f"\n{'='*80}")
print("✓ Test completato")
print("="*80)
