"""
Analisi per capire perché il CLIP adapter non converge sotto 0.18
"""
import re

# Parse log file
log_file = "./wandb/run-20251125_124500-t3u4p2qg/files/output.log"

try:
    with open(log_file, 'r') as f:
        content = f.read()
except:
    print("Log file not found, using most recent")
    import subprocess
    result = subprocess.run("ls -lt ./wandb/run-*/files/output.log | head -1 | awk '{print $NF}'",
                          shell=True, capture_output=True, text=True)
    log_file = result.stdout.strip()
    with open(log_file, 'r') as f:
        content = f.read()

# Extract epoch losses
pattern = r'Node 0 Epoch (\d+)/10:.*?clip=([\d.]+), t5=([\d.]+)'
matches = re.findall(pattern, content)

print("=" * 80)
print("ANALISI CONVERGENZA CLIP ADAPTER")
print("=" * 80)

print("\nLoss per Epoch:")
print(f"{'Epoch':<10} {'CLIP Loss':<15} {'T5 Loss':<15} {'Delta CLIP':<15} {'Delta T5':<15}")
print("-" * 80)

prev_clip = None
prev_t5 = None

for epoch, clip_loss, t5_loss in matches:
    clip_loss = float(clip_loss)
    t5_loss = float(t5_loss)

    delta_clip = f"{clip_loss - prev_clip:+.4f}" if prev_clip is not None else "N/A"
    delta_t5 = f"{t5_loss - prev_t5:+.4f}" if prev_t5 is not None else "N/A"

    print(f"{epoch:<10} {clip_loss:<15.4f} {t5_loss:<15.4f} {delta_clip:<15} {delta_t5:<15}")

    prev_clip = clip_loss
    prev_t5 = t5_loss

# Analisi
print("\n" + "=" * 80)
print("ANALISI")
print("=" * 80)

if matches:
    first_clip = float(matches[0][1])
    last_clip = float(matches[-1][1])
    first_t5 = float(matches[0][2])
    last_t5 = float(matches[-1][2])

    clip_reduction = first_clip - last_clip
    t5_reduction = first_t5 - last_t5

    clip_reduction_pct = (clip_reduction / first_clip) * 100
    t5_reduction_pct = (t5_reduction / first_t5) * 100

    print(f"\nRiduzione totale loss:")
    print(f"  CLIP: {first_clip:.4f} → {last_clip:.4f} (delta: {clip_reduction:.4f}, {clip_reduction_pct:.1f}%)")
    print(f"  T5:   {first_t5:.4f} → {last_t5:.4f} (delta: {t5_reduction:.4f}, {t5_reduction_pct:.1f}%)")

    # Find minimum
    clip_losses = [float(m[1]) for m in matches]
    t5_losses = [float(m[2]) for m in matches]

    min_clip = min(clip_losses)
    min_clip_epoch = clip_losses.index(min_clip) + 1

    print(f"\nMinimo loss CLIP: {min_clip:.4f} (epoch {min_clip_epoch})")
    print(f"Loss finale CLIP: {last_clip:.4f}")

    if last_clip > min_clip:
        print(f"\n⚠️  ATTENZIONE: Loss CLIP è AUMENTATA di {last_clip - min_clip:.4f} dopo epoch {min_clip_epoch}")
        print("    Questo indica OVERFITTING o problemi con:")
        print("    - Weight decay troppo alto")
        print("    - Dropout durante training")
        print("    - Learning rate non ottimale")

    # Check convergence pattern
    if clip_reduction_pct < 20:
        print(f"\n⚠️  Loss CLIP è scesa solo del {clip_reduction_pct:.1f}% - convergenza limitata")
        print("    Possibili cause:")
        print("    - Target embeddings troppo diversi dagli audio embeddings")
        print("    - Capacità del modello insufficiente")
        print("    - Learning rate troppo basso (attuale: 1e-4)")

print("\n" + "=" * 80)
print("RACCOMANDAZIONI")
print("=" * 80)

print("""
1. RIDURRE Weight Decay per CLIP:
   Da: clip_adapter_weight_decay: 0.01
   A:  clip_adapter_weight_decay: 0.0001 o 0.001

2. AUMENTARE Learning Rate per CLIP:
   Da: clip_adapter_learning_rate: 1e-4
   A:  clip_adapter_learning_rate: 5e-4 o 1e-3

3. RIDURRE Dropout nell'Adapter:
   Modificare Adapter() per usare dropout=0.0 o 0.05 invece di 0.1

4. VERIFICARE la qualità dei target embeddings:
   - Controllare che i text embeddings siano corretti
   - Verificare che la dimensione corrisponda (768)

5. CONSIDERARE un warmup del learning rate:
   - Iniziare con LR più basso e aumentare gradualmente
""")

print("=" * 80)
