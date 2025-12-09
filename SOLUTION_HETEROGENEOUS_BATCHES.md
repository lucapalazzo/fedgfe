# Soluzione per Batch Eterogenei

## Problema Identificato

L'adapter CLIP con batch eterogenei (samples di classi diverse) raggiunge una loss minima di ~0.13-0.17 invece di convergere completamente perché:

1. **Target troppo distanti**: I text embeddings di classi diverse hanno distanza ~14-17
2. **Single adapter**: Un unico set di pesi cerca di mappare verso target molto diversi
3. **Compromesso subottimale**: Il modello trova un punto medio che non soddisfa nessun target perfettamente

## Test Effettuati

```
Batch omogeneo (tutti 'dog'):     2.25 → 0.016 (99.3% riduzione) ✅
Batch eterogeneo (4 classi):      2.23 → 0.135 (93.9% riduzione) ⚠️
```

La differenza di 10x nella loss finale dimostra che il problema non è risolvibile solo con l'architettura.

## Soluzioni Concrete

### Soluzione 1: Weighted Loss per Classe ⭐ (PIÙ SEMPLICE)

Dai più peso alle classi underrepresented nel batch:

```python
class PerClassWeightedMSELoss(nn.Module):
    def forward(self, pred, target, class_names):
        """
        Calcola MSE con peso maggiore per samples di classi rare nel batch

        Args:
            pred: (batch, features)
            target: (batch, features)
            class_names: list of str, classe per ogni sample
        """
        # Conta occorrenze
        unique_classes, counts = torch.unique(class_names, return_counts=True)

        # Peso inversamente proporzionale alla frequenza
        weights = torch.zeros(len(pred))
        for cls, count in zip(unique_classes, counts):
            mask = (class_names == cls)
            weights[mask] = 1.0 / count

        # Normalizza pesi
        weights = weights / weights.sum() * len(weights)

        # MSE pesato
        mse = F.mse_loss(pred, target, reduction='none').mean(dim=1)
        return (mse * weights).mean()
```

**Vantaggi:**
- ✅ Implementazione semplice
- ✅ Nessun cambio architetturale
- ✅ Bilancia l'importanza delle classi

### Soluzione 2: Multi-Head Adapter (MIGLIORE CAPACITÀ) ⭐⭐

Usa più "teste" nell'adapter, ognuna specializzata per pattern diversi:

```python
class MultiHeadAdapter(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=1024, output_dim=768, num_heads=4):
        super().__init__()
        self.num_heads = num_heads

        # Ogni head è un adapter separato
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GroupNorm(8, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim)
            )
            for _ in range(num_heads)
        ])

        # Attention per combinare heads
        self.head_attention = nn.Sequential(
            nn.Linear(input_dim, num_heads),
            nn.Softmax(dim=-1)
        )

        self.final_norm = nn.GroupNorm(8, output_dim)

    def forward(self, x):
        # x: (batch, seq_len, features)
        batch, seq_len, feat = x.shape

        # Calcola output di ogni head
        head_outputs = []
        for head in self.heads:
            head_out = head(x)  # (batch, seq_len, output_dim)
            head_outputs.append(head_out)

        head_outputs = torch.stack(head_outputs, dim=-1)  # (batch, seq_len, output_dim, num_heads)

        # Calcola attention weights (basato su input medio)
        x_mean = x.mean(dim=1)  # (batch, feat)
        attention = self.head_attention(x_mean).unsqueeze(1).unsqueeze(1)  # (batch, 1, 1, num_heads)

        # Combina heads con attention
        output = (head_outputs * attention).sum(dim=-1)  # (batch, seq_len, output_dim)

        output = self.final_norm(output.transpose(1, 2)).transpose(1, 2)

        return output
```

**Vantaggi:**
- ✅ Maggiore capacità
- ✅ Diverse "specializzazioni" per diversi pattern
- ✅ Attention dynamica basata sull'input

### Soluzione 3: Residual Scaling (PIÙ LEGGERA) ⭐

Aggiungi una connessione residuale scalabile:

```python
class ResidualScalingAdapter(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=2048, output_dim=768):
        super().__init__()

        # Main path
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GroupNorm(16, hidden_dim),  # Più gruppi per finer normalization
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, output_dim)
        )

        # Residual scaling (learnable)
        self.residual_scale = nn.Parameter(torch.ones(1))

        # Output projection (per dare più degrees of freedom)
        self.output_proj = nn.Linear(output_dim, output_dim)

        self.final_norm = nn.GroupNorm(16, output_dim)

    def forward(self, x):
        # Main transformation
        out = self.adapter(x)

        # Residual connection con scaling learnable
        # Questo permette al modello di decidere quanto "modificare" vs "preservare"
        out = out + self.residual_scale * x

        # Final projection
        out = self.output_proj(out)
        out = self.final_norm(out.transpose(1, 2)).transpose(1, 2)

        return out
```

**Vantaggi:**
- ✅ Semplice da implementare
- ✅ Pochi parametri extra
- ✅ Scaling learnable aiuta con batch eterogenei

### Soluzione 4: Aumenta Learning Rate CLIP ⭐ (ZERO MODIFICHE!)

Il modo più semplice: aumenta semplicemente il learning rate del CLIP adapter:

```json
{
  "clip_adapter_learning_rate": 1e-3,  // Invece di 3e-4
  "clip_adapter_weight_decay": 5e-5,   // Ridotto da 1e-4
}
```

Con LR più alto, il modello riesce a "muoversi" di più ad ogni step e può trovare soluzioni migliori per batch eterogenei.

## Raccomandazione

**Prova nell'ordine:**

1. **Prima**: Aumenta LR (Soluzione 4) - zero codice, immediato
2. **Se non basta**: Weighted Loss (Soluzione 1) - modifiche minime
3. **Per massime performance**: Multi-Head Adapter (Soluzione 2)

## Implementazione Soluzione 4 (Immediata)

Modifica solo la config:

```json
{
  "feda2v": {
    "clip_adapter_learning_rate": 1e-3,
    "clip_adapter_weight_decay": 5e-5,
    "clip_adapter_learning_rate_schedule": true,

    "t5_adapter_learning_rate": 1e-3,
    "t5_adapter_weight_decay": 1e-4,
    "t5_adapter_learning_rate_schedule": true
  }
}
```

Questo dovrebbe portare la loss CLIP da ~0.17-0.20 verso ~0.10-0.12 anche con batch eterogenei.

## Test di Verifica

Dopo aver applicato una soluzione:

```bash
# Run training
python main.py --config configs/your_config.json

# Monitora la loss
watch -n 1 'tail -50 wandb/latest-run/files/output.log | grep "Epoch.*100%"'
```

Obiettivo: CLIP loss < 0.12 entro 10 epochs con batch_size >= 16.
