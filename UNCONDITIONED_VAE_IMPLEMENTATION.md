# Implementazione VAE Non Condizionato

## Panoramica

È stata aggiunta la possibilità di utilizzare `VAEGenerator` senza condizionamenti visivi (CLIP/T5), permettendo al generatore di apprendere la distribuzione degli audio embeddings in modo completamente non supervisionato.

## Modifiche Implementate

### 1. **VAEGenerator - Metodo sample()**
**File:** [system/flcore/trainmodel/generators.py](system/flcore/trainmodel/generators.py#L150-L162)

Aggiunto il metodo `sample()` alla classe `VAEGenerator` per generare campioni sintetici dal latent space senza conditioning:

```python
def sample(self, num_samples, device='cuda'):
    """
    Generate new samples from the latent space without conditioning.

    Args:
        num_samples (int): Number of samples to generate
        device (str): Device to generate samples on

    Returns:
        samples: [num_samples, sequence_length, output_dim]
    """
    z = torch.randn(num_samples, self.latent_dim).to(device)
    return self.decode(z)
```

### 2. **Parametro di Configurazione**
**File:** [system/flcore/clients/clientA2V.py](system/flcore/clients/clientA2V.py#L164)

Aggiunto il parametro `use_conditioned_vae` (default: `True`) per controllare il tipo di VAE:
- `True`: usa `ConditionedVAEGenerator` o `MultiModalVAEGenerator` (condizionato su CLIP/T5)
- `False`: usa `VAEGenerator` senza condizionamenti

### 3. **Inizializzazione Generatori**
**File:** [system/flcore/clients/clientA2V.py](system/flcore/clients/clientA2V.py#L1696-L1758)

Modificato `initialize_generators()` per supportare entrambe le modalità:
- **Granularità unified**: scelta tra `ConditionedVAEGenerator` e `VAEGenerator`
- **Granularità per_class/per_group**: scelta tra `MultiModalVAEGenerator` e `VAEGenerator`

### 4. **Training del Generatore**
**File:** [system/flcore/clients/clientA2V.py](system/flcore/clients/clientA2V.py#L2606-L2625)

Aggiornato `_train_single_generator()` per gestire il forward pass corretto:
- **VAEGenerator non condizionato**: `generator(audio_emb_aug)`
- **MultiModalVAEGenerator**: `generator(audio_emb_aug, clip_embedding=..., t5_embedding=...)`
- **ConditionedVAEGenerator**: `generator(audio_emb_aug, visual_condition=...)`

### 5. **Raccolta Embeddings**
**File:** [system/flcore/clients/clientA2V.py](system/flcore/clients/clientA2V.py#L2208-L2338)

Modificato `collect_embeddings_for_generator_training()`:
- Per VAE condizionato: raccoglie audio + CLIP + T5 embeddings
- Per VAE non condizionato: raccoglie solo audio embeddings

### 6. **Preparazione Dati per Training**
**File:** [system/flcore/clients/clientA2V.py](system/flcore/clients/clientA2V.py#L2356-L2409)

Aggiornato `train_node_generator()`:
- Verifica solo i campi richiesti in base al tipo di VAE
- Per VAE non condizionato: richiede solo `audio_embeddings`
- Per VAE condizionato: richiede `audio_embeddings`, `clip`, e opzionalmente `t5`

### 7. **Generazione Campioni Sintetici**
**File:** [system/flcore/clients/clientA2V.py](system/flcore/clients/clientA2V.py#L2686-L2716)

Modificato `generate_synthetic_samples()`:
- **VAE non condizionato**: genera campioni una sola volta e li assegna a tutte le classi
- **VAE condizionato**: genera campioni specifici per ogni classe basati su CLIP/T5

## File di Configurazione

### Esempio: VAE Non Condizionato
**File:** [configs/vae_unconditioned_example.json](configs/vae_unconditioned_example.json)

```json
{
  "feda2v": {
    "use_generator": true,
    "generator_type": "vae",
    "use_conditioned_vae": false,
    "generator_training_mode": true,
    "generator_only_mode": true,
    ...
  }
}
```

### Esempio: VAE Condizionato (default)
**File:** [configs/example_vae_generator_config.json](configs/example_vae_generator_config.json)

```json
{
  "feda2v": {
    "use_generator": true,
    "generator_type": "vae",
    "use_conditioned_vae": true,
    ...
  }
}
```

## Comportamento

### VAE Non Condizionato (`use_conditioned_vae: false`)

**Training:**
- Il generatore si addestra **solo** sugli audio embeddings (output AST model)
- Non utilizza embeddings CLIP o T5 come conditioning
- Impara la distribuzione latente degli audio embeddings in modo non supervisionato

**Generazione:**
- Genera campioni sintetici campionando dal latent space: `z ~ N(0, I)`
- I campioni generati sono **identici per tutte le classi**
- Non dipende da condizioni visive o testuali

**Vantaggi:**
- Più semplice e veloce
- Non richiede embeddings testuali pre-generati
- Utile per data augmentation generica

**Svantaggi:**
- Non genera campioni specifici per classe
- Minore controllo sulla generazione

### VAE Condizionato (`use_conditioned_vae: true`)

**Training:**
- Il generatore si addestra su audio embeddings **condizionati** da CLIP/T5
- Apprende la relazione tra audio e rappresentazioni visive/testuali
- Supporta FLUX (T5+CLIP) e SD (solo CLIP)

**Generazione:**
- Genera campioni condizionati su embeddings specifici per classe
- Ogni classe riceve campioni diversi basati sui suoi embeddings visivi
- Maggiore controllo e specificità

**Vantaggi:**
- Generazione class-specific
- Campioni più allineati al contenuto semantico
- Migliore per scenari multi-classe

**Svantaggi:**
- Richiede embeddings testuali pre-generati
- Maggiore complessità computazionale

## Compatibilità

Le modifiche sono **retrocompatibili**:
- Se `use_conditioned_vae` non è specificato, il default è `True` (comportamento precedente)
- Tutte le configurazioni esistenti continuano a funzionare senza modifiche
- È possibile passare da condizionato a non condizionato modificando solo il parametro config

## Testing

Per testare il VAE non condizionato:

```bash
python main.py --config configs/vae_unconditioned_example.json
```

Il sistema:
1. Raccoglierà solo audio embeddings dal dataset
2. Addestrerà il VAE senza conditioning
3. Genererà campioni sintetici dal latent space
4. Salverà i checkpoint del generatore

## Note Tecniche

- **Input dimension**: 768 (dimensione pooled degli audio embeddings da AST)
- **Sequence length**: 4 (numero di token nella sequenza)
- **Latent dimension**: 256 (dimensione dello spazio latente)
- **Output**: `[batch_size, sequence_length, input_dim]` = `[batch_size, 4, 768]`

## Architettura VAEGenerator

```
Input: [batch, 4, 768] → Flatten: [batch, 2048]
    ↓
Encoder: Linear(2048 → 512 → 512) + ReLU + Dropout
    ↓
Latent: mu, logvar [batch, 256]
    ↓
Reparameterization: z = mu + eps * std
    ↓
Decoder: Linear(256 → 512 → 512 → 2048)
    ↓
Output: [batch, 4, 768]
```

Loss: MSE + β-KL + Cosine Similarity
