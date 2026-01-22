# Integrazione Generatori nel Server Federato

## Panoramica

I generatori VAE/GAN sono stati integrati lato server per allenare gli adapter globali aggregati utilizzando gli output ricevuti dai client. Questo approccio migliora il federated learning preservando la privacy e creando sample sintetici.

## Architettura

```
┌─────────────────────────────────────────────────┐
│               CLIENT SIDE                        │
├─────────────────────────────────────────────────┤
│                                                  │
│  Audio → AST → Audio Embeddings (512-dim)       │
│                       ↓                          │
│              Local Adapters (CLIP, T5)          │
│                       ↓                          │
│            Text Embeddings per classe            │
│                                                  │
│  Invia al server:                                │
│  - training_adapter_outputs_mean (per classe)   │
│  - adapters weights                              │
│                                                  │
└─────────────────────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────┐
│               SERVER SIDE                        │
├─────────────────────────────────────────────────┤
│                                                  │
│  1. AGGREGATION:                                 │
│     FedAvg degli adapter weights → Global Adapter│
│                                                  │
│  2. GENERATOR TRAINING:                          │
│     ┌─────────────────────────────┐             │
│     │  VAE/GAN Generator          │             │
│     │  Input: noise/latent        │             │
│     │  Output: synthetic prompts  │             │
│     └────────────┬────────────────┘             │
│                  ↓                               │
│     Target: adapter outputs aggregati dai client │
│                                                  │
│  3. ADAPTER REFINEMENT:                          │
│     Global Adapter + Generated Prompts           │
│     → Fine-tuning degli adapter aggregati        │
│                                                  │
└─────────────────────────────────────────────────┘
```

## File Modificati

### 1. `system/flcore/servers/serverA2V.py`

**Aggiunte principali:**

- **Import dei generatori** (linee 51-53):
  ```python
  from flcore.trainmodel.generators import ConditionedVAEGenerator, VAELoss, GANGenerator, GANDiscriminator
  import torch.nn.functional as F
  ```

- **Inizializzazione nel `__init__`** (linee 102-111):
  - Configurazione generatore (VAE o GAN)
  - Variabili per i moduli del generatore
  - Flag `use_generator` per abilitare/disabilitare

- **Metodo `initialize_generators()`** (linee 271-334):
  - Inizializza VAE condizionato o GAN
  - Crea optimizer dedicati
  - Supporta sia CLIP che T5 per FLUX

- **Metodo `train_generator_from_class_prompts()`** (linee 583-660):
  - Allena il generatore VAE/GAN usando gli output degli adapter dai client
  - Calcola VAE loss (reconstruction + KL divergence + similarity)
  - Per GAN: allena discriminator e generator

- **Metodo `train_gan_step()`** (linee 662-708):
  - Step di training per GAN
  - Discriminator: distingue prompts reali da quelli fake
  - Generator: cerca di ingannare il discriminator

- **Metodo `finetune_adapters_with_prompts()`** (linee 710-784):
  - Fine-tuning degli adapter globali aggregati
  - Usa sample reali dai client + sample sintetici dal generatore
  - Ottimizza gli adapter per produrre output migliori

- **Aggiornamento `round_ending_hook()`** (linee 463-483):
  - Integrazione del training del generatore nel loop federato
  - Logging delle metriche su wandb
  - Gestione corretta dei valori di ritorno

### 2. `configs/a2v_esc50_3n_generate.json`

**Nuove configurazioni** (linee 25-34):

```json
{
  "feda2v": {
    "adapter_aggregation_mode": "avg",
    "global_model_train": true,
    "global_model_train_from_nodes_adapters": true,
    "global_model_train_epochs": 5,
    "use_generator": true,
    "generator_type": "vae"
  }
}
```

## Parametri di Configurazione

### `feda2v` section:

| Parametro | Tipo | Default | Descrizione |
|-----------|------|---------|-------------|
| `use_generator` | bool | false | Abilita l'uso del generatore |
| `generator_type` | str | "vae" | Tipo di generatore: "vae" o "gan" |
| `global_model_train` | bool | false | Abilita training del modello globale |
| `global_model_train_from_nodes_adapters` | bool | false | Usa output adapter dai client |
| `global_model_train_epochs` | int | 1 | Epoche di training per generatore/adapter |
| `adapter_aggregation_mode` | str | "none" | Modalità aggregazione: "avg", "weighted", "none" |

## Flusso di Esecuzione

### Round Federato:

1. **Client Training**:
   - Ogni client allena i propri adapter locali
   - Salva `training_adapter_outputs_mean` per classe
   - Invia adapter weights e output al server

2. **Server Aggregation**:
   - Aggrega adapter weights con FedAvg → `global_adapters`
   - Raccoglie output adapter per classe da tutti i client

3. **Generator Training** (se `use_generator=true`):
   - VAE: impara a ricostruire audio embeddings condizionati su visual embeddings
   - GAN: genera prompts sintetici che "ingannano" il discriminator
   - Loss: reconstruction + KL divergence + similarity (VAE) o adversarial (GAN)

4. **Adapter Fine-tuning**:
   - Usa sample reali + sintetici per fine-tuning
   - Ottimizza `global_adapters` per produrre output più vicini ai target
   - Loss: MSE tra output adapter e target dai client

5. **Send to Clients**:
   - Invia `global_adapters` aggiornati ai client

## Vantaggi

1. **Privacy-Preserving**: I client non condividono dati raw, solo output aggregati
2. **Data Augmentation**: Il generatore crea sample sintetici per classi con pochi dati
3. **Continual Learning**: Il generatore impara la distribuzione globale dei prompt
4. **Federated Distillation**: Distilla conoscenza dai client nel server
5. **Robustezza**: Riduce overfitting su dati limitati dei singoli client

## Metriche Wandb

Il server logga automaticamente:

- `server/generator_loss`: Loss del generatore (VAE o GAN)
- `server/adapter_finetuning_loss`: Loss del fine-tuning degli adapter
- `round`: Numero del round corrente

## Esempio di Utilizzo

```bash
# Con generatore VAE abilitato
python main.py -config configs/a2v_esc50_3n_generate.json

# Output atteso:
# Round 1: Training Generator and Global Adapters
#   Class 'airplane': VAE Loss=0.5234 (Recon=0.3421, KL=0.1234, Sim=0.0579)
#   Class 'breathing': VAE Loss=0.4987 (Recon=0.3156, KL=0.1289, Sim=0.0542)
# Generator training completed. Average loss: 0.5110
# Adapter fine-tuning completed. Average loss: 0.2345
# Generator Loss: 0.5110, Adapter Fine-tuning Loss: 0.2345
```

## Debug e Troubleshooting

### Generator non inizializzato:

```python
# Verifica che use_generator sia true
self.use_generator = getattr(self.config.feda2v, 'use_generator', False)

# Controlla i log
logger.info(f"Initializing {self.generator_type} generator on server")
```

### Nessun output dai client:

```python
# Verifica che i client abbiano training_adapter_outputs_mean
if not all_class_prompts:
    logger.warning("No adapter outputs received from clients")
```

### Dimensioni incompatibili:

```python
# Assicurati che le dimensioni siano corrette:
# - Audio embeddings: 512 (AST output)
# - CLIP embeddings: 768
# - T5 embeddings: 4096 (solo per FLUX)
```

## Sviluppi Futuri

1. **Novelty Loss**: Integrazione di `compute_novelty_loss` per incentivare diversità
2. **Class-Conditional GAN**: GAN condizionata su label di classe
3. **Progressive Training**: Aumento graduale della complessità del generatore
4. **Multi-Modal Generator**: Generatore che produce sia CLIP che T5 embeddings simultaneamente
5. **Personalized Generators**: Un generatore per ogni client per personalization

## References

- Paper VAE: [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- Paper GAN: [Generative Adversarial Networks](https://arxiv.org/abs/1406.2661)
- Federated Learning: [Communication-Efficient Learning](https://arxiv.org/abs/1602.05629)
