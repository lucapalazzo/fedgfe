# Embeddings Checkpoint System - Guide

Sistema per salvare gli embeddings (T5/CLIP) in checkpoint e generare le immagini in un secondo momento, separando la fase di inferenza degli adapter dalla fase di generazione con FLUX.

## Panoramica

Il sistema permette di:
1. **Fase 1**: Eseguire il training federato e salvare gli embeddings in checkpoint (veloce)
2. **Fase 2**: Generare le immagini dai checkpoint usando FLUX (lento, può essere fatto offline)

### Vantaggi
- ✅ Riduce il tempo di training federato
- ✅ Permette di rigenerare le immagini con diversi parametri FLUX
- ✅ Facilita il debugging (embeddings salvati permanentemente)
- ✅ Permette generazione batch su hardware dedicato

---

## 1. Salvare Embeddings Durante il Training

### Opzione A: Nel Client (per ogni nodo)

```python
# Durante o dopo il training del nodo
checkpoint_path = client.save_embeddings_to_checkpoint(
    split='all',  # 'train', 'val', 'test', o 'all'
    checkpoint_dir='checkpoints/embeddings',
    round_num=current_round
)
```

### Opzione B: Nel Server (embeddings globali)

```python
# Nel server, dopo aggregazione
checkpoint_path = server.save_server_embeddings_to_checkpoint(
    checkpoint_dir='checkpoints/embeddings',
    round_num=current_round
)
```

### Struttura Checkpoint

Ogni checkpoint contiene:

```python
{
    'node_id': 0,  # o 'server'
    'round': 5,
    'timestamp': '2026-01-22T12:00:00',
    'embeddings': [
        {
            'split': 'train',
            'index': 0,
            'class_name': 'dog',
            'image_path': '/path/to/output_images/node_0_train_dog_0_r5.png',
            'image_filename': 'node_0_train_dog_0_r5.png',
            't5_embedding': tensor([...]),  # Shape: [1, 512, 4096]
            'clip_embedding': tensor([...]),  # Shape: [1, 77, 768]
            'sample_metadata': {
                'sample_id': 'train_0',
                'audio_path': '/path/to/audio.wav'
            }
        },
        # ... più embeddings
    ],
    'metadata': {
        'diffusion_type': 'flux',
        'selected_classes': ['dog', 'cat', 'bird']
    }
}
```

---

## 2. Generare Immagini dai Checkpoint

### Script: `generate_images_from_checkpoint.py`

Lo script legge i checkpoint e genera le immagini usando FLUX.

### Uso Base

```bash
# Generare da un singolo checkpoint
python system/generate_images_from_checkpoint.py \
    --checkpoint checkpoints/embeddings/node_0_embeddings_r5.pt

# Processare tutti i checkpoint in una directory
python system/generate_images_from_checkpoint.py \
    --checkpoint_dir checkpoints/embeddings/

# Specificare output directory custom
python system/generate_images_from_checkpoint.py \
    --checkpoint_dir checkpoints/embeddings/ \
    --output_dir output_images/round_5/
```

### Opzioni Avanzate

```bash
# Usare Stable Diffusion invece di FLUX
python system/generate_images_from_checkpoint.py \
    --checkpoint_dir checkpoints/embeddings/ \
    --diffusion_type sd

# Generazione batch (più veloce)
python system/generate_images_from_checkpoint.py \
    --checkpoint_dir checkpoints/embeddings/ \
    --batch_size 4

# Usare CPU invece di GPU
python system/generate_images_from_checkpoint.py \
    --checkpoint_dir checkpoints/embeddings/ \
    --device cpu
```

### Output

```
Loading checkpoint: checkpoints/embeddings/node_0_embeddings_r5.pt
Checkpoint info:
  Node ID: 0
  Round: 5
  Timestamp: 2026-01-22T12:00:00
  Total embeddings: 320
  Classes: ['dog']

Generating 320 images...
Generating images (batch_size=1): 100%|████████| 320/320 [10:20<00:00,  1.94s/it]

✓ Complete! Generated 320 images
```

---

## 3. Integrare nel Workflow Federato

### Modificare Config JSON

Aggiungi parametri per abilitare il salvataggio embeddings:

```json
{
  "feda2v": {
    "save_embeddings_instead_of_images": true,
    "embeddings_checkpoint_dir": "checkpoints/embeddings/experiment1",
    "embeddings_save_frequency": 5,

    "generate_nodes_images_frequency": 0,
    "generate_global_images_frequency": 0
  }
}
```

### Modificare il Server Hook

Aggiungi nel `round_ending_hook` del server:

```python
def round_ending_hook(self):
    # ... codice esistente ...

    # Salva embeddings invece di generare immagini
    if self.config.feda2v.get('save_embeddings_instead_of_images', False):
        if self.current_round % self.config.feda2v.get('embeddings_save_frequency', 1) == 0:
            checkpoint_dir = self.config.feda2v.get('embeddings_checkpoint_dir', 'checkpoints/embeddings')

            # Salva embeddings di ogni nodo
            for client in self.clients:
                client.save_embeddings_to_checkpoint(
                    split='all',
                    checkpoint_dir=checkpoint_dir,
                    round_num=self.current_round
                )

            # Salva embeddings globali del server
            self.save_server_embeddings_to_checkpoint(
                checkpoint_dir=checkpoint_dir,
                round_num=self.current_round
            )
```

---

## 4. Workflow Completo Esempio

### Passo 1: Training Federato (Salva Embeddings)

```bash
# Config con save_embeddings_instead_of_images=true
python system/main.py --config configs/a2v_vegas_10n_10c_400s_avg.json

# Output:
# - Training completo in 2 ore
# - Checkpoint salvati in: checkpoints/embeddings/
#   - node_0_embeddings_r1.pt
#   - node_1_embeddings_r1.pt
#   - ...
#   - node_9_embeddings_r1.pt
#   - server_embeddings_r1.pt
```

### Passo 2: Generare Immagini Offline

```bash
# Genera tutte le immagini dai checkpoint
python system/generate_images_from_checkpoint.py \
    --checkpoint_dir checkpoints/embeddings/ \
    --batch_size 4

# Output:
# - Immagini generate in: output_images/vegas-10n-10c-400s-avg/
# - Tempo: ~5-10 ore (può essere fatto overnight)
```

### Passo 3: Rigenerare con Parametri Diversi

```bash
# Rigenerare le stesse immagini con steps diversi
python system/generate_images_from_checkpoint.py \
    --checkpoint checkpoints/embeddings/node_0_embeddings_r5.pt \
    --output_dir output_images/high_quality/ \
    --num_inference_steps 100 \
    --guidance_scale 10.0
```

---

## 5. Struttura Directory

```
fedgfe/
├── checkpoints/embeddings/          # Checkpoint embeddings
│   ├── experiment1/
│   │   ├── node_0_embeddings_r1.pt
│   │   ├── node_0_embeddings_r5.pt
│   │   ├── server_embeddings_r1.pt
│   │   └── ...
│   └── experiment2/
│       └── ...
├── output_images/                   # Immagini generate
│   ├── vegas-10n-10c-400s-avg/
│   │   ├── node_0_train_dog_0_r1.png
│   │   └── ...
│   └── high_quality/
│       └── ...
└── system/
    └── generate_images_from_checkpoint.py
```

---

## 6. Vantaggi per Esperimenti

### Scenario 1: Debugging Embeddings

```bash
# Analizza gli embeddings senza generare immagini
python -c "
import torch
ckpt = torch.load('checkpoints/embeddings/node_0_embeddings_r5.pt')
print(f'Embeddings: {len(ckpt[\"embeddings\"])}')
for emb in ckpt['embeddings'][:5]:
    print(f'{emb[\"class_name\"]}: t5_shape={emb[\"t5_embedding\"].shape}')
"
```

### Scenario 2: Comparare Round Diversi

```bash
# Genera immagini solo per round specifici
for round in 1 5 10 15 20; do
    python system/generate_images_from_checkpoint.py \
        --checkpoint checkpoints/embeddings/node_0_embeddings_r${round}.pt \
        --output_dir output_images/round_${round}/
done
```

### Scenario 3: Generazione Distribuita

```bash
# Split checkpoint per generazione parallela su più GPU
# GPU 0: nodes 0-4
python system/generate_images_from_checkpoint.py \
    --checkpoint_dir checkpoints/embeddings/ \
    --pattern "node_[0-4]*.pt" \
    --device cuda:0

# GPU 1: nodes 5-9
python system/generate_images_from_checkpoint.py \
    --checkpoint_dir checkpoints/embeddings/ \
    --pattern "node_[5-9]*.pt" \
    --device cuda:1
```

---

## 7. Troubleshooting

### Problema: Out of Memory durante generazione

**Soluzione**: Usa batch_size=1 e libera cache

```bash
python system/generate_images_from_checkpoint.py \
    --checkpoint_dir checkpoints/embeddings/ \
    --batch_size 1 \
    --clear_cache_every 10
```

### Problema: Embeddings shape mismatch

**Verifica**: Controlla la shape degli embeddings

```python
import torch
ckpt = torch.load('checkpoint.pt')
print(ckpt['metadata']['diffusion_type'])  # Deve corrispondere a quello usato
```

### Problema: Immagini non generate

**Debug**: Attiva logging verbose

```bash
python system/generate_images_from_checkpoint.py \
    --checkpoint checkpoint.pt \
    --verbose \
    2>&1 | tee generation.log
```

---

## 8. Riferimenti

- **Client method**: `clientA2V.save_embeddings_to_checkpoint()` (line ~4818)
- **Server method**: `serverA2V.save_server_embeddings_to_checkpoint()` (line ~864)
- **Generation script**: `system/generate_images_from_checkpoint.py`

---

## Esempio Completo SLURM

```bash
#!/bin/bash
#SBATCH --job-name=gen_imgs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

cd $HOME/fedgfe
source $(conda info --base)/etc/profile.d/conda.sh
conda activate flvit

# Genera tutte le immagini
python system/generate_images_from_checkpoint.py \
    --checkpoint_dir checkpoints/embeddings/vegas_10n_10c/ \
    --batch_size 2 \
    --output_dir output_images/final/

echo "Generation complete!"
```
