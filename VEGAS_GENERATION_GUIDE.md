# VEGAS Image Generation Guide

## Overview
Config per generare immagini usando generatori VAE e adapter preallenati, senza fare training.

## File Creato
**Config**: `configs/a2v_vegas_2n_2c_200s_generate.json`

---

## Workflow Completo

### 1. Training dei Generatori (FATTO)
```bash
python system/main.py -c configs/a2v_vegas_2n_2c_200s_train_generator.json
```
**Output**: Generatori VAE salvati in `checkpoints/generators/vegas-2n-2c-200s/`

### 2. Training degli Adapter (FATTO)
```bash
python system/main.py -c configs/a2v_vegas_2n_2c_200s_train_adapters.json
```
**Output**: Adapter T5/CLIP salvati in `checkpoints/adapters/vegas-2n-2c-200s/`

### 3. Generazione Immagini (NUOVO)
```bash
python system/main.py -c configs/a2v_vegas_2n_2c_200s_generate.json
```
**Output**: 400 immagini totali (200 per nodo) in `output_images/vegas-2n-2c-200s-generate/`

---

## Configurazione Dettagliata

### Parametri Chiave

#### Generatori (Pretrained)
```json
{
  "use_generator": true,
  "use_pretrained_generators": true,
  "generator_load_checkpoint": true,
  "generator_checkpoint_dir": "checkpoints/generators/vegas-2n-2c-200s",
  "generator_training_mode": false  // NO training!
}
```

#### Adapter (Pretrained)
```json
{
  "adapter_load_checkpoint": true,
  "adapter_checkpoint_dir": "checkpoints/adapters/vegas-2n-2c-200s",
  "global_model_train": false  // NO training!
}
```

#### Generazione Immagini
```json
{
  "generate_nodes_images_frequency": 1,  // Genera ad ogni round
  "generate_from_clip_text_embeddings": true,
  "generate_from_t5_text_embeddings": true,
  "images_output_dir": "output_images/vegas-2n-2c-200s-generate"
}
```

#### Nodi
```json
{
  "nodes": {
    "0": {
      "selected_classes": ["dog"],
      "samples_per_node": 200  // Genera 200 immagini di "dog"
    },
    "1": {
      "selected_classes": ["chainsaw"],
      "samples_per_node": 200  // Genera 200 immagini di "chainsaw"
    }
  }
}
```

#### Rounds
```json
{
  "global_rounds": 1,  // Solo 1 round (generazione unica)
  "local_epochs": 1
}
```

---

## Processo di Generazione

### Per ogni nodo:

1. **Carica generatore preallenato** per la sua classe
   - Nodo 0: generatore "dog"
   - Nodo 1: generatore "chainsaw"

2. **Carica adapter preallenati** (T5 + CLIP)

3. **Genera N campioni sintetici** (N = samples_per_node = 200)
   - Usa il generatore VAE per creare embeddings audio sintetici

4. **Passa attraverso adapter**
   - `t5_embeddings = adapter.t5(synthetic_audio)`
   - `clip_embeddings = adapter.clip(synthetic_audio)`

5. **Genera immagini** con modello diffusion (FLUX)
   - Input: T5 + CLIP embeddings
   - Output: Immagine corrispondente

6. **Salva immagini**
   - Path: `output_images/vegas-2n-2c-200s-generate/node_{id}/`
   - 200 immagini per nodo

---

## Output Atteso

```
output_images/vegas-2n-2c-200s-generate/
├── node_0/
│   ├── dog_0001.png
│   ├── dog_0002.png
│   ├── ...
│   └── dog_0200.png  (200 immagini totali)
│
└── node_1/
    ├── chainsaw_0001.png
    ├── chainsaw_0002.png
    ├── ...
    └── chainsaw_0200.png  (200 immagini totali)

TOTALE: 400 immagini
```

---

## Verifica Pre-Esecuzione

### Checkpoint Necessari

Verifica che esistano:

```bash
# Generatori
ls -lh checkpoints/generators/vegas-2n-2c-200s/
# Dovrebbe mostrare: vae_perclass_round_X_dog.pth, vae_perclass_round_X_chainsaw.pth

# Adapter
ls -lh checkpoints/adapters/vegas-2n-2c-200s/
# Dovrebbe mostrare: vegas-2n-2c-200s_round_X_clip.pth, vegas-2n-2c-200s_round_X_t5.pth
```

Se i checkpoint non esistono, esegui prima gli step 1 e 2 (training).

---

## Parametri da Modificare (Opzionali)

### Numero di Immagini per Nodo
```json
{
  "nodes": {
    "0": {
      "samples_per_node": 500  // Cambia qui per generare più/meno immagini
    }
  }
}
```

### Classi Diverse
```json
{
  "nodes": {
    "0": {
      "selected_classes": ["cat"]  // Usa un'altra classe (deve avere generatore)
    }
  }
}
```

### Output Directory
```json
{
  "images_output_dir": "output_images/my_custom_generation"
}
```

### Usare Solo T5 o Solo CLIP
```json
{
  "generate_from_clip_text_embeddings": true,
  "generate_from_t5_text_embeddings": false  // Solo CLIP
}
```

---

## Troubleshooting

### Errore: "Checkpoint not found"
**Causa**: I generatori/adapter non sono stati trainati
**Soluzione**: Esegui prima gli step 1 e 2

### Errore: "Generator for class 'X' not found"
**Causa**: Il generatore per quella classe non esiste
**Soluzione**: Modifica `selected_classes` con classi disponibili

### Poche immagini generate
**Causa**: `samples_per_node` troppo basso
**Soluzione**: Aumenta il valore nel config

### Out of Memory
**Causa**: Troppi samples generati contemporaneamente
**Soluzione**:
- Riduci `batch_size`
- Riduci `samples_per_node`
- Abilita `generate_low_memomy_footprint: true` (già attivo)

---

## Note Tecniche

### Memory Footprint
- **Generatori VAE**: ~100MB per generatore
- **Adapter**: ~50MB (T5) + ~20MB (CLIP)
- **FLUX diffusion**: ~5-6GB VRAM
- **Totale stimato**: ~6-7GB VRAM

### Performance
- **Tempo per immagine**: ~1-2 secondi (con FLUX on GPU)
- **200 immagini**: ~5-10 minuti per nodo
- **400 immagini totali**: ~10-20 minuti

### Qualità
- Dipende da:
  1. Qualità dei generatori trainati
  2. Qualità degli adapter trainati
  3. Diversità dei campioni audio originali
  4. Parametri del modello FLUX

---

## Workflow di Sviluppo

### Test Rapido (10 immagini)
```json
{
  "nodes": {
    "0": {"samples_per_node": 10},
    "1": {"samples_per_node": 10}
  }
}
```

### Produzione (200+ immagini)
```json
{
  "nodes": {
    "0": {"samples_per_node": 200},
    "1": {"samples_per_node": 200}
  }
}
```

### Dataset Completo (1000+ immagini)
```json
{
  "nodes": {
    "0": {"samples_per_node": 1000},
    "1": {"samples_per_node": 1000}
  }
}
```

---

## Integrazione con KD Aggregation

Questo config può essere usato **dopo** l'aggregazione KD per valutare la qualità degli adapter aggregati:

1. Train adapter localmente (train_adapters.json)
2. Aggrega con KD (a2v_vegas_2n_2c_200s_kd.json)
3. **Genera immagini** con adapter aggregati (questo config)
4. Confronta qualità: adapter locali vs adapter aggregati

---

## Comandi Utili

### Esecuzione Standard
```bash
python system/main.py -c configs/a2v_vegas_2n_2c_200s_generate.json
```

### Con Log Dettagliato
```bash
python system/main.py -c configs/a2v_vegas_2n_2c_200s_generate.json --verbose
```

### Background (se supportato)
```bash
nohup python system/main.py -c configs/a2v_vegas_2n_2c_200s_generate.json > generate.log 2>&1 &
```

### Check Progress
```bash
watch -n 5 "ls -lh output_images/vegas-2n-2c-200s-generate/node_*/*.png | wc -l"
```

---

## File Correlati

- **Training Generatori**: `configs/a2v_vegas_2n_2c_200s_train_generator.json`
- **Training Adapter**: `configs/a2v_vegas_2n_2c_200s_train_adapters.json`
- **KD Aggregation**: `configs/a2v_vegas_2n_2c_200s_kd.json`
- **Generazione (questo)**: `configs/a2v_vegas_2n_2c_200s_generate.json`
