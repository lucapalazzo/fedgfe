# Refactoring: `node_id` → `node_split_id` in VEGASDataset

## Data: 2026-01-20

## Motivazione

Il parametro `node_id` in VEGASDataset aveva un significato ambiguo:
- Originariamente rappresentava l'ID del nodo nella federazione (0, 1, 2, ...)
- Veniva però usato anche per identificare **split diversi degli stessi dati**

Nel caso del config `a2v_generator_vegas_20n_10c_200s_real.json`:
- 20 nodi totali con 10 classi
- Ogni classe appare su 2 nodi diversi
- I due nodi con la stessa classe devono avere **split diversi** dei campioni

Il parametro `node_split_id` è più chiaro semanticamente:
- Indica quale porzione dei dati deve essere usata
- Permette a nodi diversi con la stessa classe di avere dati non sovrapposti

## Modifiche Effettuate

### 1. VEGASDataset (`system/datautils/dataset_vegas.py`)

#### Parametri del costruttore:
- **RIMOSSO**: `node_id: Optional[int] = None`
- **AGGIUNTO**: `node_split_id: Optional[int] = None`

#### Documentazione aggiornata:
```python
node_split_id: Data split ID for this node (0-indexed). Allows multiple nodes with
              the same class to have different data splits.
              Example: Two nodes with class 'dog' can have node_split_id=0 and node_split_id=1
              to get different portions of the 'dog' samples.
```

#### Variabili interne:
- Tutte le occorrenze di `self.node_id` sostituite con `self.node_split_id`
- Aggiornati i messaggi di log per usare "Node split" invece di "Node"

#### Funzioni modificate:
- `__init__()`: Parametro rinominato
- `_create_all_splits()`: Passa `node_split_id` invece di `node_id`
- `_apply_split()`: Usa `self.node_split_id` per il random seed
- `_apply_node_split()`:
  - Usa `self.node_split_id` per calcolare gli indici
  - Documentazione migliorata per spiegare il comportamento
  - Log messages aggiornati
- `_load_samples()`: Controlla `self.node_split_id` invece di `self.node_id`
- `_get_cache_key()`: Usa `self.node_split_id` nella cache key

### 2. ServerA2V (`system/flcore/servers/serverA2V.py`)

#### Creazione VEGASDataset:
```python
# PRIMA:
node_dataset = VEGASDataset(
    ...
    node_id=node_split_id,  # Confusing: parametro config → parametro dataset
    ...
)

# DOPO:
node_dataset = VEGASDataset(
    ...
    node_split_id=node_split_id,  # Chiaro: mapping diretto
    ...
)
```

## Funzionamento

### Modalità `samples_per_node`

Con `samples_per_node=200` e due nodi con la stessa classe:

```python
# Nodo 0 (node_split_id=0)
start_idx = 0 * 200 = 0
end_idx = 200
# Prende campioni [0:200] dopo shuffle

# Nodo 10 (node_split_id=1)
start_idx = 1 * 200 = 200
end_idx = 400
# Prende campioni [200:400] dopo shuffle
```

### Seed di shuffle

Entrambi i nodi usano lo stesso `node_split_seed=42`:
- Questo garantisce che l'ordine dei campioni sia **consistente**
- Ma ogni nodo prende una **finestra diversa** di campioni
- **Risultato**: Non c'è sovrapposizione tra i campioni

## Test di Verifica

### Test 1: Base test con 2 nodi (`test_node_split_id.py`)

```
Dataset 0 (node_split_id=0):
  Total samples: 180 (200 * 0.9 train_ratio)
  First 10 file IDs: ['video_00211', 'video_00318', ...]

Dataset 1 (node_split_id=1):
  Total samples: 180
  First 10 file IDs: ['video_00168', 'video_00254', ...]

Overlapping file IDs: 0

✓ SUCCESS: The two datasets have different samples!
```

### Test 2: Configurazione completa 20 nodi (`test_20n_10c_splits.py`)

Simula esattamente il config `a2v_generator_vegas_20n_10c_200s_real.json`:
- 20 nodi totali
- 10 classi
- Ogni classe su 2 nodi con `node_split_id` diversi (0 e 1)

**Risultati:**
```
✓ OK Class 'dog': Node 0 (180 samples) vs Node 10 (180 samples) - Overlap: 0
✓ OK Class 'chainsaw': Node 1 (180 samples) vs Node 11 (180 samples) - Overlap: 0
✓ OK Class 'drum': Node 2 (180 samples) vs Node 12 (180 samples) - Overlap: 0
✓ OK Class 'rail_transport': Node 3 (180 samples) vs Node 13 (180 samples) - Overlap: 0
✓ OK Class 'helicopter': Node 4 (180 samples) vs Node 14 (180 samples) - Overlap: 0
✓ OK Class 'baby_cry': Node 5 (180 samples) vs Node 15 (180 samples) - Overlap: 0
✓ OK Class 'printer': Node 6 (180 samples) vs Node 16 (180 samples) - Overlap: 0
✓ OK Class 'snoring': Node 7 (180 samples) vs Node 17 (180 samples) - Overlap: 0
✓ OK Class 'water_flowing': Node 8 (180 samples) vs Node 18 (180 samples) - Overlap: 0
✓ OK Class 'fireworks': Node 9 (180 samples) vs Node 19 (180 samples) - Overlap: 0

Total nodes: 20
Total samples across all nodes: 3600
Average samples per node: 180.0

✓ SUCCESS: All nodes have non-overlapping samples!
```

## Compatibilità

### Breaking Changes:
- **NO**: Il parametro `node_id` è stato completamente rimosso
- Codice esistente che usa `node_id` con VEGASDataset deve essere aggiornato

### Config Files:
- I config che usano `node_split_id` nel JSON sono già compatibili
- Il parametro viene correttamente letto e passato al dataset

## Esempio di Utilizzo

```python
# Due nodi con la stessa classe ma dati diversi
dataset_node_0 = VEGASDataset(
    selected_classes=['dog'],
    samples_per_node=200,
    node_split_id=0,  # Primi 200 campioni
    node_split_seed=42
)

dataset_node_10 = VEGASDataset(
    selected_classes=['dog'],
    samples_per_node=200,
    node_split_id=1,  # Successivi 200 campioni
    node_split_seed=42
)

# I due dataset NON hanno campioni in comune
```

## Note Importanti

1. **Semantica chiara**: `node_split_id` indica quale split dei dati, non quale nodo
2. **Consistenza**: Stesso seed garantisce split riproducibili
3. **Flessibilità**: Permette configurazioni complesse (20 nodi, 10 classi ripetute)
4. **Log migliorati**: Messaggi più chiari ("Node split X" invece di "Node X")

## Files Modificati

### Core Files:
- `system/datautils/dataset_vegas.py` - Rinominato parametro `node_id` → `node_split_id` e aggiornata logica
- `system/flcore/servers/serverA2V.py` - Aggiornata chiamata a VEGASDataset per usare `node_split_id`

### Example Files:
- `system/datautils/example_vegas_esc50_usage.py` - Aggiornati esempi per usare `node_split_id`
- `system/datautils/example_vegas_usage.py` - Aggiornati esempi per usare `node_split_id`

### Test Files:
- `test_node_split_id.py` - Test base con 2 nodi e 1 classe
- `test_20n_10c_splits.py` - Test completo simulando config 20 nodi, 10 classi
