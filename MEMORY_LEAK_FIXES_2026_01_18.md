# Memory Leak Fixes - 18 Gennaio 2026

## Sommario
Risolti critical memory leak che causavano CUDA OOM errors durante il training, con 68.90 GB di memoria GPU allocata su 79.19 GB totali.

## Root Causes Identificate

### 1. **LEAK PRINCIPALE: ESC50Dataset.to() sposta tutti i samples su GPU**
- **File**: `system/datautils/dataset_esc50.py:246-262`
- **Problema**: Il metodo `.to(device)` iterava su TUTTI i samples e spostava audio, audio_emb, text_emb su GPU
- **Impact**: Con migliaia di samples, questo caricava gigabyte di dati in GPU
- **Fix**: Commentato il bulk move. Il metodo ora setta solo `self.device` come preferenza. I dati vengono spostati batch-by-batch dal DataLoader.

```python
def to(self, device: torch.device):
    """
    Set the target device for the dataset.
    NOTE: We do NOT move data to GPU here to prevent memory leaks.
    Data is moved batch-by-batch by the DataLoader during training.
    """
    self.device = device
    # Commented out bulk data move (lines 248-254)
    return self
```

### 2. **LEAK SECONDARIO: NodeData.to() sposta train_data/val_data/test_data su GPU**
- **File**: `system/datautils/node_dataset.py:267-295`
- **Problema**: Il metodo `.to(device)` spostava i tensori precaricati (train_data, val_data, test_data) su GPU
- **Impact**: Se usato l'approccio legacy con NPZ files, caricava interi dataset in GPU
- **Fix**: Rimosso completamente. Mantiene solo la propagazione del device ai Dataset objects che settano solo la preferenza.

```python
def to(self, device):
    """
    Set the target device for this NodeData.

    IMPORTANT: This method does NOT move bulk data to GPU.
    - For Dataset objects: we call their .to() which sets device preference only
    - For preloaded data: we keep it on CPU to prevent OOM

    Data is moved to GPU batch-by-batch by the DataLoader during training.
    """
    self.device = device

    # Propagate device to Dataset objects (modern approach)
    if self.train_dataset is not None and hasattr(self.train_dataset, 'to'):
        self.train_dataset.to(device)
    # ... same for val and test

    # MEMORY LEAK FIX: Do NOT move train_data/val_data/test_data to GPU!
    # (Removed lines 275-304)

    return self
```

### 3. **LEAK: Adapter outputs con computational graphs**
- **File**: `system/flcore/clients/clientA2V.py:707-716`
- **Problema**: `training_adapter_outputs` accumulava tensori con computational graphs attraverso tutti i batch dell'ultima epoca
- **Impact**: Ogni batch manteneva grafo computazionale fino a fine epoca
- **Fix**: Detach immediato dopo backward() per liberare grafi computazionali

```python
# CRITICAL: Detach stored adapter outputs after backward to free GPU memory
if epoch == epochs - 1 and len(training_adapter_outputs) > 0:
    for class_name in training_adapter_outputs:
        for adapter_name in training_adapter_outputs[class_name]:
            if adapter_name != 'audio_embeddings':
                tensor_list = training_adapter_outputs[class_name][adapter_name]
                if len(tensor_list) > 0 and torch.is_tensor(tensor_list[-1]):
                    # Detach only the last added tensor (from this batch)
                    training_adapter_outputs[class_name][adapter_name][-1] = tensor_list[-1].detach()
```

### 4. **BUG FIX: Variable name collision**
- **File**: `system/flcore/clients/clientA2V.py:680`
- **Problema**: Loop variable `loss_names` sovrascriveva variabile outer scope
- **Fix**: Rinominato in `adapter_name`

## Memory Tracking Enhancements

### Aggiunti metodi di debug per training normale
- **File**: `system/flcore/clients/clientA2V.py:554-892`

Implementato sistema di tracking memoria simile a quello dei generatori:

1. **Pre-training snapshot**: Memoria all'inizio del training
2. **Per-epoch tracking**: Memoria dopo ogni epoca con delta dal start
3. **Final report**: Report dettagliato se crescita > 1GB

```python
# Memory tracking structure
epoch_memory_tracker = {
    'start': mem_start,
    'epochs': [
        {
            'epoch': 1,
            'memory': {...},
            'delta_from_start': {...},
            'avg_loss': 0.123
        },
        ...
    ],
    'end': mem_end,
    'total_growth': {...}
}
```

Output esempio quando detecta leak:
```
====================================================================================================
[Node 0] ⚠️  MEMORY GROWTH DETECTED - Round 1
====================================================================================================

📊 MEMORY SUMMARY:
  Start:  Allocated=10.50 GB, Reserved=15.20 GB (19.2%)
  End:    Allocated=12.80 GB, Reserved=17.50 GB (22.1%)
  Growth: Allocated=+2.300 GB, Reserved=+2.300 GB

📈 PER-EPOCH BREAKDOWN:
Epoch    Allocated (GB)       Reserved (GB)        Δ Alloc         Δ Reserved      Avg Loss
-------- -------------------- -------------------- --------------- --------------- ------------
1         11.20 GB             15.80 GB            +0.700 GB       +0.600 GB       0.5234
2         11.90 GB             16.50 GB            +1.400 GB       +1.300 GB       0.4821
3         12.80 GB             17.50 GB            +2.300 GB       +2.300 GB       0.4512

  🔴 Max growth at Epoch 3: +2.300 GB reserved
====================================================================================================
```

## Verification dei Dataset

Verificato che altri dataset già implementano correttamente `.to()`:

✅ **FLSplittedDataset** (`fl_splitted_dataset.py:218-221`)
- Setta solo `self.device`, non muove dati

✅ **FLNodeDataset** (`flnode_dataset.py:21-23`)
- Setta solo `self.device`, non muove dati
- Righe 34, 38, 41-42 già commentate (non spostano su GPU)

✅ **VEGASDataset** (`dataset_vegas.py:945-969`)
- Bulk move già commentato (linee 952-969)

## Testing

Dopo questi fix, la memoria GPU dovrebbe:
1. Partire da un baseline basso (solo modelli)
2. Crescere minimamente durante training (solo batch corrente + gradienti)
3. Tornare al baseline dopo ogni batch (con empty_cache ogni 5 batches)
4. Non accumulare memoria tra epoche

## Files Modificati

1. `system/datautils/dataset_esc50.py` - Fix ESC50Dataset.to()
2. `system/datautils/node_dataset.py` - Fix NodeData.to()
3. `system/flcore/clients/clientA2V.py` - Detach fix + memory tracking + bug fix

## Next Steps

Se il problema persiste:
1. Controllare output dei memory tracker (abilitare con `--debug` o impostare logging level a DEBUG)
2. Verificare se ci sono altri dataset custom che implementano `.to()` incorrettamente
3. Usare `torch.cuda.memory_summary()` per analisi dettagliata
4. Considerare `torch.cuda.set_per_process_memory_fraction(0.9)` per limitare uso GPU

## Note

- **NON usare** `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` - è un workaround che maschera il leak
- I dataset **devono** tenere dati in CPU, solo il DataLoader sposta batch su GPU
- Il metodo `.to(device)` nei dataset serve solo per settare la preferenza, non per bulk move
