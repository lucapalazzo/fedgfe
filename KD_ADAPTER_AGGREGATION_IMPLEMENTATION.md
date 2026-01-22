# Knowledge Distillation Adapter Aggregation Implementation

## Obiettivo
Implementare una nuova modalità di aggregazione degli adapter chiamata "kd" (Knowledge Distillation) che funziona come segue:

1. **Raccolta adapter dai nodi**: I nodi inviano i loro adapter al server
2. **Freeze adapter nodi**: Set adapter nodi a eval mode (frozen)
3. **Training sul server**: Il server allena **UN SOLO adapter** su campioni di **TUTTE le classi**:
   - Per ogni classe: genera campioni sintetici usando generatori preallenati (frozen)
   - Per ogni classe: passa campioni attraverso adapter nodi frozen per ottenere target
   - Calcola MSE loss tra output adapter server e output target (media adapter nodi)
   - Accumula loss di tutte le classi e aggiorna l'adapter server
4. **Distribuzione**: Carica adapter allenato in global_adapters_modules e lo invia a tutti i nodi

## Specifiche Tecniche

### Configurazione
- **adapter_aggregation_mode**: `"kd"` (nuova modalità)
- **Adapter nodi reali**: frozen (eval mode)
- **Generatori**: frozen (no gradient)
- **Epoche training**: da configurazione (`kd_training_epochs`)
- **Batch size**: da configurazione (`batch_size`)
- **Learning rate**: da configurazione (`kd_learning_rate`)
- **Samples per class**: da configurazione (`kd_samples_per_class`)

## Stato Implementazione

### ✅ COMPLETATO - Implementazione corretta finita!

#### 1. Integrazione in aggregate_parameters() ✅
- **File**: `system/flcore/servers/serverA2V.py`
- **Linee**: 2860-2866
- Aggiunto branch per `adapter_aggregation_mode == 'kd'`

#### 2. Metodo principale aggregate_adapters_knowledge_distillation() ✅
- **Linee**: 3021-3087
- Coordina tutto il processo KD in 5 step con logging completo
- **CORRETTO**: allena 1 SOLO adapter su TUTTE le classi

#### 3. Helper _freeze_nodes_adapters() ✅
- **Linee**: 3089-3098
- Freeze adapter nodi (eval mode, requires_grad=False)

#### 4. Helper _get_classes_from_nodes() ✅
- **Linee**: 3100-3118
- Raccoglie classi presenti nei nodi
- Filtra solo classi con generatori disponibili
- Return: lista di class names

#### 5. Helper _create_single_server_adapter() ✅
- **Linee**: 3120-3139
- Crea UN SOLO DownstreamSinestesiaAdapters sul server
- Return: adapter module

#### 6. Helper _train_single_adapter_kd() ✅
- **Linee**: 3141-3288
- Training loop su UN SOLO adapter con samples di TUTTE le classi
- Per ogni epoch:
  - Per ogni classe:
    - Genera samples con generator frozen
    - Ottiene target da adapter nodi frozen (media)
    - Accumula loss
  - Backward e optimizer step UNICO
- Logging: loss totale + loss per classe
- Optimizer AdamW SINGOLO per l'adapter

#### 7. Aggiornamento send_models() ✅
- **Linee**: 2831-2833
- Supporto invio adapter aggregati ai nodi per modalità KD

#### 8. Import torch.nn ✅
- **Linea**: 29
- Aggiunto `import torch.nn as nn` per MSELoss

#### 9. File configurazione esempio ✅
- **File**: `configs/a2v_generator_pretrained_kd_adapters.json`
- Config completo con parametri KD
- Esempio: 3 nodi ESC50, 5 classi/nodo

### 📊 Testing
- ⏳ Da eseguire testing end-to-end del flusso completo

## Metodi Implementati

### `aggregate_adapters_knowledge_distillation()`
Metodo principale che coordina tutto il processo KD con 1 SOLO adapter.

### `_freeze_nodes_adapters()`
- Set adapter nodi a eval mode
- Freeze parametri (requires_grad=False)

### `_get_classes_from_nodes()`
- Raccoglie tutte le classi presenti nei nodi
- Filtra solo classi con generatori disponibili
- Ritorna: lista di class names

### `_create_single_server_adapter()`
- Crea UNA SOLA istanza di `DownstreamSinestesiaAdapters`
- Ritorna: adapter module

### `_train_single_adapter_kd(server_adapter, classes_to_train, ...)`
- UN SOLO optimizer per l'adapter server
- Per ogni epoch:
  - Per ogni classe in classes_to_train:
    - Genera samples sintetici (generator frozen)
    - Ottiene target da adapter nodi frozen (media)
    - Forward adapter server
    - Accumula MSE loss
  - Backward e optimizer step UNICO con loss accumulata
- Log loss totale + loss per classe + memoria GPU

## File Modificati
- `system/flcore/servers/serverA2V.py` - implementazione principale

## File da Creare
- `configs/a2v_generator_pretrained_kd_adapters.json` - configurazione esempio

## Note Tecniche
- **Generatori**: Usano metodo `.sample(num_samples, device)` per generazione
- **Adapter structure**: Ha attributi `.clip` e `.t5` per i due encoder
- **Device management**: Usare `self.device` se disponibile, altrimenti auto-detect
- **Memory cleanup**: Fondamentale fare cleanup dopo training per evitare leak

## Parametri Configurazione KD

Nuovi parametri nel config `feda2v`:

```json
{
  "adapter_aggregation_mode": "kd",  // Attiva modalità Knowledge Distillation
  "kd_training_epochs": 10,          // Epoche training server adapters
  "kd_learning_rate": 0.001,         // Learning rate per optimizer server adapters
  "kd_samples_per_class": 100        // Numero samples sintetici per classe
}
```

Parametri esistenti riutilizzati:
- `batch_size` (da `training` section)
- `adapters_weight_decay` (da `feda2v` section)

## Come Usare

1. **Prerequisiti**:
   - Generatori preallenati per classe (checkpoint salvato)
   - Config con `use_pretrained_generators: true`
   - Config con `generator_load_checkpoint: true`

2. **Setup**:
   ```json
   "feda2v": {
     "adapter_aggregation_mode": "kd",
     "use_pretrained_generators": true,
     "generator_load_checkpoint": true,
     "generator_checkpoint_dir": "checkpoints/generators_per_class",
     "kd_training_epochs": 10,
     "kd_learning_rate": 0.001,
     "kd_samples_per_class": 100
   }
   ```

3. **Run**:
   ```bash
   python system/main.py -c configs/a2v_generator_pretrained_kd_adapters.json
   ```

## Flusso Esecuzione

```
Round N:
├─ 1. Nodi trainano adapter localmente
├─ 2. Server riceve adapter dai nodi
├─ 3. aggregate_parameters() chiamato
│  └─ aggregate_adapters_knowledge_distillation()
│     ├─ Step 1: Freeze adapter nodi (eval mode)
│     ├─ Step 2: Raccoglie classi dai nodi (con generatori)
│     ├─ Step 3: Crea 1 SOLO adapter sul server
│     ├─ Step 4: Training loop (kd_training_epochs epoche)
│     │  ├─ Per ogni epoch:
│     │  │  ├─ Per ogni classe:
│     │  │  │  ├─ Genera samples con generator frozen
│     │  │  │  ├─ Ottiene target da adapter nodi frozen (media)
│     │  │  │  ├─ Forward attraverso adapter server
│     │  │  │  └─ Accumula MSE loss
│     │  │  └─ Backward + optimizer step UNICO (loss accumulata)
│     │  └─ Log: loss totale, loss per classe, GPU memory
│     ├─ Step 5: Carica adapter allenato → global_adapters_modules
│     └─ Step 6: Cleanup memoria
└─ 4. Server invia adapter allenato a TUTTI i nodi
```

## Logging

Il sistema produce logging dettagliato:
- `[KD Aggregation]` - Step principali aggregazione
- `[KD]` - Collezione target outputs e mapping classi
- `[KD Training]` - Training loop, loss per classe/epoca, memoria GPU

Esempio output:
```
[KD Aggregation] Starting Knowledge Distillation adapter aggregation
[KD Aggregation] Config: epochs=10, batch_size=8, lr=0.001, samples_per_class=100
[KD] Frozen 2 node adapters
[KD] Found 2 classes in nodes, 2 have generators
[KD Aggregation] Classes to train on: ['dog', 'chainsaw']
[KD] Created single server adapter on device cuda
[KD Training] Node classes mapping: {0: ['dog'], 1: ['chainsaw']}
[KD Training] Training single adapter on 2 classes
[KD Training] Epoch 1/10 - Total Loss: 0.425316
[KD Training] Epoch 1/10 - Per-class Losses: dog: 0.423145, chainsaw: 0.427487
[KD Training] GPU Memory: 3245.67 MB
[KD Training] Completed training single adapter over 10 epochs
[KD Aggregation] Knowledge Distillation aggregation completed successfully
```
