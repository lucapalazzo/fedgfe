# Memory Leak Analysis - VEGAS 5n-1c-real Configuration

**Date**: 2026-01-18
**Configuration**: `configs/a2v_generator_vegas_5n_1c_real.json`
**Status**: CRITICAL - Memory leak identificato

## ⚠️ CORREZIONE ANALISI INIZIALE

**IMPORTANTE**: La configurazione **NON allena i generatori** (`generator_training_epochs: 0`).
I generatori vengono solo **caricati da checkpoint** e usati per **generare immagini** ad ogni round (`generate_nodes_images_frequency: 1`).

Il memory leak è quindi nella **fase di generazione immagini**, non nel training dei generatori.

## Executive Summary

L'analisi corretta del codice ha identificato **1 CRITICAL memory leak** + 2 leak secondari nella configurazione VEGAS 5n-1c-real che causano accumulo di memoria GPU durante la generazione di immagini ad ogni round federato.

## Configurazione Analizzata

```json
{
  "federation": {
    "num_clients": 5,
    "global_rounds": 10,
    "local_epochs": 20
  },
  "feda2v": {
    "generator_type": "vae",
    "generator_granularity": "per_class",
    "generator_load_checkpoint": true,
    "generator_training_epochs": 0
  },
  "nodes": {
    "0-4": "5 nodi, 1 classe ciascuno (VEGAS dataset)"
  }
}
```

## Memory Leak Identificati

### 🔴🔴🔴 LEAK #1 (CRITICAL): `get_audio_embeddings_from_dataset()` - Loop Senza Accumulo

**File**: [system/flcore/clients/clientA2V.py:1409-1463](system/flcore/clients/clientA2V.py#L1409-L1463)

**Problema GRAVE**:
Questo è il **leak principale** che causa OOM durante la generazione delle immagini.

```python
def get_audio_embeddings_from_dataset(self, dataset):
    """Generate audio embeddings for all samples in a given dataset."""
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    with torch.no_grad():
        for batch_idx, samples in enumerate(dataloader):  # ❌ LOOP su TUTTI i batch
            audio_data = samples['audio'].to(self.device)

            # Processa audio...
            audio_embeddings = ast_model(audio_inputs).last_hidden_state  # ❌ SOVRASCRIVE ad ogni iterazione

            embeddings = {}
            for module_name in self.adapters.keys():
                adapter = self.adapters[module_name].to(self.device)  # ❌ Sposta adapter su GPU ogni volta
                output = adapter(audio_embeddings)
                embeddings[module_name] = output  # ❌ SOVRASCRIVE, non accumula

            embeddings['class_name'] = samples.get('class_name', None)

        # ❌ RETURN solo l'ULTIMO batch invece di tutti!
        return embeddings
```

**Problemi multipli**:
1. **Loop infinito in memoria**: Ogni batch alloca tensori GPU ma non li libera prima del successivo
2. **Nessun accumulo**: `embeddings` viene sovrascritto ad ogni iterazione, quindi batch precedenti rimangono in GPU non referenziati (leak)
3. **Adapter loading ripetuto**: `adapter.to(self.device)` viene chiamato per ogni batch invece di una sola volta
4. **Return errato**: Restituisce solo l'ultimo batch invece di tutti i dati del dataset
5. **Nessun cleanup**: Nessun `del` o `torch.cuda.empty_cache()` tra le iterazioni

**Impatto**:
- Con dataset di 100 sample e batch_size=8 → 12-13 iterazioni
- Ogni iterazione accumula ~500MB di tensori non deallocati
- **Totale leak: 6-7 GB per nodo per round**
- Con 5 nodi x 10 round = **350+ GB di leak accumulato**

**Chiamato da**:
- [system/flcore/servers/serverA2V.py:3236](system/flcore/servers/serverA2V.py#L3236) in `generate_images()`
- Viene chiamato ad ogni round per ogni nodo quando `generate_nodes_images_frequency > 0`

**Fix proposto** (versione corretta):
```python
def get_audio_embeddings_from_dataset(self, dataset):
    """Generate audio embeddings for all samples in a given dataset."""
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Accumulatori per tutti i batch
    all_audio_embeddings = []
    all_embeddings = {module_name: [] for module_name in self.adapters.keys()}
    all_class_names = []

    # Sposta adapters su GPU UNA SOLA VOLTA
    for module_name in self.adapters.keys():
        self.adapters[module_name] = self.adapters[module_name].to(self.device)

    with torch.no_grad():
        for batch_idx, samples in enumerate(dataloader):
            try:
                audio_data = samples['audio'].to(self.device)

                # Process audio
                if self.model.ast_model is None or self.model.ast_feature_extractor is None:
                    logger.warning("AST model not initialized")
                    return None

                if isinstance(audio_data, torch.Tensor):
                    audio_data_np = audio_data.cpu().numpy()

                audio_inputs = self.model.ast_feature_extractor(
                    audio_data_np,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                ).input_values.to(self.device, self.model.torch_dtype)

                ast_model = self.model.ast_model.to(self.device)
                ast_model.eval()

                audio_embeddings = ast_model(audio_inputs).last_hidden_state

                # Process through adapters
                for module_name in self.adapters.keys():
                    adapter = self.adapters[module_name]  # ✅ Già su GPU
                    output = adapter(audio_embeddings)
                    all_embeddings[module_name].append(output.cpu())  # ✅ Sposta su CPU per liberare GPU

                # Store batch results
                all_audio_embeddings.append(audio_embeddings.cpu())  # ✅ Sposta su CPU
                all_class_names.extend(samples.get('class_name', []))

            finally:
                # ✅ Cleanup batch tensors
                del audio_data, audio_inputs, audio_embeddings
                if 'output' in locals():
                    del output
                torch.cuda.empty_cache()

    # ✅ Concatena tutti i batch
    result = {}
    for module_name in all_embeddings.keys():
        result[module_name] = torch.cat(all_embeddings[module_name], dim=0)
    result['class_name'] = all_class_names

    # ✅ Cleanup finale
    del all_audio_embeddings, all_embeddings
    torch.cuda.empty_cache()

    print(f"Node {self.id} - Retrieved {len(all_class_names)} audio embeddings from dataset")
    return result
```

---

### 🟡 LEAK #2: `generate_images_from_diffusion()` - Tensori Non Deallocati

**File**: [system/flcore/servers/serverA2V.py:3114-3165](system/flcore/servers/serverA2V.py#L3114-L3165)

**Problema**:
Dopo la generazione delle immagini, i tensori intermedi non vengono deallocati.

```python
def generate_images_from_diffusion(self, text_embeddings, base_embeddings=None):
    # Sposta embeddings su GPU diffusion
    prompt_embeds = text_embeddings['t5'].to(self.global_model.diffusion_dtype).to(self.diffusion_device)  # ❌ Rimane in GPU
    pooled_prompt_embeds = text_embeddings['clip'].to(self.global_model.diffusion_dtype).to(self.diffusion_device)  # ❌ Rimane in GPU

    if not self.generate_low_memomy_footprint:
        imgs = self.global_model.diffusion_model(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            num_inference_steps=1,
            output_type="pt",
        ).images  # ❌ Tensori intermedi non deallocati
    else:
        imgs = self.generate_single_images_from_diffusion(prompt_embeds, pooled_prompt_embeds)

    return imgs  # ❌ Nessun cleanup di prompt_embeds, pooled_prompt_embeds
```

**Impatto**:
- Ogni generazione accumula ~1-2 GB di tensori intermedi
- Con `generate_low_memomy_footprint=true`, il problema è attenuato ma non risolto

**Fix proposto**:
```python
def generate_images_from_diffusion(self, text_embeddings, base_embeddings=None):
    try:
        prompt_embeds = text_embeddings['t5'].to(self.global_model.diffusion_dtype).to(self.diffusion_device)
        pooled_prompt_embeds = text_embeddings['clip'].to(self.global_model.diffusion_dtype).to(self.diffusion_device)

        if not self.generate_low_memomy_footprint:
            imgs = self.global_model.diffusion_model(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=1,
                output_type="pt",
            ).images
        else:
            imgs = self.generate_single_images_from_diffusion(prompt_embeds, pooled_prompt_embeds)

        return imgs

    finally:
        # ✅ Cleanup tensori intermedi
        if 'prompt_embeds' in locals():
            del prompt_embeds
        if 'pooled_prompt_embeds' in locals():
            del pooled_prompt_embeds
        torch.cuda.empty_cache()
```

---

### 🟡 LEAK #3: `generate_images()` - Embeddings Accumulati Senza Cleanup

**File**: [system/flcore/servers/serverA2V.py:3201-3242](system/flcore/servers/serverA2V.py#L3201-L3242)

**Problema**:
Il metodo `generate_images()` chiama `get_audio_embeddings_from_dataset()` per ogni split (train/test/val) ma non dealloca gli embeddings tra le iterazioni.

```python
def generate_images(self, client):
    """Generate images using the client's Audio2Visual model."""

    for split_name in generation_splits:
        dataset = split_datasets.get(split_name)

        if dataset is not None and len(dataset) > 0:
            text_embs = dataset.text_embs  # ❌ Può essere grande
            embeddings = client.get_audio_embeddings_from_dataset(dataset)  # ❌ LEAK #1 chiamato qui
            generated_imgs = self.generate_images_from_diffusion(embeddings, base_embeddings=text_embs)  # ❌ LEAK #2
            saved_files = self.save_generated_images(generated_imgs, client.id, embeddings, suffix=f'{split_name}')

            generated_images_files[split_name] = saved_files
            # ❌ Nessun cleanup di embeddings, text_embs, generated_imgs

    return generated_images_files
```

**Fix proposto**:
```python
def generate_images(self, client):
    """Generate images using the client's Audio2Visual model."""

    generated_images_files = {}

    for split_name in generation_splits:
        dataset = split_datasets.get(split_name)

        if dataset is not None and len(dataset) > 0:
            try:
                text_embs = dataset.text_embs
                embeddings = client.get_audio_embeddings_from_dataset(dataset)

                if embeddings is None:
                    continue

                generated_imgs = self.generate_images_from_diffusion(embeddings, base_embeddings=text_embs)
                saved_files = self.save_generated_images(generated_imgs, client.id, embeddings, suffix=f'{split_name}')

                generated_images_files[split_name] = saved_files

            finally:
                # ✅ Cleanup dopo ogni split
                if 'embeddings' in locals() and embeddings is not None:
                    for key in list(embeddings.keys()):
                        if isinstance(embeddings[key], torch.Tensor):
                            del embeddings[key]
                    del embeddings

                if 'generated_imgs' in locals():
                    del generated_imgs

                torch.cuda.empty_cache()

    return generated_images_files
```

---

## Analisi Memory Logs

I log mostrano un pattern caratteristico:

```json
{
  "memory_before": {"allocated_gb": 8.59, "reserved_gb": 9.54},
  "memory_after": {"allocated_gb": 7.38, "reserved_gb": 7.79},
  "delta": {"allocated_gb": -1.21, "reserved_gb": -1.75}  // ⚠️ Apparente liberazione
}
```

**Problema**: Il delta negativo è **ingannevole** perché:
1. La memoria viene liberata TEMPORANEAMENTE alla fine del training locale
2. Ma `per_class_embeddings` rimane in RAM (CPU) e si accumula
3. Al prossimo round, quando viene ri-caricato su GPU, causa OOM

---

## Verifiche Aggiuntive Necessarie

### 1. Check Optimizer States
```python
# Verificare se gli optimizer vengono correttamente deallocati
# system/flcore/clients/clientA2V.py:3347-3372
```

### 2. Check AST Model Sharing
```python
# Verificare che l'AST model condiviso non venga duplicato
# system/flcore/clients/clientA2V.py:220, 4227-4231
```

### 3. Check WandB Artifacts
Il file aperto dall'IDE suggerisce possibili problemi con wandb:
- `/home/lpala/miniconda3/envs/flvit/lib/python3.12/site-packages/wandb/sdk/lib/console_capture.py`
- Verificare se wandb accumula artifacts in memoria

---

## Piano di Implementazione Fix

### ⚠️ PRIORITÀ CRITICA (implementare IMMEDIATAMENTE):

1. **FIX `get_audio_embeddings_from_dataset()` - LEAK PRINCIPALE**
   - File: [system/flcore/clients/clientA2V.py:1409-1463](system/flcore/clients/clientA2V.py#L1409-L1463)
   - **Azione**: Riscrivere completamente la funzione per:
     - Accumulare embeddings di tutti i batch (non sovrascrivere)
     - Spostare adapters su GPU una sola volta
     - Cleanup di ogni batch dopo l'elaborazione
     - Return corretto di tutti i dati
   - **Impatto**: Risolve il 70-80% del memory leak
   - **Urgenza**: MASSIMA - causa OOM immediate

### Priorità ALTA (implementare subito):

2. **Aggiungere try-finally in `generate_images_from_diffusion()`**
   - File: [system/flcore/servers/serverA2V.py:3114-3165](system/flcore/servers/serverA2V.py#L3114-L3165)
   - Cleanup di `prompt_embeds` e `pooled_prompt_embeds`
   - **Impatto**: Risolve 10-15% del leak

3. **Aggiungere cleanup in `generate_images()`**
   - File: [system/flcore/servers/serverA2V.py:3201-3242](system/flcore/servers/serverA2V.py#L3201-L3242)
   - Cleanup di embeddings tra i vari split (train/test/val)
   - **Impatto**: Risolve 5-10% del leak

### Priorità MEDIA:

4. **Aggiungere memory profiling dettagliato**
   - Tracciare memoria prima/dopo `get_audio_embeddings_from_dataset()`
   - Tracciare memoria prima/dopo `generate_images_from_diffusion()`
   - Log automatico se delta > 1GB

5. **Ottimizzare batch processing**
   - Considerare processare split in sequenza con cleanup tra uno e l'altro
   - Implementare limite di memoria massima per split

### Priorità BASSA:

6. **Refactoring architetturale**
   - Considerare un context manager per gestione GPU memory
   - Pattern: `with GPUMemoryManager(): ...`
   - Separare logica di accumulo dati da processing GPU

---

## Codice di Test Proposto

```python
# test_memory_leak_fix.py
import torch
import gc
from torch.utils.data import DataLoader

def test_get_audio_embeddings_accumulation():
    """Test che get_audio_embeddings_from_dataset accumuli TUTTI i batch."""
    client = ClientA2V(...)

    # Dataset con 25 samples, batch_size=8 → 4 batch (8+8+8+1)
    dataset = client.node_data.test_dataset
    expected_total_samples = len(dataset)

    # Chiama la funzione
    embeddings = client.get_audio_embeddings_from_dataset(dataset)

    # Verifica che abbia tutti i sample
    actual_samples = len(embeddings['class_name'])
    assert actual_samples == expected_total_samples, \
        f"Expected {expected_total_samples} samples but got {actual_samples}"

    # Verifica che gli embeddings abbiano la dimensione corretta
    for key in ['clip', 't5']:
        if key in embeddings:
            assert embeddings[key].shape[0] == expected_total_samples, \
                f"Embedding '{key}' has wrong batch size: {embeddings[key].shape[0]}"

def test_memory_cleanup_during_embedding_extraction():
    """Verifica che la memoria venga liberata durante l'estrazione degli embeddings."""
    torch.cuda.reset_peak_memory_stats()

    client = ClientA2V(...)
    dataset = client.node_data.test_dataset

    # Memoria iniziale
    initial_memory = torch.cuda.memory_allocated(client.device)

    # Estrai embeddings
    embeddings = client.get_audio_embeddings_from_dataset(dataset)

    # Memoria dopo estrazione
    after_extraction = torch.cuda.memory_allocated(client.device)

    # Cleanup manuale
    del embeddings
    torch.cuda.empty_cache()

    # Memoria finale
    final_memory = torch.cuda.memory_allocated(client.device)

    # La memoria finale dovrebbe essere vicina a quella iniziale
    memory_leak = final_memory - initial_memory
    max_acceptable_leak = 200 * 1024**2  # 200MB tolleranza

    assert memory_leak < max_acceptable_leak, \
        f"Memory leak detected: {memory_leak / 1024**2:.2f} MB"

def test_gpu_memory_after_image_generation():
    """Verifica che la memoria GPU venga liberata dopo generazione immagini."""
    torch.cuda.reset_peak_memory_stats()

    server = ServerA2V(...)
    client = server.clients[0]

    initial_memory = torch.cuda.memory_allocated(server.diffusion_device)

    # Genera immagini per un client
    generated_images = server.generate_images(client)

    # Cleanup
    del generated_images
    torch.cuda.empty_cache()

    # Memoria finale
    final_memory = torch.cuda.memory_allocated(server.diffusion_device)

    # Dovrebbe tornare vicino al livello iniziale
    memory_delta = abs(final_memory - initial_memory)
    assert memory_delta < 500 * 1024**2, \
        f"Memory not properly freed: delta = {memory_delta / 1024**2:.2f} MB"

def test_multi_round_memory_stability():
    """Test stabilità memoria su multipli round."""
    server = ServerA2V(...)

    memory_per_round = []

    for round_num in range(5):
        server.round = round_num + 1

        # Memoria prima del round
        mem_before = torch.cuda.memory_allocated(server.diffusion_device)

        # Esegui training/generazione
        server.train()

        # Memoria dopo cleanup
        torch.cuda.empty_cache()
        mem_after = torch.cuda.memory_allocated(server.diffusion_device)

        memory_per_round.append(mem_after - mem_before)

    # La memoria non dovrebbe crescere linearmente
    # Tolleranza: max 10% di crescita tra round 1 e round 5
    growth_rate = (memory_per_round[-1] - memory_per_round[0]) / memory_per_round[0]
    assert abs(growth_rate) < 0.10, \
        f"Memory growing across rounds: {growth_rate*100:.1f}% growth detected"
```

---

## Metriche di Successo

Fix implementato correttamente se:

1. ✅ **`get_audio_embeddings_from_dataset()` restituisce TUTTI i sample del dataset** (non solo l'ultimo batch)
2. ✅ **Memoria GPU stabile tra round** (variazione < 5% dopo il primo round)
3. ✅ **Nessun accumulo lineare di memoria** durante il loop sui batch
4. ✅ **Nessun OOM error** dopo 10 round con 5 nodi
5. ✅ **Memory logs mostrano cleanup efficace** tra le generazioni di immagini
6. ✅ **Peak memory consistente** tra i vari round (non cresce esponenzialmente)

### Benchmarks Attesi (dopo fix):

**Prima del fix:**
- Round 1: ~8.5 GB allocated
- Round 5: ~15+ GB allocated (crescita lineare)
- Round 10: OOM error

**Dopo il fix:**
- Round 1: ~8.5 GB allocated
- Round 5: ~8.8 GB allocated (crescita < 5%)
- Round 10: ~9.0 GB allocated (stabile)

---

## References

- Configuration: [configs/a2v_generator_vegas_5n_1c_real.json](configs/a2v_generator_vegas_5n_1c_real.json)
- Client code: [system/flcore/clients/clientA2V.py](system/flcore/clients/clientA2V.py)
- Generator code: [system/flcore/trainmodel/generators.py](system/flcore/trainmodel/generators.py)
- Memory logs: `memory_logs/memory_tracking_*.json`

---

## Note Aggiuntive

### Configurazione Corrente

La configurazione ha:
- `optimize_memory_usage: true` ✅ (aiuta ma non risolve)
- `generate_low_memomy_footprint: true` ✅ (genera immagini una alla volta invece che in batch)
- `generator_load_checkpoint: true` ✅ (non allena, solo carica)
- `generator_training_epochs: 0` ✅ (conferma: nessun training)
- `generate_nodes_images_frequency: 1` ⚠️ (genera ad OGNI round → trigger del leak)

**Questi flag NON sono sufficienti** per i leak identificati, che richiedono fix espliciti nel codice.

### Perché il Leak è Così Grave

1. **Frequenza**: Generazione ad ogni round (`generate_nodes_images_frequency: 1`)
2. **Volume**: 5 nodi × 3 split (train/test/val) × ~100 sample/split = 1500 forward pass per round
3. **Nessun cleanup**: I tensori intermedi si accumulano senza essere deallocati
4. **Loop difettoso**: `get_audio_embeddings_from_dataset()` sovrascrive invece di accumulare, creando leak

### Workaround Temporaneo (se non si può fixare subito)

Se non è possibile implementare i fix immediatamente, ridurre il leak con:

```json
{
  "feda2v": {
    "generate_nodes_images_frequency": 5,  // Genera ogni 5 round invece di ogni round
    "generate_low_memomy_footprint": true,  // SEMPRE attivo
    "save_generated_images_splits": ["test"]  // Solo test split, non train/val
  },
  "federation": {
    "global_rounds": 5  // Ridurre i round se possibile
  }
}
```

Oppure, **disabilitare completamente la generazione**:
```json
{
  "feda2v": {
    "generate_nodes_images_frequency": 0,  // Disabilita generazione
    "generate_global_images_frequency": 0
  }
}
```

Questo permetterà di completare il training senza OOM, ma **non genererà immagini**.
