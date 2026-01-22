# Memory Leak - Riepilogo Esecutivo

**Configuration**: `configs/a2v_generator_vegas_5n_1c_real.json`
**Status**: 🔴 CRITICAL
**Data**: 2026-01-18

---

## Il Problema in 3 Punti

1. ❌ **Generazione immagini ad ogni round** (`generate_nodes_images_frequency: 1`)
2. ❌ **Funzione difettosa che perde memoria**: `get_audio_embeddings_from_dataset()`
3. ❌ **Accumulo senza cleanup**: Tensori GPU non vengono deallocati

**Risultato**: OOM (Out of Memory) dopo pochi round

---

## Il Bug Principale

**File**: `system/flcore/clients/clientA2V.py` linee **1409-1463**

### Codice Attuale (DIFETTOSO):

```python
def get_audio_embeddings_from_dataset(self, dataset):
    dataloader = DataLoader(dataset, batch_size=8)

    with torch.no_grad():
        for batch_idx, samples in enumerate(dataloader):  # Loop su tutti i batch
            audio_data = samples['audio'].to(self.device)
            audio_embeddings = ast_model(audio_inputs).last_hidden_state

            embeddings = {}  # ❌ SOVRASCRIVE ad ogni iterazione!
            for module_name in self.adapters.keys():
                adapter = self.adapters[module_name].to(self.device)  # ❌ Ripete ad ogni batch!
                output = adapter(audio_embeddings)
                embeddings[module_name] = output  # ❌ Perde il batch precedente!

        return embeddings  # ❌ Restituisce SOLO l'ultimo batch!
```

### Problemi:

1. **Loop senza accumulo**: Processa 10-15 batch ma salva solo l'ultimo
2. **Leak di memoria**: I batch precedenti rimangono in GPU senza riferimenti
3. **Return errato**: Restituisce solo l'ultimo batch invece di tutti
4. **Adapter reload**: Sposta adapter su GPU ad ogni iterazione (inutile)

### Impatto:

- Dataset 100 samples, batch 8 → 13 iterazioni
- Ogni iterazione: ~500MB leak
- **Totale: 6-7 GB per nodo per round**
- Con 5 nodi × 10 round = **350+ GB leak**

---

## La Soluzione

### File da Modificare:

1. **PRIORITY 1** (CRITICAL): `system/flcore/clients/clientA2V.py:1409-1463`
   - Sostituire `get_audio_embeddings_from_dataset()` con versione corretta
   - Fix completo in: `MEMORY_LEAK_FIX_CRITICAL.py`

2. **PRIORITY 2** (HIGH): `system/flcore/servers/serverA2V.py:3114-3165`
   - Aggiungere try-finally in `generate_images_from_diffusion()`

3. **PRIORITY 3** (MEDIUM): `system/flcore/servers/serverA2V.py:3201-3242`
   - Aggiungere cleanup in `generate_images()`

### Fix Principale (get_audio_embeddings_from_dataset):

```python
def get_audio_embeddings_from_dataset(self, dataset):
    dataloader = DataLoader(dataset, batch_size=8)

    # ✅ Accumulatori per TUTTI i batch
    all_embeddings = {module_name: [] for module_name in self.adapters.keys()}
    all_class_names = []

    # ✅ Sposta adapters UNA SOLA VOLTA
    for module_name in self.adapters.keys():
        self.adapters[module_name] = self.adapters[module_name].to(self.device)

    with torch.no_grad():
        for batch_idx, samples in enumerate(dataloader):
            try:
                audio_data = samples['audio'].to(self.device)
                audio_embeddings = ast_model(audio_inputs).last_hidden_state

                # Process attraverso adapters
                for module_name in self.adapters.keys():
                    adapter = self.adapters[module_name]  # ✅ Già su GPU
                    output = adapter(audio_embeddings)
                    all_embeddings[module_name].append(output.cpu())  # ✅ Sposta su CPU

                all_class_names.extend(samples.get('class_name', []))

            finally:
                # ✅ Cleanup dopo ogni batch
                del audio_data, audio_embeddings
                torch.cuda.empty_cache()

    # ✅ Concatena TUTTI i batch
    result = {}
    for module_name in all_embeddings.keys():
        result[module_name] = torch.cat(all_embeddings[module_name], dim=0)
    result['class_name'] = all_class_names

    return result  # ✅ Restituisce TUTTI i dati
```

**Differenze chiave**:
- ✅ Accumulatori separati per ogni batch
- ✅ Concatenazione finale di tutti i batch
- ✅ Cleanup esplicito dopo ogni iterazione
- ✅ Return corretto di tutti i dati

---

## Workaround Temporaneo

Se non puoi applicare il fix subito, **disabilita la generazione di immagini**:

```json
{
  "feda2v": {
    "generate_nodes_images_frequency": 0,
    "generate_global_images_frequency": 0
  }
}
```

Oppure **riduci la frequenza**:

```json
{
  "feda2v": {
    "generate_nodes_images_frequency": 5,  // Ogni 5 round invece di ogni round
    "save_generated_images_splits": ["test"]  // Solo test, non train/val
  }
}
```

---

## Test di Verifica

Dopo aver applicato il fix, verifica che:

1. ✅ **Tutti i sample vengano restituiti**:
   ```python
   embeddings = client.get_audio_embeddings_from_dataset(dataset)
   assert len(embeddings['class_name']) == len(dataset)  # Deve essere uguale!
   ```

2. ✅ **Memoria stabile tra round**:
   - Round 1: ~8.5 GB
   - Round 10: ~9.0 GB (< 5% crescita)

3. ✅ **Nessun OOM** dopo 10 round con 5 nodi

---

## File di Riferimento

- 📄 **Analisi completa**: `MEMORY_LEAK_ANALYSIS_VEGAS_5N_1C.md`
- 💾 **Fix implementato**: `MEMORY_LEAK_FIX_CRITICAL.py`
- ⚙️ **Configurazione**: `configs/a2v_generator_vegas_5n_1c_real.json`

---

## Azione Richiesta

**IMMEDIATE**:
1. Sostituire `get_audio_embeddings_from_dataset()` in `system/flcore/clients/clientA2V.py:1409-1463`
2. Testare con un singolo round
3. Verificare che restituisca tutti i sample

**FOLLOW-UP**:
4. Applicare fix secondari (generate_images_from_diffusion, generate_images)
5. Testare con 10 round completi
6. Monitorare memoria con memory logs

---

**Priorità**: 🔴 MASSIMA
**Impatto**: Memory leak causa OOM e blocca esperimenti
**Difficoltà Fix**: ⭐⭐ (Medium - richiede refactoring della funzione)
**Testing**: Necessario test approfondito prima del merge
