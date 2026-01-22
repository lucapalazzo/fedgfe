# AST Embedding Cache Implementation for VEGAS Dataset

## Panoramica

Implementato un sistema di caching per gli embedding AST del dataset VEGAS che:

- ✅ Salva automaticamente gli embedding AST in file `.pt` con metadata
- ✅ Verifica la compatibilità della configurazione (sample_rate, duration, model_name)
- ✅ Usa memory mapping per caricamenti veloci senza occupare tutta la RAM
- ✅ Evita ricalcoli quando la configurazione corrisponde
- ✅ Supporta cache invalidation automatica quando cambia la configurazione

## Vantaggi

### Prima (senza cache)
```
Caricamento dataset VEGAS → Estrazione AST embeddings (10-30 minuti) → Training
                              ↑ Ogni volta che si riavvia!
```

### Dopo (con cache)
```
Prima volta:
Caricamento dataset VEGAS → Estrazione AST embeddings (10-30 minuti) → Salva cache → Training

Volte successive:
Caricamento dataset VEGAS → Carica cache (10-30 secondi) → Training
                              ↑ Usa memory mapping, velocissimo!
```

## Struttura Cache

### File di cache

**Directory Raccomandata** (centralizzata):
```
/home/lpala/fedgfe/dataset/VEGAS.cache/
├── ast_embeddings_<hash1>.pt  # sample_rate=16000, duration=5.0, model=ast-finetuned
├── ast_embeddings_<hash2>.pt  # sample_rate=16000, duration=10.0, model=ast-finetuned
└── ...
```

**Directory Default** (dentro dataset):
```
/home/lpala/fedgfe/dataset/Audio/VEGAS/ast_cache/
├── ast_embeddings_<hash1>.pt
└── ...
```

**Raccomandazione**: Usare la directory centralizzata `/home/lpala/fedgfe/dataset/VEGAS.cache/` per condividere la cache tra diverse configurazioni.

Il nome del file include un hash della configurazione:
- `model_name`: Nome/versione del modello AST
- `sample_rate`: Sample rate audio (es. 16000)
- `duration`: Durata audio in secondi (es. 5.0)

### Struttura del file cache

```python
{
    'embeddings': {
        'video_00001:dog': torch.Tensor(...),  # embedding per file_id:class_name
        'video_00002:dog': torch.Tensor(...),
        ...
    },
    'metadata': {
        'sample_rate': 16000,
        'duration': 5.0,
        'model_name': 'ast-finetuned',
        'embedding_shape': (768,),  # Shape degli embedding
        'num_embeddings': 500,
        'creation_time': '2026-01-19T10:30:00',
        'dataset_root': '/home/lpala/fedgfe/dataset/Audio/VEGAS'
    }
}
```

## API

### 1. Inizializzazione Dataset con Cache

```python
from system.datautils.dataset_vegas import VEGASDataset

# Crea dataset con cache AST abilitata
dataset = VEGASDataset(
    root_dir="/home/lpala/fedgfe/dataset/Audio/VEGAS",
    selected_classes=['dog', 'baby_cry', 'chainsaw'],
    split='train',
    enable_ast_cache=True,  # Abilita cache (default True)
    ast_cache_dir="/home/lpala/fedgfe/dataset/VEGAS.cache",  # Directory centralizzata (raccomandato)
    # ast_cache_dir=None,  # Alternativa: None = root_dir/ast_cache (default)
    load_audio=True
)
```

### 2. Caricamento da Cache

```python
# Prova a caricare dalla cache
cache_loaded = dataset.load_ast_embeddings_from_cache(
    sample_rate=16000,
    duration=5.0,
    model_name="ast-finetuned"
)

if cache_loaded:
    print("✓ Caricato dalla cache!")
    # Filtra embeddings per le classi attive
    dataset.filter_audio_embeddings_from_file()
else:
    print("Cache non trovata o incompatibile")
```

### 3. Salvataggio in Cache

```python
# Estrai embeddings (esempio)
embeddings = {}
for sample in dataset.samples:
    audio = dataset._load_audio(sample['audio_path'])
    with torch.no_grad():
        emb = ast_model(audio.unsqueeze(0))

    # Usa stesso formato di audio_embs: 'file_id:class_name'
    sample_id = f"{sample['file_id']}:{sample['class_name'].lower()}"
    embeddings[sample_id] = emb.squeeze(0).cpu()

# Salva in cache
success = dataset.save_ast_embeddings_to_cache(
    embeddings=embeddings,
    sample_rate=16000,
    duration=5.0,
    model_name="ast-finetuned"
)
```

### 4. Verifica Compatibilità

La cache viene automaticamente verificata per compatibilità:

```python
# Configurazione corrente
sample_rate = 16000
duration = 5.0
model_name = "ast-finetuned"

# Verifica cache
cache_file = dataset._get_ast_cache_filepath(sample_rate, duration, model_name)
is_compatible = dataset._verify_ast_cache_compatibility(
    cache_file,
    sample_rate,
    duration,
    model_name
)
```

La cache viene considerata **incompatibile** se:
- Il file non esiste
- Sample rate diverso
- Duration diversa
- Model name diverso
- Metadata mancante o corrotto

### 5. Pulizia Cache

```python
# Pulisci tutta la cache
dataset.clear_ast_cache()

# Pulisci cache specifica
dataset.clear_ast_cache(
    sample_rate=16000,
    duration=5.0,
    model_name="ast-finetuned"
)
```

## Integrazione con ServerA2V

### Esempio di integrazione nel server

```python
# In serverA2V.py o dove estrai gli embedding AST

def extract_or_load_ast_embeddings(self, dataset):
    """
    Estrae o carica AST embeddings usando la cache.
    """
    # Configurazione AST
    sample_rate = self.audio_sample_rate  # es. 16000
    duration = self.audio_duration  # es. 5.0
    model_name = "ast-finetuned"  # o self.ast_model_name

    # 1. Prova a caricare dalla cache
    cache_loaded = dataset.load_ast_embeddings_from_cache(
        sample_rate=sample_rate,
        duration=duration,
        model_name=model_name
    )

    if cache_loaded:
        logger.info("✓ AST embeddings caricati dalla cache")
        # Filtra per classi attive
        dataset.filter_audio_embeddings_from_file()
        return

    # 2. Cache miss - estrai embeddings
    logger.info("Cache non trovata, estrazione AST embeddings in corso...")

    embeddings = {}
    self.ast_model.eval()

    for idx, sample in enumerate(dataset.samples):
        try:
            # Carica audio
            audio = dataset._load_audio(sample['audio_path'])

            # Estrai embedding
            with torch.no_grad():
                audio_input = audio.unsqueeze(0).to(self.device)
                embedding = self.ast_model(audio_input)

            # Salva embedding
            sample_id = f"{sample['file_id']}:{sample['class_name'].lower()}"
            embeddings[sample_id] = embedding.squeeze(0).cpu()

            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(dataset)} samples")

        except Exception as e:
            logger.error(f"Error processing {sample['audio_path']}: {e}")
            continue

    # 3. Salva in cache per la prossima volta
    success = dataset.save_ast_embeddings_to_cache(
        embeddings=embeddings,
        sample_rate=sample_rate,
        duration=duration,
        model_name=model_name
    )

    if success:
        logger.info(f"✓ {len(embeddings)} AST embeddings salvati in cache")

    # 4. Carica embeddings nel dataset
    dataset.audio_embs_from_file = embeddings
    dataset.filter_audio_embeddings_from_file()


# Utilizzo nel server
def prepare_data(self):
    # Crea dataset con cache abilitata
    self.train_dataset = VEGASDataset(
        root_dir="/home/lpala/fedgfe/dataset/Audio/VEGAS",
        selected_classes=self.selected_classes,
        split='train',
        enable_ast_cache=True,
        ast_cache_dir="/home/lpala/fedgfe/dataset/VEGAS.cache",  # Directory centralizzata
        load_audio=True
    )

    # Estrai o carica da cache
    self.extract_or_load_ast_embeddings(self.train_dataset)

    # Ora il dataset ha gli embeddings pronti!
```

## Memory Mapping

La cache usa `torch.load(..., mmap=True)` per memory mapping:

**Vantaggi:**
- Non carica tutto in RAM subito
- Accesso rapido agli embeddings quando servono
- Riduce drasticamente l'uso di memoria
- Perfetto per dataset grandi

**Esempio di utilizzo memoria:**

```
Senza mmap: 10 GB di embeddings → 10 GB RAM occupata
Con mmap:   10 GB di embeddings → ~100 MB RAM occupata (solo index)
```

## Cache Invalidation

La cache viene automaticamente invalidata quando:

1. **Sample rate cambia**:
   ```python
   # Cache con 16000 Hz
   dataset.load_ast_embeddings_from_cache(sample_rate=16000, ...)  # ✓ OK

   # Cambio a 44100 Hz
   dataset.load_ast_embeddings_from_cache(sample_rate=44100, ...)  # ✗ Cache miss
   ```

2. **Duration cambia**:
   ```python
   # Cache con 5.0s
   dataset.load_ast_embeddings_from_cache(duration=5.0, ...)  # ✓ OK

   # Cambio a 10.0s
   dataset.load_ast_embeddings_from_cache(duration=10.0, ...)  # ✗ Cache miss
   ```

3. **Model name cambia**:
   ```python
   # Cache con "ast-finetuned"
   dataset.load_ast_embeddings_from_cache(model_name="ast-finetuned", ...)  # ✓ OK

   # Cambio a "ast-base"
   dataset.load_ast_embeddings_from_cache(model_name="ast-base", ...)  # ✗ Cache miss
   ```

## Best Practices

### 1. Abilita sempre la cache in produzione
```python
dataset = VEGASDataset(
    ...,
    enable_ast_cache=True,  # Sempre True in produzione
    ast_cache_dir="/path/to/persistent/cache"  # Directory persistente
)
```

### 2. Usa configurazioni coerenti
```python
# Definisci configurazione in un posto solo
AST_CONFIG = {
    'sample_rate': 16000,
    'duration': 5.0,
    'model_name': 'ast-finetuned'
}

# Usa ovunque
dataset.load_ast_embeddings_from_cache(**AST_CONFIG)
dataset.save_ast_embeddings_to_cache(embeddings, **AST_CONFIG)
```

### 3. Gestisci cache miss gracefully
```python
cache_loaded = dataset.load_ast_embeddings_from_cache(...)

if not cache_loaded:
    logger.warning("Cache miss - this will take longer...")
    # Estrai embeddings
    embeddings = extract_embeddings(...)
    # Salva per la prossima volta
    dataset.save_ast_embeddings_to_cache(embeddings, ...)
```

### 4. Pulisci cache vecchia periodicamente
```python
# Quando aggiorni il modello o cambi configurazione
dataset.clear_ast_cache()  # Pulisci tutto

# O solo quella vecchia
dataset.clear_ast_cache(
    sample_rate=old_sample_rate,
    duration=old_duration,
    model_name=old_model_name
)
```

### 5. Monitor dimensioni cache
```python
import os

def get_cache_size(cache_dir):
    total_size = 0
    for filename in os.listdir(cache_dir):
        if filename.startswith('ast_embeddings_'):
            filepath = os.path.join(cache_dir, filename)
            total_size += os.path.getsize(filepath)
    return total_size / (1024**3)  # GB

size_gb = get_cache_size(dataset.ast_cache_dir)
logger.info(f"AST cache size: {size_gb:.2f} GB")
```

## Risoluzione Problemi

### Problema: Cache non viene trovata

**Causa:** Configurazione diversa tra salvataggio e caricamento

**Soluzione:**
```python
# Verifica che la configurazione sia identica
print(f"Sample rate: {sample_rate}")
print(f"Duration: {duration}")
print(f"Model name: {model_name}")

# Verifica file cache
cache_file = dataset._get_ast_cache_filepath(sample_rate, duration, model_name)
print(f"Cache file: {cache_file}")
print(f"Exists: {os.path.exists(cache_file)}")
```

### Problema: Out of memory durante il load

**Causa:** Non usi memory mapping

**Soluzione:** Il codice usa già `mmap=True`, ma verifica:
```python
# Verifica che mmap sia abilitato
cached_data = torch.load(cache_file, map_location='cpu', mmap=True)
```

### Problema: Cache corrotto

**Causa:** Interruzione durante il salvataggio

**Soluzione:** Il codice usa atomic write (temp file + rename), ma se necessario:
```python
# Rimuovi cache corrotto
dataset.clear_ast_cache(sample_rate, duration, model_name)

# Rigenera
embeddings = extract_embeddings(...)
dataset.save_ast_embeddings_to_cache(embeddings, ...)
```

## File Modificati

- `system/datautils/dataset_vegas.py`: Aggiunte funzionalità cache AST
- `system/datautils/example_ast_cache_usage.py`: Esempi di utilizzo
- `AST_CACHE_IMPLEMENTATION.md`: Questa documentazione

## Testing

Vedere [example_ast_cache_usage.py](system/datautils/example_ast_cache_usage.py) per esempi completi.

## Prossimi Passi

1. **Integrare in serverA2V.py**: Usare `extract_or_load_ast_embeddings()` durante la preparazione dei dati
2. **Testare con dataset completo**: Verificare che funzioni con tutte le 10 classi VEGAS
3. **Monitoring**: Aggiungere log per tracciare hit/miss rate della cache
4. **Multi-node**: Condividere cache tra nodi in federated learning (opzionale)

## Performance Attese

| Operazione | Senza Cache | Con Cache (primo load) | Con Cache (già in RAM) |
|------------|-------------|------------------------|------------------------|
| Dataset VEGAS 500 samples | 10-15 min | 10-15 min + save | 10-30 sec |
| Dataset VEGAS 5000 samples | 1-2 ore | 1-2 ore + save | 1-3 min |
| Uso RAM | ~10 GB | ~100 MB | ~100 MB |

**Conclusione:** Dopo il primo caricamento, i tempi si riducono da ore/minuti a secondi! 🚀
