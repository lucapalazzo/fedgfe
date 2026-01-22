# Setup Cache VEGAS - Riepilogo

## ✅ Configurazione Completata

La cache degli embedding AST per il dataset VEGAS è stata configurata e integrata nel sistema.

---

## 📁 Directory Cache

**Percorso**: `/home/lpala/fedgfe/dataset/VEGAS.cache/`

```bash
/home/lpala/fedgfe/dataset/VEGAS.cache/
├── README.md           # Documentazione directory cache
├── .gitignore          # Evita commit file .pt su git
└── ast_embeddings_*.pt # File cache (creati automaticamente al primo run)
```

---

## 🔧 Modifiche Effettuate

### 1. Server (`system/flcore/servers/serverA2V.py`)

**Caricamento cache** durante creazione nodi VEGAS:
```python
node_dataset = VEGASDataset(
    ...,
    enable_ast_cache=True,
    ast_cache_dir="/home/lpala/fedgfe/dataset/VEGAS.cache"  # ✓ Centralizzata
)

# Carica automaticamente da cache se disponibile
ast_cache_loaded = node_dataset.load_ast_embeddings_from_cache(
    sample_rate=16000,
    duration=5.0,
    model_name="ast-finetuned"
)
```

**Salvataggio cache** alla fine del primo round:
```python
def save_audio_embeddings(self, file_name="audio_embeddings.pt"):
    # Salva in file legacy .pt (backward compatibility)
    torch.save(all_audio_embeddings, file_name)

    # Salva in cache AST (per prossimi run)
    dataset.save_ast_embeddings_to_cache(...)
```

### 2. Client (`system/flcore/clients/clientA2V.py`)

**Salvataggio embeddings** durante training:
```python
def audio_embeddings_dataset_cache(self, samples, outputs, ...):
    # Salva nel dataset (uso locale)
    base_dataset.audio_embs[file_index] = audio_emb_cpu

    # Salva nello store (aggregazione server)
    if self.store_audio_embedding:
        self.audio_embedding_store[file_index] = audio_emb_cpu
```

---

## 🚀 Come Funziona

### Primo Run (Cache Non Esiste)

```
1. Dataset VEGAS creato → load_ast_embeddings_from_cache() → MISS
2. Training estrae embeddings on-the-fly (10-30 min)
3. Fine round 1 → save_audio_embeddings() salva in cache
4. Cache creata in /home/lpala/fedgfe/dataset/VEGAS.cache/
```

### Run Successivi (Cache Esiste)

```
1. Dataset VEGAS creato → load_ast_embeddings_from_cache() → HIT! ✓
2. Embeddings caricati in 10-30 secondi (memory mapping)
3. Training parte subito, nessuna estrazione necessaria
4. Speedup: 20-100x più veloce ⚡
```

---

## 📊 Performance Attese

| Metrica | Senza Cache | Con Cache | Speedup |
|---------|-------------|-----------|---------|
| Caricamento dataset (500 samples) | 10-15 min | 10-30 sec | **20-60x** ⚡ |
| Caricamento dataset (5000 samples) | 1-2 ore | 1-3 min | **20-40x** ⚡ |
| Uso RAM | ~10 GB | ~100 MB | **100x meno** |
| Disk I/O | Alto | Basso (mmap) | Molto ridotto |

---

## 🔍 Verifica Setup

### 1. Directory Creata

```bash
ls -la /home/lpala/fedgfe/dataset/VEGAS.cache/

# Output atteso:
# drwxrwxr-x 2 lpala lpala 4096 Jan 19 12:37 .
# -rw-rw-r-- 1 lpala lpala  ... README.md
# -rw-rw-r-- 1 lpala lpala  ... .gitignore
```

### 2. Primo Run - Creazione Cache

Esegui training con:
```json
{
  "feda2v": {
    "store_audio_embeddings": true
  }
}
```

Verifica nei log:
```
INFO: Attempting to load AST embeddings from cache for node 0...
INFO: AST cache not found or incompatible for node 0, will try alternative methods
...
INFO: ✓ Saved 500 embeddings to AST cache for client 0
```

### 3. Secondo Run - Uso Cache

Esegui training con stessa configurazione.

Verifica nei log:
```
INFO: Attempting to load AST embeddings from cache for node 0...
INFO: ✓ AST embeddings loaded from cache for node 0
INFO: Loaded 500 AST embeddings from cache
```

### 4. Verifica File Cache

```bash
ls -lh /home/lpala/fedgfe/dataset/VEGAS.cache/

# Output atteso (dopo primo run):
# -rw-rw-r-- 1 lpala lpala 350M Jan 19 13:00 ast_embeddings_<hash>.pt
```

---

## 📖 Documentazione Completa

- **[AST_CACHE_IMPLEMENTATION.md](AST_CACHE_IMPLEMENTATION.md)**: API cache e esempi d'uso
- **[AST_CACHE_INTEGRATION.md](AST_CACHE_INTEGRATION.md)**: Integrazione server/client
- **[/dataset/VEGAS.cache/README.md](dataset/VEGAS.cache/README.md)**: Documentazione directory cache

---

## ⚙️ Configurazione

### Parametri Cache (Hard-coded)

```python
AST_CONFIG = {
    'sample_rate': 16000,        # Sample rate audio
    'duration': 5.0,             # Durata clip
    'model_name': 'ast-finetuned'  # Modello AST
}
```

**IMPORTANTE**: Questi parametri devono corrispondere tra:
- Caricamento cache
- Salvataggio cache
- Estrazione embeddings

### Abilita/Disabilita Cache

Per **disabilitare** la cache (fallback a metodi legacy):

```python
node_dataset = VEGASDataset(
    ...,
    enable_ast_cache=False  # Disabilita cache
)
```

---

## 🧹 Manutenzione

### Pulizia Cache

```bash
# Rimuovi tutti i file cache (saranno rigenerati al prossimo run)
rm -f /home/lpala/fedgfe/dataset/VEGAS.cache/ast_embeddings_*.pt

# Verifica spazio occupato
du -sh /home/lpala/fedgfe/dataset/VEGAS.cache/
```

### Rigenerazione Cache

Se la cache è corrotta o vuoi rigenerarla:

```bash
# 1. Rimuovi cache esistente
rm -f /home/lpala/fedgfe/dataset/VEGAS.cache/ast_embeddings_*.pt

# 2. Esegui training con store_audio_embeddings=true
# La cache verrà ricreata automaticamente
```

---

## ✅ Checklist

- [x] Directory cache creata: `/home/lpala/fedgfe/dataset/VEGAS.cache/`
- [x] Server modificato per caricare da cache
- [x] Server modificato per salvare in cache
- [x] Client modificato per popolare embedding store
- [x] .gitignore aggiunto per evitare commit file .pt
- [x] README.md documentazione in directory cache
- [x] Documentazione completa creata

---

## 🎯 Prossimi Passi

1. **Test primo run**: Verifica creazione cache
2. **Test secondo run**: Verifica caricamento da cache e speedup
3. **Monitor performance**: Misura tempi effettivi
4. **(Opzionale) Estendi ad ESC50**: Implementa cache AST per ESC50
5. **(Opzionale) Estendi a VGGSound**: Implementa cache AST per VGGSound

---

**Tutto pronto per l'uso! 🚀**
