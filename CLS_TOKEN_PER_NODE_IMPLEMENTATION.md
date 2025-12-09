# Implementazione Configurazione CLS Token Per Nodo

## Riepilogo

Estensione della funzionalità `use_cls_token_only` per supportare configurazione a livello di singolo nodo, permettendo ad ogni nodo di scegliere indipendentemente se utilizzare solo il CLS token o l'intera sequenza dall'output AST.

## Modifiche Implementate

### 1. File Modificati

#### `system/flcore/clients/clientA2V.py` (linee 90-102)

**Prima:**
```python
self.use_cls_token_only = self.feda2v_config.get('use_cls_token_only', False)
```

**Dopo:**
```python
# Get use_cls_token_only configuration
# Priority: node_config > feda2v global config > default (False)
global_use_cls_token = self.feda2v_config.get('use_cls_token_only', False) if self.feda2v_config else False
node_use_cls_token = getattr(node_config, 'use_cls_token_only', None)

if node_use_cls_token is not None:
    # Node-specific configuration takes priority
    self.use_cls_token_only = node_use_cls_token
    logger.info(f"Node {node_id}: Using node-specific use_cls_token_only={self.use_cls_token_only}")
else:
    # Fall back to global configuration
    self.use_cls_token_only = global_use_cls_token
    logger.info(f"Node {node_id}: Using global use_cls_token_only={self.use_cls_token_only}")
```

**Funzionalità:**
- Legge configurazione da due sorgenti: globale (`feda2v`) e specifica del nodo
- Applica priorità: nodo > globale > default
- Log informativi per tracciare quale configurazione viene utilizzata

### 2. Documentazione Aggiornata

#### `CLS_TOKEN_CONFIGURATION.md`

Aggiunte sezioni:
- **Configurazione JSON con priorità** - Spiega i tre livelli di configurazione
- **Opzione 2: Configurazione Per Nodo** - Esempi di configurazione per singolo nodo
- **Opzione 3: Configurazione Mista** - Combinazione di globale e override
- **Casi d'Uso per Configurazione Per Nodo** - Scenari pratici di utilizzo
- **Logging migliorato** - Dettagli sui messaggi di log per ogni livello

### 3. File di Configurazione Creati

#### `configs/a2v_cls_token_mixed_nodes.json`

Configurazione di esempio con 5 nodi:
- **Node 0** (Animals): `use_cls_token_only: true` - Override specifico
- **Node 1** (Nature): `use_cls_token_only: false` - Override specifico
- **Node 2** (Human): `use_cls_token_only: true` - Override specifico
- **Node 3** (Interior): `use_cls_token_only: false` - Override specifico
- **Node 4** (Exterior): Nessun override - usa configurazione globale (false)

Questo permette di confrontare performance tra diverse configurazioni nello stesso esperimento.

## Gerarchia di Configurazione

```
┌─────────────────────────────────────────┐
│  Livello 1: Configurazione Nodo         │  PRIORITÀ MASSIMA
│  "nodes": {                             │
│    "0": {                               │
│      "use_cls_token_only": true         │
│    }                                    │
│  }                                      │
└─────────────────────────────────────────┘
                    ↓ (se non specificato)
┌─────────────────────────────────────────┐
│  Livello 2: Configurazione Globale      │  PRIORITÀ MEDIA
│  "feda2v": {                            │
│    "use_cls_token_only": false          │
│  }                                      │
└─────────────────────────────────────────┘
                    ↓ (se non specificato)
┌─────────────────────────────────────────┐
│  Livello 3: Default                     │  PRIORITÀ MINIMA
│  use_cls_token_only = False             │
└─────────────────────────────────────────┘
```

## Esempi di Configurazione

### Esempio 1: Tutti i nodi usano CLS token

```json
{
  "feda2v": {
    "use_cls_token_only": true
  },
  "nodes": {
    "0": { /* usa globale: true */ },
    "1": { /* usa globale: true */ },
    "2": { /* usa globale: true */ }
  }
}
```

**Risultato:** Tutti i nodi usano CLS token

### Esempio 2: Configurazione mista con override

```json
{
  "feda2v": {
    "use_cls_token_only": false
  },
  "nodes": {
    "0": { "use_cls_token_only": true },  // OVERRIDE
    "1": { /* usa globale: false */ },
    "2": { "use_cls_token_only": true }   // OVERRIDE
  }
}
```

**Risultato:**
- Node 0: CLS token only (override)
- Node 1: Full sequence (globale)
- Node 2: CLS token only (override)

### Esempio 3: Ogni nodo configurato individualmente

```json
{
  "feda2v": {
    /* no global config */
  },
  "nodes": {
    "0": { "use_cls_token_only": true },
    "1": { "use_cls_token_only": false },
    "2": { "use_cls_token_only": true }
  }
}
```

**Risultato:**
- Node 0: CLS token only
- Node 1: Full sequence
- Node 2: CLS token only

## Vantaggi della Configurazione Per Nodo

### 1. Flessibilità Sperimentale
- Confrontare direttamente performance tra diverse configurazioni
- A/B testing nello stesso run federato
- Valutare impatto su diversi tipi di dati

### 2. Ottimizzazione Risorse
- Nodi con GPU limitata → CLS token (meno memoria)
- Nodi con GPU potente → Full sequence (più espressività)
- Bilanciamento automatico delle risorse

### 3. Specializzazione
- Adattare configurazione al tipo di dati del nodo
- Ottimizzare per caratteristiche specifiche del dataset
- Sperimentare diverse architetture in parallelo

### 4. Graduale Rollout
- Testare nuova funzionalità su subset di nodi
- Rollout progressivo con monitoraggio
- Rollback facile per singoli nodi

## Verifica della Configurazione

### Durante l'avvio

I log mostrano quale configurazione viene applicata:

```bash
INFO: Node 0: Using node-specific use_cls_token_only=True
INFO: Node 1: Using node-specific use_cls_token_only=False
INFO: Node 2: Using global use_cls_token_only=False
```

### Durante il training

I log confermano l'estrazione del CLS token:

```bash
INFO: Using CLS token only. Original shape: torch.Size([8, 1214, 768]), CLS shape: torch.Size([8, 1, 768])
```

## Testing

### Test Manuale

```bash
# Usa configurazione mista
python main.py --config configs/a2v_cls_token_mixed_nodes.json

# Monitora i log per verificare configurazione
grep "use_cls_token_only" logs/output.log
```

### Test Automatico

Lo script `test_cls_token_feature.py` già esistente testa la funzionalità base. Per testare la configurazione per nodo, verifica che:
1. I log mostrino la configurazione corretta per ogni nodo
2. I nodi con `use_cls_token_only=true` producano tensor con shape `(batch, 1, 768)`
3. I nodi con `use_cls_token_only=false` producano tensor con shape `(batch, 1214, 768)`

## Backward Compatibility

✅ **Completamente retrocompatibile**

- Configurazioni esistenti continuano a funzionare senza modifiche
- Il comportamento default rimane invariato (`use_cls_token_only=false`)
- Se non specificato né a livello globale né per nodo, usa il default

### Esempi di Compatibilità

```json
// Configurazione vecchia (ancora valida)
{
  "feda2v": {
    "use_cls_token_only": true
  }
}
// ✅ Funziona: tutti i nodi usano CLS token

// Configurazione senza il parametro (ancora valida)
{
  "feda2v": {
    // nessuna menzione di use_cls_token_only
  }
}
// ✅ Funziona: tutti i nodi usano full sequence (default)

// Nuova configurazione mista
{
  "feda2v": {
    "use_cls_token_only": false
  },
  "nodes": {
    "0": { "use_cls_token_only": true }
  }
}
// ✅ Funziona: Node 0 usa CLS, altri usano full sequence
```

## File Modificati/Creati

### Modificati
1. `system/flcore/clients/clientA2V.py` - Logica di priorità configurazione
2. `CLS_TOKEN_CONFIGURATION.md` - Documentazione aggiornata

### Creati
1. `configs/a2v_cls_token_mixed_nodes.json` - Esempio configurazione mista
2. `CLS_TOKEN_PER_NODE_IMPLEMENTATION.md` - Questo documento

### Esistenti (non modificati)
1. `system/flcore/trainmodel/downstreamsinestesiaadapters.py` - Già supporta il parametro
2. `system/flcore/servers/serverA2V.py` - Già supporta configurazione globale
3. `configs/a2v_cls_token_only_example.json` - Esempio configurazione globale
4. `test_cls_token_feature.py` - Test funzionalità base

## Prossimi Passi

Per utilizzare la nuova funzionalità:

1. **Scegli la strategia di configurazione:**
   - Globale: tutti i nodi uguali
   - Per nodo: controllo granulare
   - Mista: combinazione dei due

2. **Crea/modifica il file di configurazione:**
   - Parti da un esempio esistente
   - Aggiungi `use_cls_token_only` dove necessario
   - Verifica la sintassi JSON

3. **Esegui l'esperimento:**
   ```bash
   python main.py --config configs/your_config.json
   ```

4. **Monitora i log:**
   - Verifica configurazione applicata a ciascun nodo
   - Controlla shape dei tensor
   - Valuta performance

5. **Analizza i risultati:**
   - Confronta metriche tra nodi con diverse configurazioni
   - Valuta trade-off memoria/performance
   - Ottimizza configurazione basandosi sui risultati

## Note Implementative

### Gestione della Priorità

Il codice implementa la priorità usando Python's pattern:
```python
node_use_cls_token = getattr(node_config, 'use_cls_token_only', None)
if node_use_cls_token is not None:
    # Usa configurazione nodo
else:
    # Usa configurazione globale
```

Questo permette di distinguere tra:
- `use_cls_token_only: false` (configurazione esplicita)
- Campo non presente (usa default successivo)

### Logging

Due livelli di logging:
1. **INFO durante inizializzazione** - Quale configurazione viene applicata
2. **INFO durante forward pass** - Conferma estrazione CLS token

Questo permette debug efficace e verifica della configurazione.

## Riferimenti

- Implementazione originale CLS token: `CLS_TOKEN_CONFIGURATION.md`
- Test: `test_cls_token_feature.py`
- Esempio globale: `configs/a2v_cls_token_only_example.json`
- Esempio misto: `configs/a2v_cls_token_mixed_nodes.json`
