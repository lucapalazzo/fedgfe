# Generator Reset and Sharing - Implementation Complete

## Panoramica

Implementazione completata delle funzionalità di **condivisione** e **reset** dei generatori in modalità `generator_only_mode`. Questa implementazione permette:

1. ✅ **Condivisione del generatore tra nodi**: Tutti i client usano la stessa istanza del generatore dal server
2. ✅ **Reset automatico dei parametri**: I generatori possono essere resettati quando appaiono nuove classi
3. ✅ **Ottimizzazione memoria**: Il server non carica il diffusion model quando in `generator_only_mode`

---

## Modifiche Implementate

### 1. Server: Ottimizzazione Caricamento Diffusion Model

**File:** `system/flcore/servers/serverA2V.py` (linee 366-372)

**Modifica:**
```python
# Solo se NON in generator_only_mode, carica il diffusion model
if not self.generator_only_mode and (self.generate_global_images_frequency or ...):
    self.global_model.start_diffusion(...)
```

**Beneficio:** Risparmio di ~8-20GB VRAM quando si fa solo training dei generatori

---

### 2. Client: Ricezione Generatori Condivisi

**File:** `system/flcore/clients/clientA2V.py` (linee 3466-3500)

**Modifica:**
```python
# I client ricevono i generatori dal server quando:
# 1. use_generator=True (comportamento normale), OR
# 2. shared_generator_in_only_mode=True AND generator_only_mode=True (modalità condivisa)

should_use_server_generators = self.use_generator or (shared_mode and generator_only_mode)

if should_use_server_generators:
    # Ricevono il RIFERIMENTO ai generatori (non una copia!)
    self.prompt_generators = self.server.prompt_generators
    self.generator_optimizers = self.server.generator_optimizers
    self.generator_loss_fn = self.server.generator_loss_fn
    self.model.generators_dict = self.prompt_generators
```

**Beneficio:** I client allenano la stessa istanza condivisa del generatore

---

### 3. Client: Parametri di Configurazione

**File:** `system/flcore/clients/clientA2V.py` (linee 215-220)

**Nuovi parametri:**
```python
# Generator sharing and reset configuration
self.shared_generator_in_only_mode = getattr(self.feda2v_config, 'shared_generator_in_only_mode', True)
self.reset_generator_on_class_change = getattr(self.feda2v_config, 'reset_generator_on_class_change', False)

# Track previously trained classes for reset detection
self.previously_trained_classes = set()
```

---

### 4. Client: Funzione `reset_generator_parameters()`

**File:** `system/flcore/clients/clientA2V.py` (linee 1988-2129)

**Funzionalità:**
- Resetta i pesi del generatore ricreandolo da zero
- Ricrea anche l'optimizer con stato fresco
- Supporta tutte e tre le granularità:
  - `unified`: Resetta il singolo generatore condiviso
  - `per_class`: Resetta solo il generatore della classe specificata
  - `per_group`: Resetta solo il generatore del gruppo specificato

**Esempio di utilizzo:**
```python
# Unified generator
self.reset_generator_parameters()

# Per-class generator
self.reset_generator_parameters(generator_key="dog")

# Per-group generator
self.reset_generator_parameters(generator_key="animals")
```

---

### 5. Client: Logica di Reset in `train_node_generator()`

**File:** `system/flcore/clients/clientA2V.py` (linee 2886-2924)

**Funzionalità:**
```python
if self.reset_generator_on_class_change:
    current_classes = set(classes_to_train)
    new_classes = current_classes - self.previously_trained_classes

    if new_classes:
        # Rileva nuove classi e resetta i generatori appropriati
        if self.generator_granularity == 'unified':
            # Reset completo del generatore unificato
            self.reset_generator_parameters()

        elif self.generator_granularity == 'per_class':
            # Reset solo dei generatori delle nuove classi
            for new_class in new_classes:
                self.reset_generator_parameters(generator_key=new_class)

        elif self.generator_granularity == 'per_group':
            # Reset solo dei generatori dei gruppi affetti
            for affected_group in affected_groups:
                self.reset_generator_parameters(generator_key=affected_group)

    # Aggiorna il set delle classi già viste
    self.previously_trained_classes.update(current_classes)
```

---

## Parametri di Configurazione

### `shared_generator_in_only_mode`
- **Tipo:** `boolean`
- **Default:** `true`
- **Descrizione:** Se `true`, tutti i client condividono la stessa istanza del generatore in `generator_only_mode`
- **Quando usarlo:**
  - Per training progressivo attraverso i client
  - Per ridurre uso di memoria
  - Per consistenza dei pesi tra client

### `reset_generator_on_class_change`
- **Tipo:** `boolean`
- **Default:** `false`
- **Descrizione:** Se `true`, resetta i parametri del generatore quando appaiono nuove classi
- **Quando usarlo:**
  - Class-incremental learning
  - Training sequenziale di classi
  - Per evitare catastrophic forgetting
- **Quando NON usarlo:**
  - Training incrementale continuo
  - Se si vuole mantenere conoscenza delle classi precedenti

---

## Comportamento per Granularità

### Unified (`generator_granularity: "unified"`)
- **Shared mode:** Un singolo generatore condiviso tra tutti i client
- **Reset mode:** Il generatore viene resettato quando appare QUALSIASI nuova classe
- **Uso memoria:** Minimo (1 sola istanza di generatore)

### Per-Class (`generator_granularity: "per_class"`)
- **Shared mode:** Ogni classe ha il suo generatore, condiviso tra i client
- **Reset mode:** Solo i generatori delle NUOVE classi vengono resettati
- **Uso memoria:** Medio (N generatori, dove N = numero di classi)

### Per-Group (`generator_granularity: "per_group"`)
- **Shared mode:** Ogni gruppo ha il suo generatore, condiviso tra i client
- **Reset mode:** Solo i generatori dei gruppi che contengono nuove classi vengono resettati
- **Uso memoria:** Medio-Alto (M generatori, dove M = numero di gruppi)

---

## File di Configurazione

### Esempio 1: Generatore Condiviso SENZA Reset (Continuous Learning)

**File:** `configs/a2v_generator_shared_with_reset.json`

```json
{
  "feda2v": {
    "generator_only_mode": true,
    "generator_granularity": "unified",
    "shared_generator_in_only_mode": true,
    "reset_generator_on_class_change": false
  }
}
```

**Comportamento:**
- ✅ Tutti i client usano lo stesso generatore
- ✅ Il generatore impara progressivamente da tutte le classi
- ✅ Nessun reset - apprendimento continuo
- ⚠️ Possibile catastrophic forgetting se classi molto diverse

---

### Esempio 2: Generatore Condiviso CON Reset (Sequential Class Learning)

**File:** `configs/a2v_generator_sequential_classes_with_reset.json`

```json
{
  "feda2v": {
    "generator_only_mode": true,
    "generator_granularity": "per_class",
    "shared_generator_in_only_mode": true,
    "reset_generator_on_class_change": true
  }
}
```

**Comportamento:**
- ✅ Ogni classe ha il proprio generatore condiviso
- ✅ Quando appare una nuova classe, il suo generatore viene resettato
- ✅ I generatori delle classi precedenti NON vengono toccati
- ✅ Ideale per class-incremental learning

---

## Flusso di Esecuzione Completo

```
┌─────────────────────────────────────────────────────────────┐
│ 1. INIZIALIZZAZIONE SERVER                                  │
│   ├─ Legge shared_generator_in_only_mode e                  │
│   │   generator_only_mode dalla config                      │
│   ├─ Crea i generatori (unified/per_class/per_group)        │
│   ├─ NON carica diffusion model se generator_only_mode=true │
│   └─ Imposta i generatori in global_model.generators_dict   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. INIZIALIZZAZIONE CLIENT (via set_server())               │
│   ├─ Riceve riferimento ai generatori del server            │
│   │   (se shared_generator_in_only_mode=true)               │
│   ├─ Riceve anche optimizer e loss function                 │
│   ├─ Inizializza previously_trained_classes = set()         │
│   └─ Aggiorna model.generators_dict                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. TRAINING (ogni round)                                     │
│   └─ train_node_generator()                                 │
│       ├─ Raccoglie embeddings dalle classi correnti         │
│       ├─ Identifica classi da allenare                      │
│       ├─ SE reset_generator_on_class_change=true:           │
│       │   ├─ Confronta current_classes vs                   │
│       │   │   previously_trained_classes                    │
│       │   ├─ Identifica new_classes                         │
│       │   └─ SE ci sono nuove classi:                       │
│       │       ├─ Unified: Reset completo generatore         │
│       │       ├─ Per-class: Reset generatori nuove classi   │
│       │       └─ Per-group: Reset generatori gruppi affetti │
│       ├─ Allena i generatori                                │
│       ├─ Aggiorna previously_trained_classes                │
│       └─ Salva checkpoint (se configurato)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Esempio di Output nel Log

### Scenario: Per-Class con Reset Attivo

```
[Client 0] Starting generator training mode (granularity=per_class)
[Client 0] Training on subset of classes: ['dog', 'rooster']
[Client 0] Detected new classes: ['rooster']
[Client 0] Previously trained classes: ['dog']
[Client 0] Resetting generators for new classes only
[Client 0]   Resetting generator for class 'rooster'
[Client 0] Resetting generator parameters (granularity=per_class, key=rooster)
[Client 0]   Created fresh conditioned VAE generator for 'rooster'
[Client 0]   Created fresh optimizer for 'rooster'
[Client 0] ✓ Generator reset complete
[Client 0] Prepared 2 class(es) with 150 total samples

Training generator for class 'dog':
  Epoch 1/10 - Loss: 12.3456
  ...

Training generator for class 'rooster':
  Epoch 1/10 - Loss: 15.7890 (fresh weights)
  ...
```

---

## Testing

### Test 1: Condivisione SENZA Reset
```bash
python main.py --config configs/a2v_generator_shared_with_reset.json
```

**Verificare:**
- ✅ Log: "Receiving X SHARED generators from server"
- ✅ Nessun messaggio di reset
- ✅ Loss del generatore diminuisce continuamente attraverso i round
- ✅ Memoria GPU stabile (no diffusion model)

### Test 2: Condivisione CON Reset
```bash
python main.py --config configs/a2v_generator_sequential_classes_with_reset.json
```

**Verificare:**
- ✅ Log: "Detected new classes: [...]"
- ✅ Log: "Resetting generator parameters"
- ✅ Loss del generatore riparte alta per nuove classi
- ✅ Loss delle classi vecchie continua a diminuire

### Test 3: Verificare Condivisione dei Pesi

```python
# Nel server, dopo training del primo client
server_gen_id = id(server.prompt_generators['dog'])

# Nel client, dopo set_server()
client_gen_id = id(client.prompt_generators['dog'])

# Devono essere IDENTICI (stessa istanza, non copia!)
assert server_gen_id == client_gen_id
```

---

## Note Tecniche Importanti

### 1. Condivisione vs Copia
- ⚠️ **CRITICO:** I client ricevono il **riferimento** al generatore del server, NON una copia
- ✅ Questo significa che modifiche ai pesi sono **immediatamente visibili** a tutti
- ✅ Gli optimizer sono condivisi, quindi lo stato di momentum/Adam è condiviso

### 2. Reset e Memoria
- Il reset fa `del` del vecchio generatore/optimizer prima di crearne uno nuovo
- Questo libera la memoria GPU del modello precedente
- PyTorch garbage collector pulirà la memoria automaticamente

### 3. Reset in Shared Mode
- ⚠️ Se `shared_generator_in_only_mode=true` e si fa un reset, il reset colpisce TUTTI i client
- Questo è intenzionale: il generatore è condiviso, quindi il reset deve essere coordinato
- Solo il client che esegue `train_node_generator()` farà il reset (grazie a `previously_trained_classes` locale)

### 4. Checkpoint con Reset
- I checkpoint salvati DOPO un reset conterranno i pesi freschi
- Questo permette di riprendere l'allenamento dopo un reset
- Il campo `previously_trained_classes` NON è salvato nel checkpoint (ogni run riparte con set vuoto)

---

## Debugging

### Verificare Configurazione
```python
# Nel client __init__
print(f"[Client {self.id}] shared_generator_in_only_mode: {self.shared_generator_in_only_mode}")
print(f"[Client {self.id}] reset_generator_on_class_change: {self.reset_generator_on_class_change}")
print(f"[Client {self.id}] generator_granularity: {self.generator_granularity}")
```

### Verificare Condivisione
```python
# Nel client set_server()
if self.prompt_generators:
    for key, gen in self.prompt_generators.items():
        server_gen = self.server.prompt_generators.get(key)
        is_same = id(gen) == id(server_gen) if server_gen else False
        print(f"[Client {self.id}] Generator '{key}': shared={is_same}, id={id(gen)}")
```

### Verificare Reset
```python
# Nel client train_node_generator(), dopo reset
print(f"[Client {self.id}] Previously trained classes: {self.previously_trained_classes}")
print(f"[Client {self.id}] Current classes: {classes_to_train}")
print(f"[Client {self.id}] New classes detected: {new_classes}")
```

---

## Compatibilità e Retrocompatibilità

### ✅ Retrocompatibile
- I parametri hanno default ragionevoli
- Config esistenti continueranno a funzionare
- `shared_generator_in_only_mode=true` per default (comportamento atteso)
- `reset_generator_on_class_change=false` per default (comportamento legacy)

### ⚠️ Breaking Changes
- Nessuno! L'implementazione è completamente retrocompatibile

---

## Performance

### Risparmio Memoria (Generator Only Mode)
- **Prima:** Server carica diffusion model (~8-20GB VRAM)
- **Dopo:** Server NON carica diffusion model
- **Risparmio:** ~8-20GB VRAM

### Risparmio Memoria (Shared Generators)
- **Prima:** Ogni client crea copie dei generatori
- **Dopo:** Un'unica istanza condivisa
- **Risparmio:** (N-1) × dimensione_generatore × num_generatori

---

## Limitazioni Note

1. **Reset Globale in Shared Mode:**
   - Se un client fa reset, colpisce tutti (perché condiviso)
   - Questo è by design, ma richiede coordinamento

2. **Previously Trained Classes Non Persistente:**
   - Il set `previously_trained_classes` non è salvato nei checkpoint
   - Ogni esecuzione riparte con set vuoto
   - Soluzione futura: salvarlo nel checkpoint se necessario

3. **No Reset Parziale per Unified:**
   - In modalità `unified`, il reset è sempre totale
   - Non è possibile resettare "solo la parte per una classe"
   - Usare `per_class` se serve reset selettivo

---

## Autore e Data

- **Implementazione:** Claude Sonnet 4.5
- **Data:** 8 Gennaio 2026
- **File modificati:**
  - `system/flcore/servers/serverA2V.py`
  - `system/flcore/clients/clientA2V.py`
- **Documentazione:** Questo file + `GENERATOR_SHARING_AND_RESET.md`
