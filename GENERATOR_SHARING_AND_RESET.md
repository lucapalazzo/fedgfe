# Generator Sharing and Reset Feature

## Panoramica

Questa implementazione aggiunge due nuove funzionalità per la gestione dei generatori in modalità `generator_only_mode`:

1. **Condivisione del generatore tra nodi**: Permette a tutti i nodi di utilizzare la stessa istanza del generatore, consentendo un addestramento progressivo attraverso i nodi.
2. **Reset dei parametri del generatore**: Permette di resettare i parametri del generatore quando si addestra su nuove classi.

## Nuovi Parametri di Configurazione

### `shared_generator_in_only_mode`
- **Tipo**: `boolean`
- **Default**: `true`
- **Descrizione**: Se `true`, tutti i nodi in modalità `generator_only_mode` condividono la stessa istanza del generatore. Questo permette ai nodi di contribuire progressivamente all'addestramento dello stesso generatore, senza creare copie separate.
- **Quando usarlo**:
  - Quando si vuole un generatore condiviso che apprende da tutti i nodi
  - Per ridurre l'uso di memoria (un solo generatore invece di N generatori)
  - Per avere un training più consistente attraverso i nodi

### `reset_generator_on_class_change`
- **Tipo**: `boolean`
- **Default**: `false`
- **Descrizione**: Se `true`, resetta i parametri del generatore quando vengono rilevate nuove classi durante l'addestramento. Utile per scenari di addestramento sequenziale delle classi.
- **Quando usarlo**:
  - In scenari di class-incremental learning
  - Quando si vogliono generatori "freschi" per ogni nuova classe
  - Per evitare interferenze tra classi in training sequenziale
- **Quando NON usarlo**:
  - Se si vuole un addestramento incrementale continuo
  - Se si vuole che il generatore mantenga la conoscenza delle classi precedenti

## Comportamento per Granularità

### Unified Generator (`generator_granularity: "unified"`)
- **Shared mode**: Un singolo generatore condiviso tra tutti i nodi
- **Reset mode**: Il generatore viene resettato quando appare QUALSIASI nuova classe

### Per-Class Generator (`generator_granularity: "per_class"`)
- **Shared mode**: Ogni classe ha il suo generatore, condiviso tra i nodi
- **Reset mode**: Solo i generatori delle NUOVE classi vengono resettati

### Per-Group Generator (`generator_granularity: "per_group"`)
- **Shared mode**: Ogni gruppo ha il suo generatore, condiviso tra i nodi
- **Reset mode**: Solo i generatori dei gruppi che contengono nuove classi vengono resettati

## Esempi di Configurazione

### Esempio 1: Generatore Condiviso Senza Reset (Default)
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
**Risultato**: Tutti i nodi usano lo stesso generatore e continuano ad addestrarlo progressivamente su tutte le classi.

### Esempio 2: Generatore Condiviso Con Reset per Nuove Classi
```json
{
  "feda2v": {
    "generator_only_mode": true,
    "generator_granularity": "unified",
    "shared_generator_in_only_mode": true,
    "reset_generator_on_class_change": true
  }
}
```
**Risultato**: Tutti i nodi usano lo stesso generatore, ma viene resettato quando appare una nuova classe.

### Esempio 3: Generatori Per-Class Condivisi Con Reset Selettivo
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
**Risultato**: Ogni classe ha il suo generatore condiviso tra i nodi. Solo i generatori delle nuove classi vengono resettati.

### Esempio 4: Generatori Indipendenti Per Nodo (Comportamento Legacy)
```json
{
  "feda2v": {
    "generator_only_mode": true,
    "generator_granularity": "unified",
    "shared_generator_in_only_mode": false,
    "reset_generator_on_class_change": false
  }
}
```
**Risultato**: Ogni nodo crea e addestra il proprio generatore indipendentemente.

## File di Configurazione di Esempio

### `configs/a2v_generator_shared_with_reset.json`
Configurazione completa per generatore condiviso senza reset, ideale per training continuo.

### `configs/a2v_generator_sequential_classes_with_reset.json`
Configurazione per scenari di class-incremental learning con reset automatico.

## Implementazione Tecnica

### Nuove Funzioni

#### `reset_generator_parameters(generator_key=None)`
Resetta i parametri di un generatore allo stato iniziale.
- **Args**: `generator_key` - Se specificato, resetta solo quel generatore (per modalità per_class/per_group)
- **Comportamento**: Ricrea il generatore con pesi inizializzati casualmente e un nuovo optimizer

#### Modifiche a `initialize_generators()`
- Controlla se `shared_generator_in_only_mode` è attivo
- Se sì, cerca di riutilizzare i generatori dal global_model
- Se non trova generatori da condividere, crea nuovi generatori

#### Modifiche a `train_node_generator()`
- Traccia le classi precedentemente viste in `self.previously_trained_classes`
- Se `reset_generator_on_class_change` è attivo, confronta le classi correnti con quelle precedenti
- Chiama `reset_generator_parameters()` per le nuove classi secondo la granularità

## Tracciamento dello Stato

### `self.previously_trained_classes`
- **Tipo**: `set`
- **Contenuto**: Set di nomi di classi che sono state addestrate in precedenza
- **Utilizzo**: Permette di rilevare quando appaiono nuove classi durante il training

## Flusso di Esecuzione

```
1. Inizializzazione del Client
   ├─> __init__() legge i parametri shared_generator_in_only_mode e reset_generator_on_class_change
   └─> initialize_generators()
       ├─> Se shared_generator_in_only_mode e generator_only_mode:
       │   └─> Cerca di riutilizzare generatori dal global_model
       └─> Altrimenti: crea nuovi generatori

2. Training
   └─> train_node_generator()
       ├─> Raccoglie embeddings
       ├─> Identifica classi da addestrare
       ├─> Se reset_generator_on_class_change:
       │   ├─> Confronta con previously_trained_classes
       │   ├─> Identifica nuove classi
       │   └─> Chiama reset_generator_parameters() per nuove classi
       ├─> Addestra il/i generatore/i
       └─> Aggiorna previously_trained_classes
```

## Note Importanti

1. **Memoria**: La condivisione del generatore riduce significativamente l'uso di memoria quando si hanno molti nodi.

2. **Consistenza**: Con `shared_generator_in_only_mode=true`, tutti i nodi vedono gli stessi pesi del generatore, garantendo consistenza.

3. **Class-Incremental Learning**: Il reset è utile per evitare catastrophic forgetting in scenari dove le classi arrivano sequenzialmente.

4. **Checkpointing**: I checkpoint salvano lo stato del generatore, inclusi i pesi e l'optimizer. Con la condivisione attiva, un checkpoint rappresenta lo stato condiviso da tutti i nodi.

5. **Compatibilità**: Le nuove funzionalità sono retrocompatibili. Se i parametri non sono specificati, il comportamento di default mantiene la logica precedente.

## Debugging

Per verificare il comportamento:

```python
# Nel client, durante initialize_generators()
print(f"[Client {self.id}] shared_generator_in_only_mode: {self.shared_generator_in_only_mode}")
print(f"[Client {self.id}] reset_generator_on_class_change: {self.reset_generator_on_class_change}")

# Durante train_node_generator()
print(f"[Client {self.id}] Previously trained classes: {self.previously_trained_classes}")
print(f"[Client {self.id}] Current classes: {classes_to_train}")
print(f"[Client {self.id}] New classes detected: {new_classes}")
```

## Testing

Per testare le nuove funzionalità:

1. **Test condivisione senza reset**:
   ```bash
   python main.py --config configs/a2v_generator_shared_with_reset.json
   ```

2. **Test condivisione con reset**:
   ```bash
   python main.py --config configs/a2v_generator_sequential_classes_with_reset.json
   ```

3. Verificare nei log:
   - "Reusing shared unified generator from global model" (per la condivisione)
   - "Detected new classes: [...]" (per il rilevamento di nuove classi)
   - "Resetting generator parameters" (per il reset)
