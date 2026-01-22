# Sistema di Gestione Validation e Test Splits

## Panoramica

Il sistema ora gestisce correttamente i diversi split di dati per validation e test, separando i test locali dei client dai test globali del server.

## Architettura

### Split dei dati

Ogni nodo ha tre dataset:
- **train_dataset**: Usato per il training
- **validation_dataset**: Usato per i test locali dei client durante il training
- **test_dataset**: Usato per i test globali del server (aggregato da tutti i nodi)

### Workflow di Test

#### 1. Test Locali (Client)
- **Dataset usato**: `validation_dataset`
- **Metodo**: `client.local_test()`
- **Quando**: Durante il training, per monitorare la performance locale senza toccare il test set
- **Scope**: Ogni client valuta solo se stesso

```python
# Esempio di utilizzo
validation_metrics = client.local_test()
```

#### 2. Test Globali (Server)
- **Dataset usato**: `test_dataset` (aggregato da tutti i nodi)
- **Metodo**: `server.test_metrics()`
- **Quando**: Solo quando `train_adapters=True` (gli adapters vengono allenati)
- **Scope**:
  - Se `adapter_aggregation_mode == 'none'`: Ogni client testa solo sul proprio test set
  - Se `adapter_aggregation_mode != 'none'`: Cross-client testing (ogni client testa su tutti gli altri)

```python
# Esempio di utilizzo
test_stats = server.test_metrics()  # standalone auto-determinato dal config
test_stats = server.test_metrics(standalone=True)  # forza test solo su se stessi
test_stats = server.test_metrics(standalone=False)  # forza cross-client testing
```

## Configurazione

### Flag `train_adapters`

Nuova flag aggiunta nella sezione `feda2v` dei file di configurazione:

```json
{
  "feda2v": {
    "train_adapters": true,
    ...
  }
}
```

#### Valori possibili:
- `true` (default): Gli adapters vengono allenati
  - I client possono fare test locali su validation
  - Il server fa test globali su test

- `false`: Solo il generatore viene allenato (generator-only mode)
  - Nessun test viene eseguito
  - Usato per pre-training dei generatori

## Modalità di Training

### Modalità 1: Training Adapters (senza aggregazione)
```json
{
  "feda2v": {
    "train_adapters": true,
    "adapter_aggregation_mode": "none",
    "generator_training_mode": false
  }
}
```
- **Client**: Test locali su validation ✓
- **Server**: Test su test set di ogni nodo (standalone) ✓

### Modalità 2: Training Adapters (con aggregazione)
```json
{
  "feda2v": {
    "train_adapters": true,
    "adapter_aggregation_mode": "avg",
    "generator_training_mode": false
  }
}
```
- **Client**: Test locali su validation ✓
- **Server**: Test cross-client su test set aggregato ✓

### Modalità 3: Training Adapters + Generator
```json
{
  "feda2v": {
    "train_adapters": true,
    "generator_training_mode": false,
    "use_generator": true,
    "adapter_aggregation_mode": "avg"
  }
}
```
- **Client**: Test locali su validation ✓
- **Server**: Test cross-client su test set aggregato ✓

### Modalità 4: Training Solo Generator
```json
{
  "feda2v": {
    "train_adapters": false,
    "generator_training_mode": true
  }
}
```
- **Client**: Nessun test ✗
- **Server**: Nessun test ✗

## Implementazione

### Client ([clientA2V.py](system/flcore/clients/clientA2V.py))

Nuovo metodo `local_test()`:
```python
def local_test(self):
    """
    Perform local validation using the validation dataset.
    This is used for local monitoring during training without touching the test set.

    Returns:
        NodeMetric object with validation metrics
    """
```

### Server ([serverA2V.py](system/flcore/servers/serverA2V.py))

Modifiche a `test_metrics()` e `train_metrics()`:
```python
def test_metrics(self, standalone=None):
    """
    Calculate test metrics for Audio2Visual clients using test dataset.
    Test is performed only when adapters are being trained.

    Args:
        standalone: If None (default), automatically determined based on adapter_aggregation_mode
    """
    # Skip test if adapters are not being trained
    if not self.train_adapters:
        print("[Server] Skipping test_metrics: adapters not being trained (train_adapters=False)")
        return {}

    # Auto-determine standalone mode based on aggregation configuration
    if standalone is None:
        standalone = (self.adapter_aggregation_mode == 'none')
```

## File di Configurazione Aggiornati

I seguenti file di configurazione sono stati aggiornati con la flag `train_adapters`:

1. [a2v_esc50_1n_10c.json](configs/a2v_esc50_1n_10c.json) - `train_adapters: true`
2. [a2v_esc50_1n_1c_generate.json](configs/a2v_esc50_1n_1c_generate.json) - `train_adapters: true`
3. [a2v_esc50_3n_1c_generate_avg_test.json](configs/a2v_esc50_3n_1c_generate_avg_test.json) - `train_adapters: true`
4. [a2v_generator_training_mode.json](configs/a2v_generator_training_mode.json) - `train_adapters: false`

## Benefici

1. **Separazione chiara**: Validation per test locali, test per test globali
2. **Protezione del test set**: Il test set viene usato solo per valutazioni finali
3. **Controllo granulare**: La flag `train_adapters` permette di controllare quando fare test
4. **Efficienza**: Nessun overhead di test quando si allena solo il generatore

## Note Importanti

- Il metodo `local_test()` può essere chiamato manualmente durante il training per monitorare la performance locale
- Il server continua a usare la logica esistente per i test cross-client
- La flag `train_adapters` è retrocompatibile (default: `true`)
- Quando `train_adapters=false`, sia `test_metrics()` che `train_metrics()` del server vengono skippati
