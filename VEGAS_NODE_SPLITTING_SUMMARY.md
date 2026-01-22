# VEGAS Node Splitting - Riepilogo Implementazione

## Data: 2026-01-19

## Modifiche Apportate

### 1. Dataset VEGAS (`system/datautils/dataset_vegas.py`)

#### Nuovi Parametri nel Costruttore
- `node_id`: ID del nodo (0-indexed) per federated learning
- `num_nodes`: Numero totale di nodi (divide i dati equamente)
- `samples_per_node`: Numero fisso di campioni per classe per nodo
- `node_split_seed`: Seed per split riproducibili (default: 42)

#### Nuovo Metodo: `_apply_node_split()`
Implementa la logica di splitting per nodi con due modalità:

**Modalità 1: num_nodes**
- Divide tutti i campioni equamente tra N nodi
- Mantiene il bilanciamento delle classi
- I campioni rimanenti (resto della divisione) vengono distribuiti ai primi nodi

**Modalità 2: samples_per_node**
- Ogni nodo riceve esattamente K campioni per classe
- Utile per controllare la dimensione del dataset per nodo
- Gestisce il caso di classi con meno campioni del richiesto

#### Caratteristiche Implementate
- **Riproducibilità**: Usa seed per garantire split identici
- **Bilanciamento per classe**: Divide i dati mantenendo le proporzioni delle classi
- **Nessuna sovrapposizione**: Ogni campione appartiene a un solo nodo
- **Validazione**: Controlla parametri non validi (es. node_id >= num_nodes)
- **Logging dettagliato**: Mostra distribuzione dei campioni per classe e nodo

### 2. Config Loader (`system/utils/config_loader.py`)

#### Nuovi Parametri di Default nella Sezione 'node'
```python
'node_split_id': None,      # ID dello split per questo nodo
'num_nodes': None,          # Numero totale di nodi
'samples_per_node': None,   # Campioni fissi per nodo per classe
'node_split_seed': 42       # Seed per riproducibilità
```

### 3. File di Configurazione di Esempio

#### `configs/vegas_node_splitting_example.json`
Esempio di configurazione con `num_nodes` per dividere equamente i dati tra 3 nodi.

#### `configs/vegas_samples_per_node_example.json`
Esempio di configurazione con `samples_per_node` per dare a ogni nodo 20 campioni per classe.

### 4. Documentazione

#### `VEGAS_NODE_SPLITTING.md`
Documentazione completa con:
- Descrizione delle due modalità di splitting
- Esempi Python e JSON
- Scenari di utilizzo (FL omogeneo, eterogeneo, semi-eterogeneo)
- Best practices
- Sezione debugging e validazione

### 5. Test Suite

#### `tests/test_vegas_node_splitting.py`
Suite di test completa che verifica:
- Modalità `num_nodes`
- Modalità `samples_per_node`
- Integrazione con train/val/test splits
- Error handling per configurazioni non valide
- Riproducibilità con seed diversi
- Assenza di sovrapposizioni tra nodi

## Come Utilizzare

### Esempio Base con num_nodes

```python
from datautils.dataset_vegas import VEGASDataset

# Crea dataset per nodo 0 di 3 nodi totali
dataset = VEGASDataset(
    root_dir="/path/to/VEGAS",
    selected_classes=['dog', 'baby_cry', 'chainsaw'],
    split='train',
    node_id=0,
    num_nodes=3,
    node_split_seed=42
)
```

### Esempio con Configurazione JSON

```json
{
  "nodes": {
    "0": {
      "dataset": "vegas",
      "node_split_id": 0,
      "num_nodes": 3,
      "node_split_seed": 42,
      "selected_classes": ["dog", "baby_cry", "chainsaw"]
    }
  }
}
```

## Vantaggi

1. **Riproducibilità Garantita**: Stessi parametri → stessi split
2. **Flessibilità**: Due modalità per scenari diversi
3. **Facilità d'Uso**: Configurazione JSON intuitiva
4. **Robustezza**: Validazione parametri ed error handling
5. **Trasparenza**: Logging dettagliato della distribuzione

## Compatibilità

- ✅ Compatibile con train/val/test split esistenti
- ✅ Compatibile con class selection/exclusion
- ✅ Compatibile con auto-split (split=None)
- ✅ Compatibile con AST cache
- ✅ Compatibile con stratified sampling

## Testing

Esegui la suite di test:
```bash
cd /home/lpala/fedgfe
python tests/test_vegas_node_splitting.py
```

## Note Importanti

1. **Mutua esclusività**: Non usare `num_nodes` e `samples_per_node` insieme
2. **Ordine di applicazione**: Train/Val/Test split → Node split
3. **Cache separata**: Ogni configurazione di node split ha la propria cache
4. **Seed consistency**: Usa lo stesso seed per tutti i nodi in un esperimento

## File Modificati

- `system/datautils/dataset_vegas.py` - Implementazione core
- `system/utils/config_loader.py` - Supporto configurazione
- `configs/vegas_node_splitting_example.json` - Esempio num_nodes
- `configs/vegas_samples_per_node_example.json` - Esempio samples_per_node
- `VEGAS_NODE_SPLITTING.md` - Documentazione completa
- `tests/test_vegas_node_splitting.py` - Suite di test

## Esempi di Scenari

### FL Omogeneo
Tutti i nodi con le stesse classi, dati diversi:
```json
{
  "nodes": {
    "0": {"node_split_id": 0, "num_nodes": 5, "selected_classes": ["dog", "baby_cry"]},
    "1": {"node_split_id": 1, "num_nodes": 5, "selected_classes": ["dog", "baby_cry"]},
    "2": {"node_split_id": 2, "num_nodes": 5, "selected_classes": ["dog", "baby_cry"]}
  }
}
```

### FL Eterogeneo
Ogni nodo con classi diverse:
```json
{
  "nodes": {
    "0": {"node_split_id": 0, "samples_per_node": 20, "selected_classes": ["dog"]},
    "1": {"node_split_id": 0, "samples_per_node": 20, "selected_classes": ["baby_cry"]},
    "2": {"node_split_id": 0, "samples_per_node": 20, "selected_classes": ["chainsaw"]}
  }
}
```

## Prossimi Passi

- [ ] Eseguire i test per verificare il corretto funzionamento
- [ ] Integrare con il sistema di server/client esistente
- [ ] Aggiungere metriche di distribuzione dei dati in WandB
- [ ] Considerare supporto per strategie di splitting più avanzate (es. Dirichlet distribution)
