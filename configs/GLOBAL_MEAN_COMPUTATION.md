# Global Mean Computation from Class Means

## Overview

Questa feature controlla se il server deve calcolare una media globale unificata a partire dalle medie per-classe aggregate dai nodi.

## Parametro di Configurazione

### `compute_global_mean_from_class_means`

**Posizione**: `feda2v.compute_global_mean_from_class_means`
**Tipo**: `boolean`
**Default**: `true`
**Quando usarlo**: Durante l'aggregazione dei parametri con modalità `per_class_average`

## Comportamento

### Quando `true` (default)

Il server calcola una media globale unica facendo la media di tutte le medie per-classe:

1. Raccoglie le medie per-classe da tutti i nodi
2. Aggrega le medie per ogni classe (somma e divisione per il numero di nodi)
3. **Calcola una media globale** facendo lo stack di tutte le medie per-classe e calcolandone la media

**Risultato**:
- `self.global_per_class_output_means`: Dict con medie per ogni classe
- `self.global_output_means`: Tensor con la media globale unificata

**Output di log**:
```
[Server] Computed global mean from 15 class means
[Server] Global mean shape: torch.Size([512])
```

### Quando `false`

Il server NON calcola la media globale:

1. Raccoglie le medie per-classe da tutti i nodi
2. Aggrega le medie per ogni classe
3. **NON calcola la media globale**

**Risultato**:
- `self.global_per_class_output_means`: Dict con medie per ogni classe
- `self.global_output_means`: Rimane `None` o al valore precedente

## Esempio di Configurazione

```json
{
  "feda2v": {
    "model_aggregation": "per_class_average",
    "compute_global_mean_from_class_means": true
  }
}
```

## Quando Usare `false`

Disabilitare il calcolo della media globale può essere utile in questi scenari:

### 1. Specializzazione per-classe
Se vuoi mantenere rappresentazioni separate per ogni classe senza "mescolarle" in una media globale:

```json
{
  "feda2v": {
    "compute_global_mean_from_class_means": false
  }
}
```

**Caso d'uso**: Training con classi molto diverse semanticamente dove una media globale potrebbe perdere informazioni specifiche.

### 2. Analisi delle differenze per-classe
Quando vuoi analizzare le differenze tra le medie delle varie classi senza l'influenza di una media globale:

**Caso d'uso**: Debugging, analisi della distribuzione delle features, valutazione della separabilità delle classi.

### 3. Risparmio computazionale
Su dataset con molte classi, il calcolo della media globale può essere evitato se non necessario:

**Caso d'uso**: Federazioni con centinaia di classi dove la media globale non viene utilizzata.

## Implementazione

### File modificato
- [system/flcore/servers/serverA2V.py](../system/flcore/servers/serverA2V.py)

### Metodo principale
```python
def _compute_global_mean_from_class_means(self, global_per_class_output_means):
    """
    Compute a unified global mean by averaging all per-class means.

    Args:
        global_per_class_output_means: Dict mapping class names to mean tensors

    Sets:
        self.global_output_means: The computed global mean tensor
    """
```

### Logica di esecuzione
Linea ~1587 in `aggregate_audio_encoder_parameters_per_class_avarage()`:
```python
if self.compute_global_mean_from_class_means:
    self._compute_global_mean_from_class_means(global_per_class_output_means)
```

## Vantaggi dell'Estrazione

1. **Separazione delle responsabilità**: Il calcolo della media globale è ora isolato in un metodo dedicato
2. **Configurabilità**: Può essere facilmente disabilitato/abilitato via JSON
3. **Testabilità**: Il metodo può essere testato indipendentemente
4. **Manutenibilità**: Più facile da modificare o estendere in futuro
5. **Debug**: Logging dettagliato incluso nel metodo

## Note Tecniche

- Il metodo gestisce automaticamente casi edge (nessuna classe, dict vuoto)
- Include validazione e logging per debugging
- Opera su tensori CPU (coerente con il resto dell'aggregazione)
- La media è calcolata lungo la dimensione 0 (classe) dello stack di tensori

## Esempio Output

Con 15 classi e embedding dimension 512:

```python
global_per_class_output_means = {
    'dog': torch.Size([512]),
    'cat': torch.Size([512]),
    ...  # 13 altre classi
}

# Dopo _compute_global_mean_from_class_means:
self.global_output_means  # torch.Size([512])
```
