# VEGAS Single Node Training Configuration

## 📄 File: `a2v_vegas_1n_all_classes.json`

Configurazione per allenare un singolo nodo con **tutto il dataset VEGAS** (tutte le classi).

---

## 🎯 Caratteristiche Principali

### Dataset
- **Dataset**: VEGAS completo
- **Classi**: Tutte (nessun filtro `selected_classes`)
- **Split Ratio**: 70% train, 15% validation, 15% test
- **Stratificazione**: Abilitata per bilanciamento classi

### Federazione
- **Numero Nodi**: 1 (training centralizzato)
- **Round Globali**: 50
- **Epoche Locali**: 5
- **Join Ratio**: 100%
- **Eval Gap**: Ogni 5 round

### Training
- **Optimizer**: AdamW
- **Learning Rate**: 0.0001
- **Batch Size**: 16
- **Weight Decay**: 0.0001
- **LR Schedule**: Disabilitato

### Memoria e Performance
- **Optimize Memory**: ✅ Abilitato (include fix memory leak)
- **Memory Leak Fix**: Attivo per training lunghi
- **Adapter Aggregation**: Media (avg)

### Generazione Immagini
- **Frequenza Generazione Nodi**: Ogni 10 round
- **Frequenza Generazione Globale**: Disabilitata
- **Split per Generazione**: train
- **Split Salvati**: val, test
- **Output Dir**: `vegas_1n_all_classes_output/`

### Logging
- **WandB**: ✅ Abilitato
- **Project**: federated-a2v
- **Run Name**: vegas-1n-all-classes

---

## 🚀 Come Eseguire

### 1. Comando Base
```bash
python main.py -c configs/a2v_vegas_1n_all_classes.json
```

### 2. Con Device Specifico
```bash
CUDA_VISIBLE_DEVICES=0 python main.py -c configs/a2v_vegas_1n_all_classes.json
```

### 3. Con WandB Disabilitato
Modifica il file JSON:
```json
"wandb": {
  "disabled": true
}
```

---

## 📊 Metriche Monitorate

### Durante Training
- **Train Metrics**: Computate su split `train`
- **Test Metrics**: Computate su split `val` e `test`
- **Frequenza Valutazione**: Ogni 5 round

### Metriche Specifiche
- Accuracy
- Loss
- Per-class accuracy (se disponibile)

---

## 💾 Output Generati

### Checkpoints
- Salvati in `checkpoints/VEGAS-1N-AllClasses-Training/`
- Frequenza: Configurabile

### Immagini Generate
- Directory: `vegas_1n_all_classes_output/`
- Split salvati: validation e test
- Generazione ogni 10 round

### Logs
- WandB dashboard (se abilitato)
- File di log locali

---

## ⚙️ Parametri Modificabili

### Per Training Più Veloce
```json
{
  "federation": {
    "global_rounds": 20,
    "local_epochs": 2
  },
  "training": {
    "batch_size": 32
  }
}
```

### Per Training Più Accurato
```json
{
  "federation": {
    "global_rounds": 100,
    "local_epochs": 10
  },
  "training": {
    "learning_rate": 0.00005,
    "batch_size": 8
  }
}
```

### Per Debugging
```json
{
  "federation": {
    "global_rounds": 2,
    "local_epochs": 1
  },
  "feda2v": {
    "generate_nodes_images_frequency": 1
  },
  "wandb": {
    "disabled": true
  }
}
```

---

## 🔧 Troubleshooting

### Out of Memory (OOM)
1. Riduci `batch_size` (es. 8 o 4)
2. Verifica che `optimize_memory_usage: true`
3. Monitora memoria GPU: `nvidia-smi -l 1`

### Training Lento
1. Aumenta `batch_size` se la memoria lo permette
2. Riduci `local_epochs`
3. Verifica che CUDA sia disponibile

### WandB Issues
1. Login: `wandb login`
2. Disabilita se non necessario: `"disabled": true`
3. Verifica connessione internet

---

## 📝 Note Importanti

### Memory Leak Fixes
Questa configurazione include i fix per memory leak implementati:
- ✅ Pulizia cache embeddings tra round
- ✅ Pulizia audio embedding store
- ✅ Pulizia output aggregati server
- ✅ CUDA cache emptying automatico

### Differenze con Multi-Node
- Non c'è aggregazione federata (un solo nodo)
- Equivalente a training centralizzato
- Utile per baseline e debugging

### Dataset Path
Assicurati che il dataset VEGAS sia in:
- `./data/VEGAS/` o
- Modifica `dataset.path` nel JSON

---

## 📚 Riferimenti

- **VEGAS Dataset**: [VEGAS_README.md](../VEGAS_README.md)
- **ESC50 Features**: [VEGAS_ESC50_FEATURES.md](../VEGAS_ESC50_FEATURES.md)
- **Memory Leak Fixes**: Vedi commit recenti
- **Configurazioni Simili**:
  - `a2v_1n_vegas_store_embeddings.json` (1 round, store embeddings)
  - `a2v_vegas_10n.json` (10 nodi, classi separate)

---

## ✅ Checklist Pre-Esecuzione

- [ ] Dataset VEGAS scaricato e in `./data/`
- [ ] CUDA disponibile: `nvidia-smi`
- [ ] Ambiente conda attivato
- [ ] WandB configurato (se abilitato)
- [ ] Spazio disco sufficiente per output
- [ ] Memoria GPU sufficiente (consigliato: ≥16GB)

---

**Creato**: 2025-12-14
**Versione**: 1.0
**Compatibile con**: FedA2V con memory leak fixes
