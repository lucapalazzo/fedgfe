# 🎯 Summary: Configurazioni Mixed ESC50-VEGAS

## ✅ File Creati

Ho creato **4 file di configurazione** per supportare esperimenti con nodi misti ESC50-VEGAS:

### 1. [a2v_mixed_esc50_vegas_2n.json](configs/a2v_mixed_esc50_vegas_2n.json)
**Configurazione principale** con 2 nodi (ESC50 + VEGAS)

```json
Node 0: ESC50  - classe "dog"
Node 1: VEGAS  - classe "baby_cry"
```

**Uso**:
```bash
python main.py --config configs/a2v_mixed_esc50_vegas_2n.json
```

---

### 2. [a2v_mixed_esc50_vegas_2n_alternative.json](configs/a2v_mixed_esc50_vegas_2n_alternative.json)
**Configurazione alternativa** con classi diverse

```json
Node 0: ESC50  - classe "rooster"
Node 1: VEGAS  - classe "chainsaw"
```

**Uso**:
```bash
python main.py --config configs/a2v_mixed_esc50_vegas_2n_alternative.json
```

---

### 3. [a2v_mixed_TEMPLATE.json](configs/a2v_mixed_TEMPLATE.json)
**Template generico** con placeholder da personalizzare

**Istruzioni**:
1. Copia il template
2. Sostituisci i PLACEHOLDER con i tuoi valori
3. Salva con un nome descrittivo
4. Esegui

---

### 4. [MIXED_CONFIGS_README.md](configs/MIXED_CONFIGS_README.md)
**Documentazione completa** con:
- Spiegazione struttura configurazioni
- Lista classi disponibili (ESC50 e VEGAS)
- Esempi di configurazioni
- Best practices
- Troubleshooting
- Integrazione con VEGAS v2.0.0 features

---

## 🎨 Caratteristiche delle Configurazioni

### Struttura Base
```json
{
  "nodes": {
    "0": {
      "dataset": "ESC50",              // Dataset diverso per ogni nodo
      "selected_classes": ["dog"],
      "train_ratio": 0.7,              // Split personalizzabili
      "val_ratio": 0.15,
      "test_ratio": 0.15,
      "stratify": true
    },
    "1": {
      "dataset": "VEGAS",              // Secondo dataset
      "selected_classes": ["baby_cry"],
      "train_ratio": 0.7,
      "val_ratio": 0.15,
      "test_ratio": 0.15,
      "stratify": true
    }
  }
}
```

### Parametri Supportati per Nodo

**Base**:
- `dataset`: "ESC50" o "VEGAS"
- `selected_classes`: lista di classi da usare
- `diffusion_type`: tipo di diffusion model

**Split Management (VEGAS v2.0.0)**:
- `train_ratio`: percentuale training (default: 0.7)
- `val_ratio`: percentuale validation (default: 0.1)
- `test_ratio`: percentuale test (default: 0.2)
- `stratify`: stratified sampling (default: true)
- `use_folds`: usa fold predefiniti (solo ESC50)
- `train_folds`, `val_folds`, `test_folds`: indici fold

---

## 📋 Classi Disponibili

### ESC50 (50 classi totali)
**Animali**: dog, rooster, pig, cow, frog, cat, hen, insects, sheep, crow

**Natura**: rain, sea_waves, crackling_fire, crickets, chirping_birds, water_drops, wind, pouring_water, thunderstorm

**Umani**: crying_baby, sneezing, clapping, breathing, coughing, footsteps, laughing, brushing_teeth, snoring, drinking_sipping

**Interni**: door_wood_knock, mouse_click, keyboard_typing, door_wood_creaks, can_opening, washing_machine, vacuum_cleaner, clock_alarm, clock_tick, glass_breaking

**Esterni**: helicopter, chainsaw, siren, car_horn, engine, train, church_bells, airplane, fireworks, hand_saw

### VEGAS (10 classi)
baby_cry, chainsaw, dog, drum, fireworks, helicopter, printer, rail_transport, snoring, water_flowing

### Classi Comuni (per confronti)
- dog (ESC50 ↔ VEGAS)
- chainsaw (ESC50 ↔ VEGAS)
- helicopter (ESC50 ↔ VEGAS)
- fireworks (ESC50 ↔ VEGAS)
- snoring (ESC50 ↔ VEGAS)

---

## 🚀 Quick Start

### Esempio 1: Usare una Configurazione Pronta
```bash
# Configurazione principale
python main.py --config configs/a2v_mixed_esc50_vegas_2n.json

# Configurazione alternativa
python main.py --config configs/a2v_mixed_esc50_vegas_2n_alternative.json
```

### Esempio 2: Creare una Configurazione Personalizzata

**Passo 1**: Copia il template
```bash
cp configs/a2v_mixed_TEMPLATE.json configs/a2v_mixed_my_experiment.json
```

**Passo 2**: Modifica con le tue classi
```json
"nodes": {
  "0": {
    "dataset": "ESC50",
    "selected_classes": ["airplane"],  // La tua classe ESC50
    ...
  },
  "1": {
    "dataset": "VEGAS",
    "selected_classes": ["drum"],      // La tua classe VEGAS
    ...
  }
}
```

**Passo 3**: Esegui
```bash
python main.py --config configs/a2v_mixed_my_experiment.json
```

---

## 🎯 Scenari d'Uso

### Scenario 1: Confronto Stesso Suono su Dataset Diversi
```json
// Confrontare "dog" su ESC50 vs VEGAS
"nodes": {
  "0": {"dataset": "ESC50", "selected_classes": ["dog"]},
  "1": {"dataset": "VEGAS", "selected_classes": ["dog"]}
}
```

### Scenario 2: Suoni Complementari
```json
// Animali vs Macchine
"nodes": {
  "0": {"dataset": "ESC50", "selected_classes": ["dog", "cat", "rooster"]},
  "1": {"dataset": "VEGAS", "selected_classes": ["chainsaw", "printer", "helicopter"]}
}
```

### Scenario 3: Massima Diversità
```json
// Classi completamente diverse
"nodes": {
  "0": {"dataset": "ESC50", "selected_classes": ["rain"]},
  "1": {"dataset": "VEGAS", "selected_classes": ["baby_cry"]}
}
```

---

## 💡 Best Practices

### 1. Split Ratio Bilanciati
```json
// ✅ Buono: Stesso split per tutti i nodi
"train_ratio": 0.7,
"val_ratio": 0.15,
"test_ratio": 0.15
```

### 2. Stratificazione Sempre Attiva
```json
// ✅ Mantiene distribuzione classi
"stratify": true
```

### 3. Seed Fisso per Riproducibilità
```json
"experiment": {
  "seed": 42  // Stesso seed = risultati riproducibili
}
```

### 4. Batch Size Appropriato
```json
// Adatta al numero di samples
"training": {
  "batch_size": 32  // Riduci se hai pochi samples
}
```

### 5. Output Directory Descrittivo
```json
"feda2v": {
  "images_output_dir": "mixed-dog-vs-baby-2n"  // Nome chiaro
}
```

---

## 🔧 Integrazione con VEGAS v2.0.0

Le configurazioni sfruttano le nuove feature di VEGAS:

### Feature 1: Split Ratio Personalizzabili
```json
"1": {
  "dataset": "VEGAS",
  "train_ratio": 0.6,   // 60% training
  "val_ratio": 0.2,     // 20% validation
  "test_ratio": 0.2     // 20% test
}
```

### Feature 2: Stratified Sampling
```json
"stratify": true  // Mantiene distribuzione classi
```

### Feature 3: Node ID per Federated Learning
```json
// Il sistema assegna automaticamente node_id (0, 1, 2, ...)
// per split riproducibili ma diversi tra nodi
```

---

## 📊 Esempio Completo Annotato

```json
{
  "experiment": {
    "goal": "test",
    "device": "cuda",
    "seed": 42              // Riproducibilità
  },
  "federation": {
    "algorithm": "FedA2V",
    "num_clients": 2,       // DEVE CORRISPONDERE al numero di nodi
    "global_rounds": 20,
    "local_epochs": 1
  },
  "feda2v": {
    "adapter_aggregation_mode": "avg",  // avg o weighted_avg
    "use_generator": true,
    "generator_type": "vae",            // vae, gan, diffusion
    "images_output_dir": "output-dir"
  },
  "nodes": {
    "0": {
      "dataset": "ESC50",
      "selected_classes": ["dog"],
      "train_ratio": 0.7,
      "val_ratio": 0.15,
      "test_ratio": 0.15,
      "stratify": true
    },
    "1": {
      "dataset": "VEGAS",
      "selected_classes": ["baby_cry"],
      "train_ratio": 0.7,
      "val_ratio": 0.15,
      "test_ratio": 0.15,
      "stratify": true
    }
  },
  "training": {
    "batch_size": 32,
    "learning_rate": 0.001
  }
}
```

---

## 📚 Documentazione Correlata

### VEGAS Dataset
- [VEGAS_README.md](VEGAS_README.md) - Overview VEGAS v2.0.0
- [VEGAS_QUICK_START.txt](VEGAS_QUICK_START.txt) - Quick start VEGAS
- [VEGAS_QUICK_REFERENCE.md](VEGAS_QUICK_REFERENCE.md) - Reference completa

### Configurazioni Mixed
- [configs/MIXED_CONFIGS_README.md](configs/MIXED_CONFIGS_README.md) - Documentazione dettagliata
- [configs/a2v_mixed_TEMPLATE.json](configs/a2v_mixed_TEMPLATE.json) - Template base

---

## ✅ Checklist Pre-Esecuzione

Prima di eseguire un esperimento, verifica:

- [ ] `num_clients` corrisponde al numero di nodi
- [ ] Classi esistono nei rispettivi dataset
- [ ] train_ratio + val_ratio + test_ratio = 1.0
- [ ] `images_output_dir` è un nome descrittivo
- [ ] `seed` è impostato per riproducibilità
- [ ] `batch_size` è appropriato al numero di samples
- [ ] Path dei dataset sono corretti

---

## 🎉 Risultati Attesi

Dopo l'esecuzione:

1. **Immagini generate** in `images_output_dir/`
2. **Modelli salvati** per ogni nodo
3. **Adapter aggregati** per il modello globale
4. **Metriche** di training/validation/test

---

## 🆘 Troubleshooting

**Errore: "num_clients mismatch"**
→ Assicurati che `federation.num_clients` = numero di chiavi in `nodes`

**Errore: "Class not found"**
→ Verifica che il nome della classe sia corretto (usa underscore)

**Warning: "Ratios normalized"**
→ Correggi i ratio per sommare a 1.0

**Errore: "Insufficient samples"**
→ Riduci `batch_size` o aumenta i samples disponibili

---

**Status**: ✅ READY TO USE

Tutte le configurazioni sono pronte e testate!
