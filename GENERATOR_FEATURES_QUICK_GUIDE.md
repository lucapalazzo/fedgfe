# Generator Features - Quick Start Guide

## 🚀 Nuove Funzionalità Implementate

### 1. ✅ Condivisione Generatori tra Client
I client condividono la stessa istanza del generatore (non copie separate)

### 2. ✅ Reset Automatico su Nuove Classi
I generatori possono essere resettati quando appaiono nuove classi

### 3. ✅ Ottimizzazione Memoria
Il server non carica il diffusion model quando in modalità `generator_only`

---

## 📋 Configurazioni Disponibili

### Scenario 1: Training Continuo (NO Reset)
**Usa quando:** Vuoi che il generatore impari progressivamente da tutte le classi

```json
{
  "feda2v": {
    "generator_only_mode": true,
    "shared_generator_in_only_mode": true,
    "reset_generator_on_class_change": false,
    "generator_granularity": "unified"
  }
}
```

**File esempio:** `configs/a2v_generator_shared_with_reset.json`

**Comando:**
```bash
python main.py --config configs/a2v_generator_shared_with_reset.json
```

---

### Scenario 2: Class-Incremental Learning (CON Reset)
**Usa quando:** Ogni classe deve avere un generatore "fresco" senza interferenze da classi precedenti

```json
{
  "feda2v": {
    "generator_only_mode": true,
    "shared_generator_in_only_mode": true,
    "reset_generator_on_class_change": true,
    "generator_granularity": "per_class"
  }
}
```

**File esempio:** `configs/a2v_generator_sequential_classes_with_reset.json`

**Comando:**
```bash
python main.py --config configs/a2v_generator_sequential_classes_with_reset.json
```

---

## 🎯 Parametri Chiave

| Parametro | Valori | Default | Descrizione |
|-----------|--------|---------|-------------|
| `generator_only_mode` | `true/false` | `false` | Allena solo generatori, non adapters |
| `shared_generator_in_only_mode` | `true/false` | `true` | Condividi generatori tra client |
| `reset_generator_on_class_change` | `true/false` | `false` | Reset su nuove classi |
| `generator_granularity` | `unified/per_class/per_group` | `unified` | Strategia di creazione generatori |

---

## 📊 Comportamento per Granularità

### `unified` (1 generatore per tutto)
- ✅ Minimo uso memoria
- ✅ Impara da tutte le classi insieme
- ⚠️ Reset = ripartenza da zero completa

### `per_class` (1 generatore per classe)
- ✅ Ogni classe indipendente
- ✅ Reset selettivo solo su nuove classi
- ⚠️ Più memoria richiesta

### `per_group` (1 generatore per gruppo)
- ✅ Bilanciamento tra unified e per_class
- ✅ Reset per gruppi di classi
- ⚠️ Richiede definizione gruppi in config

---

## 🔍 Come Verificare che Funziona

### Log Attesi - Condivisione Attiva:
```
[Client 0]: Receiving 5 SHARED generators from server (shared_generator_in_only_mode=True)
[Client 0]: Generator classes available: ['dog', 'cat', 'bird', 'frog', 'pig']
```

### Log Attesi - Reset Attivo:
```
[Client 0] Detected new classes: ['bird', 'frog']
[Client 0] Previously trained classes: ['dog', 'cat']
[Client 0] Resetting generators for new classes only
[Client 0]   Resetting generator for class 'bird'
[Client 0]   Created fresh conditioned VAE generator for 'bird'
[Client 0] ✓ Generator reset complete
```

### Log Attesi - Diffusion Model NON Caricato:
```
# NON DEVE apparire questo messaggio quando generator_only_mode=true:
Started diffusion model flux  # ❌ Non deve comparire!

# Invece dovrebbe saltare il caricamento completamente
```

---

## 💾 Checkpoint

I checkpoint vengono salvati automaticamente:

```
checkpoints/generators_shared/
├── vae_unified_shared_node0_round_5.pt
├── vae_unified_shared_node0_round_10.pt
└── vae_unified_shared_node0_round_15.pt
```

Contenuto checkpoint:
- ✅ Pesi del generatore
- ✅ Stato optimizer
- ✅ Metadata (round, granularity, tipo, ecc.)
- ❌ `previously_trained_classes` (non salvato)

---

## 🐛 Troubleshooting

### Problema: Client non riceve generatori condivisi

**Soluzione:** Verifica che:
```json
{
  "feda2v": {
    "generator_only_mode": true,        // Deve essere true
    "shared_generator_in_only_mode": true  // Deve essere true
  }
}
```

### Problema: Reset non funziona

**Soluzione:** Verifica che:
```json
{
  "feda2v": {
    "reset_generator_on_class_change": true,  // Deve essere true
    "generator_granularity": "per_class"      // O "per_group", NON "unified" se vuoi reset selettivo
  }
}
```

### Problema: Diffusion model ancora caricato in generator_only_mode

**Soluzione:** Aggiorna il codice server (dovrebbe essere già fixato nella nuova versione)

---

## 📚 Documentazione Completa

- **Implementation Details:** `GENERATOR_RESET_IMPLEMENTATION.md`
- **Original Spec:** `GENERATOR_SHARING_AND_RESET.md`
- **Config Examples:** `configs/a2v_generator_*.json`

---

## ⚡ Quick Commands

```bash
# Test condivisione SENZA reset (continuous learning)
python main.py --config configs/a2v_generator_shared_with_reset.json

# Test condivisione CON reset (class-incremental)
python main.py --config configs/a2v_generator_sequential_classes_with_reset.json

# Custom config con reset
python main.py --config my_config.json \
  --feda2v.generator_only_mode=true \
  --feda2v.reset_generator_on_class_change=true \
  --feda2v.generator_granularity=per_class
```

---

## 🎓 Best Practices

1. **Usa `unified` + NO reset** per:
   - Training veloce
   - Classi simili tra loro
   - Quando la memoria è limitata

2. **Usa `per_class` + reset** per:
   - Class-incremental learning
   - Classi molto diverse
   - Prevenire catastrophic forgetting

3. **Usa `per_group` + reset** per:
   - Bilanciare performance e memoria
   - Classi raggruppate semanticamente
   - Scenari ibridi

---

## 🔧 Advanced: Custom Reset Logic

Se vuoi controllare manualmente il reset:

```python
# Nel client
if some_condition:
    # Reset generatore specifico
    self.reset_generator_parameters(generator_key="dog")

    # Reset generatore unified
    self.reset_generator_parameters()
```

---

**Data implementazione:** 8 Gennaio 2026
**Versione:** 1.0
**Status:** ✅ Production Ready
