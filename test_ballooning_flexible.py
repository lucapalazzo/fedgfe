#!/usr/bin/env python3
"""
Test completo delle nuove funzionalità di allocazione flessibile.

Dimostra tutti i metodi di specifica memoria supportati.
"""

import sys
sys.path.insert(0, '/home/lpala/fedgfe')

import torch
import logging
from system.utils.ballooning import (
    GPUMemoryBalloon,
    parse_memory_spec,
    reserve_all_gpus_memory
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

print("=" * 70)
print("TEST COMPLETO - GPU MEMORY BALLOONING CON ALLOCAZIONE FLESSIBILE")
print("=" * 70)
print()

# Test 1: Formati di specifica memoria
print("=" * 70)
print("TEST 1: Tutti i formati di specifica memoria supportati")
print("=" * 70)

formats = [
    (2000, "int: 2000 MB"),
    ("2GB", "str: 2GB"),
    ("2.5GB", "str: 2.5GB"),
    ("2048MB", "str: 2048MB"),
    ("50%", "str: 50% (basato su 10000 MB)"),
    (0.7, "float: 0.7 (70%)"),
]

for spec, description in formats:
    if isinstance(spec, str) and '%' in spec:
        result = parse_memory_spec(spec, free_memory_mb=10000)
    elif isinstance(spec, float):
        result = parse_memory_spec(spec, free_memory_mb=10000)
    else:
        result = parse_memory_spec(spec)
    print(f"  {description:<30} -> {result:>6} MB")
print()

# Test 2: allocate_memory() con vari formati
print("=" * 70)
print("TEST 2: allocate_memory() con formati diversi")
print("=" * 70)

balloon = GPUMemoryBalloon(gpu_id=0, reserve_mb=1000)

test_cases = [
    ("2GB", "Alloca 2 GB"),
    (2048, "Alloca 2048 MB"),
    ("1024MB", "Alloca 1024 MB"),
]

for amount, description in test_cases:
    balloon.allocate_memory(amount, verbose=False)
    allocated = balloon.get_status()['allocated_mb']
    print(f"  {description:<25} -> Allocato: {allocated} MB")
    balloon.deflate(verbose=False)
print()

# Test 3: Allocazione incrementale
print("=" * 70)
print("TEST 3: Allocazione incrementale con allocate_additional()")
print("=" * 70)

balloon = GPUMemoryBalloon(gpu_id=0, reserve_mb=1000)

# Allocazione iniziale
balloon.allocate_memory("2GB", verbose=False)
print(f"  Allocazione iniziale (2GB):      {balloon.get_status()['allocated_mb']} MB")

# Aggiungi 1GB
balloon.allocate_additional("1GB", verbose=False)
print(f"  Dopo +1GB:                       {balloon.get_status()['allocated_mb']} MB")

# Aggiungi 512MB
balloon.allocate_additional(512, verbose=False)
print(f"  Dopo +512MB:                     {balloon.get_status()['allocated_mb']} MB")

# Aggiungi 10% della memoria libera residua
balloon.allocate_additional("10%", verbose=False)
print(f"  Dopo +10% (memoria residua):     {balloon.get_status()['allocated_mb']} MB")

balloon.deflate(verbose=False)
print()

# Test 4: Allocazione a target
print("=" * 70)
print("TEST 4: Allocazione a target con allocate_to_target()")
print("=" * 70)

balloon = GPUMemoryBalloon(gpu_id=0, reserve_mb=1000)

# Inizia con 2GB
balloon.allocate_memory("2GB", verbose=False)
print(f"  Allocazione iniziale (2GB):      {balloon.get_status()['allocated_mb']} MB")

# Aumenta a 5GB
balloon.allocate_to_target("5GB", verbose=False)
print(f"  Target 5GB (aumenta):            {balloon.get_status()['allocated_mb']} MB")

# Diminuisci a 3GB
balloon.allocate_to_target("3GB", verbose=False)
print(f"  Target 3GB (diminuisce):         {balloon.get_status()['allocated_mb']} MB")

# Usa percentuale come target
balloon.allocate_to_target("20%", verbose=False)
target_mb = balloon.get_status()['allocated_mb']
print(f"  Target 20% (della mem. libera):  {target_mb} MB")

balloon.deflate(verbose=False)
print()

# Test 5: Workflow completo realistico
print("=" * 70)
print("TEST 5: Workflow realistico - Training con allocazione dinamica")
print("=" * 70)

balloon = GPUMemoryBalloon(gpu_id=0, reserve_mb=1000, chunk_size_mb=200)

# Fase 1: Setup iniziale
print("  [Fase 1] Setup - Alloca 3GB per caricamento modello")
balloon.allocate_memory("3GB", verbose=False)
print(f"    -> Allocato: {balloon.get_status()['allocated_mb']} MB")

# Fase 2: Pre-training - serve più memoria
print("  [Fase 2] Pre-training - Aumenta a 6GB")
balloon.allocate_to_target("6GB", verbose=False)
print(f"    -> Allocato: {balloon.get_status()['allocated_mb']} MB")

# Fase 3: Validation - serve meno memoria
print("  [Fase 3] Validation - Riduci a 4GB")
balloon.allocate_to_target("4GB", verbose=False)
print(f"    -> Allocato: {balloon.get_status()['allocated_mb']} MB")

# Fase 4: Fine-tuning - memoria moderata
print("  [Fase 4] Fine-tuning - Alloca fino a 50% memoria libera")
balloon.allocate_to_target("50%", verbose=False)
print(f"    -> Allocato: {balloon.get_status()['allocated_mb']} MB")

# Cleanup
print("  [Cleanup] Rilascio memoria")
balloon.deflate(verbose=False)
print(f"    -> Allocato: {balloon.get_status()['allocated_mb']} MB")
print()

# Test 6: Multi-GPU con formati flessibili
if torch.cuda.device_count() > 1:
    print("=" * 70)
    print(f"TEST 6: Multi-GPU - Riserva su tutte le {torch.cuda.device_count()} GPU")
    print("=" * 70)

    with reserve_all_gpus_memory(reserve_mb_per_gpu="1GB", chunk_size_mb=200) as multi:
        status_all = multi.get_status_all()
        print(f"  GPU totali gestite: {len(status_all)}")
        for gpu_id, status in status_all.items():
            print(f"    GPU {gpu_id}: {status['allocated_mb']} MB allocati, "
                  f"{status['free_memory_mb']} MB liberi")
    print("  Memoria rilasciata automaticamente su tutte le GPU")
    print()

# Riepilogo finale
print("=" * 70)
print("RIEPILOGO")
print("=" * 70)
print("✅ Tutti i test completati con successo!")
print()
print("Formati supportati per specificare memoria:")
print("  - int:           2000 (MB)")
print("  - str GB:        '2GB', '2.5GB'")
print("  - str MB:        '2048MB'")
print("  - str %:         '50%', '75%'")
print("  - float 0.0-1.0: 0.5, 0.7")
print()
print("Metodi di allocazione:")
print("  - allocate_memory():      Alloca quantità specifica")
print("  - allocate_additional():  Aggiunge memoria incrementalmente")
print("  - allocate_to_target():   Alloca/dealloca fino a target")
print("  - resize_balloon():       Ridimensiona (dealloca + rialloca)")
print()
print("=" * 70)
