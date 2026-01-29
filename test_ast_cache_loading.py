#!/usr/bin/env python3
"""
Quick test to verify AST cache loading in VEGASDataset
"""

import os
import sys
import torch
import logging

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'system'))

from datautils.dataset_vegas import VEGASDataset

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_ast_cache_loading():
    """Test that AST cache is loaded and used correctly."""

    print("="*80)
    print("Testing AST Cache Loading in VEGASDataset")
    print("="*80)

    # Test parameters
    dataset_path = 'dataset/Audio/VEGAS'
    cache_dir = 'cache/ast/vegas'
    selected_classes = ['chainsaw', 'dog']  # Use 2 classes for quick test

    print(f"\nDataset path: {dataset_path}")
    print(f"Cache dir: {cache_dir}")
    print(f"Selected classes: {selected_classes}")

    # Check if cache exists
    cache_exists = os.path.exists(cache_dir)
    print(f"\nCache directory exists: {cache_exists}")

    if cache_exists:
        for class_name in selected_classes:
            class_cache_dir = os.path.join(cache_dir, class_name)
            manifest_file = os.path.join(class_cache_dir, 'manifest.json')
            print(f"  {class_name}: {os.path.exists(manifest_file)}")

    # Create dataset
    print(f"\n{'='*80}")
    print("Creating VEGASDataset...")
    print(f"{'='*80}")

    dataset = VEGASDataset(
        root_dir=dataset_path,
        selected_classes=selected_classes,
        samples_per_node=10,  # Limit to 10 samples for quick test
        node_split_id=0,
        enable_ast_cache=True,
        ast_cache_dir=cache_dir
    )

    print(f"\nDataset created: {len(dataset)} samples")
    print(f"Active classes: {list(dataset.active_classes.keys())}")

    # Check internal cache state
    print(f"\n{'='*80}")
    print("Checking internal cache state...")
    print(f"{'='*80}")

    has_ast_cache = hasattr(dataset, '_ast_cache')
    print(f"Has _ast_cache attribute: {has_ast_cache}")

    if has_ast_cache:
        cache_size = len(dataset._ast_cache)
        print(f"Cache size: {cache_size} classes")
        print(f"Cached classes: {list(dataset._ast_cache.keys())}")

        if cache_size > 0:
            for class_name, cache_data in dataset._ast_cache.items():
                num_samples = cache_data['manifest']['total_samples']
                num_chunks = len(cache_data['chunks'])
                print(f"  {class_name}: {num_samples} samples, {num_chunks} chunks")

    # Test __getitem__ with cache
    print(f"\n{'='*80}")
    print("Testing __getitem__ with cache...")
    print(f"{'='*80}")

    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        class_name = sample['class_name']
        file_id = sample['file_id']
        has_audio_emb = 'audio_emb' in sample

        print(f"\nSample {i}: class={class_name}, file_id={file_id}")
        print(f"  Has audio_emb in output: {has_audio_emb}")

        if has_audio_emb:
            emb_shape = sample['audio_emb'].shape
            print(f"  ✓ Audio embedding loaded from cache: {emb_shape}")
        else:
            print(f"  ✗ No audio embedding (will be computed during training)")

    # Test get_cached_ast_embedding directly
    print(f"\n{'='*80}")
    print("Testing get_cached_ast_embedding() directly...")
    print(f"{'='*80}")

    test_sample = dataset.samples[0]
    class_name = test_sample['class_name']
    file_id = test_sample['file_id']

    print(f"\nTest sample: class={class_name}, file_id={file_id}")

    cached_emb = dataset.get_cached_ast_embedding(class_name, file_id)

    if cached_emb is not None:
        print(f"✓ Successfully retrieved from cache!")
        print(f"  Shape: {cached_emb.shape}")
        print(f"  Type: {type(cached_emb)}")
        print(f"  Dtype: {cached_emb.dtype}")
    else:
        print(f"✗ Not found in cache")
        print(f"  This could mean:")
        print(f"    1. Cache doesn't exist for this class")
        print(f"    2. This file_id is not in the cache")
        print(f"    3. Cache loading failed")

    print(f"\n{'='*80}")
    print("Test completed!")
    print(f"{'='*80}")

if __name__ == '__main__':
    test_ast_cache_loading()
