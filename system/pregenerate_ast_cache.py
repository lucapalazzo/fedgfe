#!/usr/bin/env python3
"""
Pre-generate AST embeddings cache for VEGAS dataset.

This script computes AST (Audio Spectrogram Transformer) embeddings for all samples
in the VEGAS dataset and saves them to disk cache for faster training.

Usage:
    python system/pregenerate_ast_cache.py --dataset_path dataset/Audio/VEGAS \
                                           --cache_dir cache/ast/vegas \
                                           --batch_size 32 \
                                           --device cuda

Features:
    - Incremental save (appends to existing cache)
    - Memory efficient (processes in batches)
    - Automatic duplicate detection
    - Progress tracking with tqdm
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import ASTFeatureExtractor, ASTModel

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'system'))

from datautils.dataset_vegas import VEGASDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pre-generate AST embeddings cache for VEGAS dataset'
    )

    parser.add_argument(
        '--dataset_path',
        type=str,
        default='dataset/Audio/VEGAS',
        help='Path to VEGAS dataset directory'
    )

    parser.add_argument(
        '--cache_dir',
        type=str,
        default='cache/ast/vegas',
        help='Directory to save AST embeddings cache'
    )

    parser.add_argument(
        '--selected_classes',
        type=str,
        nargs='+',
        default=None,
        help='List of classes to process (default: all classes)'
    )

    parser.add_argument(
        '--samples_per_class',
        type=int,
        default=None,
        help='Limit number of samples per class (default: all samples)'
    )

    parser.add_argument(
        '--node_split_id',
        type=int,
        default=0,
        help='Node split ID for data partitioning'
    )

    parser.add_argument(
        '--ast_model_name',
        type=str,
        default='MIT/ast-finetuned-audioset-10-10-0.4593',
        help='HuggingFace AST model name'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for processing'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda or cpu)'
    )

    parser.add_argument(
        '--save_every',
        type=int,
        default=100,
        help='Save cache every N samples (for incremental saves)'
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of DataLoader workers'
    )

    return parser.parse_args()


def collate_fn(batch):
    """Custom collate function that handles batch of samples."""
    # Stack tensors where possible
    batch_dict = {}

    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            batch_dict[key] = torch.stack([item[key] for item in batch])
        elif isinstance(batch[0][key], (str, int, float)):
            batch_dict[key] = [item[key] for item in batch]
        else:
            batch_dict[key] = [item[key] for item in batch]

    return batch_dict


def main():
    args = parse_args()

    logger.info("="*80)
    logger.info("AST Embeddings Cache Pre-generation")
    logger.info("="*80)
    logger.info(f"Dataset path: {args.dataset_path}")
    logger.info(f"Cache dir: {args.cache_dir}")
    logger.info(f"Selected classes: {args.selected_classes if args.selected_classes else 'ALL'}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"AST model: {args.ast_model_name}")
    logger.info("="*80)

    # Create cache directory
    os.makedirs(args.cache_dir, exist_ok=True)
    logger.info(f"Created cache directory: {args.cache_dir}")

    # Load dataset
    logger.info("Loading VEGAS dataset...")
    dataset = VEGASDataset(
        root_dir=args.dataset_path,
        selected_classes=args.selected_classes,
        samples_per_node=args.samples_per_class,
        node_split_id=args.node_split_id,
        ast_cache_dir=args.cache_dir,
        enable_ast_cache=True
    )

    logger.info(f"Loaded dataset: {len(dataset)} samples")
    logger.info(f"Active classes: {list(dataset.active_classes.keys())}")

    # Load AST model
    logger.info(f"Loading AST model: {args.ast_model_name}")
    ast_feature_extractor = ASTFeatureExtractor.from_pretrained(args.ast_model_name)
    ast_model = ASTModel.from_pretrained(args.ast_model_name)
    ast_model = ast_model.to(args.device)
    ast_model.eval()
    logger.info(f"AST model loaded on {args.device}")

    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(args.device == 'cuda')
    )

    # Process batches and accumulate embeddings
    ast_outputs_by_class = {}
    processed_samples = 0

    logger.info("Computing AST embeddings...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
            # Get audio data
            audio = batch['audio']
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()

            # Compute AST embeddings
            audio_inputs = ast_feature_extractor(
                audio,
                sampling_rate=16000,
                return_tensors="pt",
                padding=True
            ).input_values.to(args.device)

            ast_outputs = ast_model(audio_inputs).last_hidden_state

            # Organize by class
            class_names = batch['class_name']
            file_ids = batch['file_id']

            for i in range(len(class_names)):
                class_name = class_names[i]
                file_id = file_ids[i]
                embedding = ast_outputs[i].cpu()

                if class_name not in ast_outputs_by_class:
                    ast_outputs_by_class[class_name] = {}

                ast_outputs_by_class[class_name][file_id] = embedding
                processed_samples += 1

            # Incremental save
            if (batch_idx + 1) % (args.save_every // args.batch_size) == 0:
                logger.info(f"Saving checkpoint at {processed_samples} samples...")
                saved_counts = dataset.save_ast_embeddings_to_cache(
                    ast_outputs_dict=ast_outputs_by_class,
                    cache_dir=args.cache_dir
                )
                logger.info(f"Saved: {saved_counts}")

                # Clear accumulator after save to free memory
                ast_outputs_by_class = {}

                # Empty CUDA cache
                if args.device == 'cuda':
                    torch.cuda.empty_cache()

    # Final save
    if len(ast_outputs_by_class) > 0:
        logger.info(f"Saving final checkpoint...")
        saved_counts = dataset.save_ast_embeddings_to_cache(
            ast_outputs_dict=ast_outputs_by_class,
            cache_dir=args.cache_dir
        )
        logger.info(f"Final save: {saved_counts}")

    # Summary
    logger.info("="*80)
    logger.info("Cache generation completed!")
    logger.info(f"Total samples processed: {processed_samples}")
    logger.info(f"Cache location: {args.cache_dir}")
    logger.info("="*80)

    # Verify cache
    logger.info("Verifying cache...")
    loaded_cache = dataset.load_ast_embeddings_from_cache(
        cache_dir=args.cache_dir,
        classes=list(dataset.active_classes.keys())
    )

    if loaded_cache:
        logger.info("Cache verification successful:")
        for class_name, cache_info in loaded_cache.items():
            num_samples = cache_info['manifest']['total_samples']
            num_chunks = len(cache_info['chunks'])
            logger.info(f"  - {class_name}: {num_samples} samples, {num_chunks} chunks")
    else:
        logger.warning("Cache verification failed - no cache found")

    logger.info("Done!")


if __name__ == '__main__':
    main()
