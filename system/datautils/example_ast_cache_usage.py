"""
Example usage of AST embedding cache for VEGAS dataset.

This script demonstrates how to:
1. Extract AST embeddings from audio files
2. Save embeddings to cache with metadata
3. Load embeddings from cache on subsequent runs
4. Verify cache compatibility
"""

import os
import torch
import logging
from dataset_vegas import VEGASDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_ast_embeddings(dataset, ast_model, sample_rate=16000, duration=5.0):
    """
    Extract AST embeddings from dataset audio files.

    Args:
        dataset: VEGASDataset instance
        ast_model: AST model for embedding extraction
        sample_rate: Audio sample rate
        duration: Audio duration in seconds

    Returns:
        Dictionary mapping file IDs to embeddings
    """
    logger.info(f"Extracting AST embeddings for {len(dataset)} samples...")

    embeddings = {}

    for idx, sample in enumerate(dataset.samples):
        try:
            # Load audio
            audio = dataset._load_audio(sample['audio_path'])

            # Extract AST embedding
            with torch.no_grad():
                audio_input = audio.unsqueeze(0)  # Add batch dimension
                embedding = ast_model(audio_input)

            # Store with file_id and class_name as key (same format as audio_embs)
            sample_id = f"{sample['file_id']}:{sample['class_name'].lower()}"
            embeddings[sample_id] = embedding.squeeze(0).cpu()

            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(dataset)} samples")

        except Exception as e:
            logger.error(f"Error processing {sample['audio_path']}: {e}")
            continue

    logger.info(f"Extracted {len(embeddings)} AST embeddings")
    return embeddings


def example_with_cache_save(ast_model, root_dir="/home/lpala/fedgfe/dataset/Audio/VEGAS"):
    """
    Example: Extract AST embeddings and save to cache.
    """
    logger.info("=== Example 1: Extracting and caching AST embeddings ===")

    # Configuration
    sample_rate = 16000
    duration = 5.0
    model_name = "ast-finetuned"

    # Create dataset with AST cache enabled
    dataset = VEGASDataset(
        root_dir=root_dir,
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='train',
        enable_ast_cache=True,
        load_audio=True
    )

    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Check if cache already exists
    cache_loaded = dataset.load_ast_embeddings_from_cache(
        sample_rate=sample_rate,
        duration=duration,
        model_name=model_name
    )

    if cache_loaded:
        logger.info("✓ AST embeddings loaded from cache!")
        logger.info(f"Total embeddings: {len(dataset.audio_embs_from_file)}")
    else:
        logger.info("Cache not found, extracting AST embeddings...")

        # Extract embeddings
        embeddings = extract_ast_embeddings(dataset, ast_model, sample_rate, duration)

        # Save to cache
        success = dataset.save_ast_embeddings_to_cache(
            embeddings=embeddings,
            sample_rate=sample_rate,
            duration=duration,
            model_name=model_name
        )

        if success:
            logger.info("✓ AST embeddings cached successfully!")
        else:
            logger.error("✗ Failed to cache AST embeddings")


def example_with_cache_load(root_dir="/home/lpala/fedgfe/dataset/Audio/VEGAS"):
    """
    Example: Load AST embeddings from cache (fast).
    """
    logger.info("=== Example 2: Loading AST embeddings from cache ===")

    # Configuration (must match what was used during extraction)
    sample_rate = 16000
    duration = 5.0
    model_name = "ast-finetuned"

    # Create dataset
    dataset = VEGASDataset(
        root_dir=root_dir,
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='train',
        enable_ast_cache=True,
        load_audio=False  # No need to load audio if using cached embeddings
    )

    # Load from cache
    cache_loaded = dataset.load_ast_embeddings_from_cache(
        sample_rate=sample_rate,
        duration=duration,
        model_name=model_name
    )

    if cache_loaded:
        logger.info("✓ AST embeddings loaded from cache!")

        # Filter embeddings for active classes
        dataset.filter_audio_embeddings_from_file()
        logger.info(f"Filtered to {len(dataset.audio_embs)} embeddings for active classes")

        # Use embeddings in training
        sample = dataset[0]
        if 'audio_emb' in sample:
            logger.info(f"Sample embedding shape: {sample['audio_emb'].shape}")
        else:
            logger.warning("No audio embedding found in sample")
    else:
        logger.error("✗ Failed to load AST embeddings from cache")


def example_cache_invalidation(root_dir="/home/lpala/fedgfe/dataset/Audio/VEGAS"):
    """
    Example: Cache invalidation when configuration changes.
    """
    logger.info("=== Example 3: Cache invalidation ===")

    # Create dataset
    dataset = VEGASDataset(
        root_dir=root_dir,
        enable_ast_cache=True
    )

    # Try to load with different configurations
    logger.info("\n1. Trying to load cache with config: 16000Hz, 5.0s")
    cache_loaded_1 = dataset.load_ast_embeddings_from_cache(
        sample_rate=16000,
        duration=5.0,
        model_name="ast-finetuned"
    )
    logger.info(f"Result: {'✓ Loaded' if cache_loaded_1 else '✗ Not found/incompatible'}")

    logger.info("\n2. Trying to load cache with config: 16000Hz, 10.0s (different duration)")
    cache_loaded_2 = dataset.load_ast_embeddings_from_cache(
        sample_rate=16000,
        duration=10.0,  # Different duration
        model_name="ast-finetuned"
    )
    logger.info(f"Result: {'✓ Loaded' if cache_loaded_2 else '✗ Not found/incompatible'}")

    logger.info("\n3. Trying to load cache with config: 44100Hz, 5.0s (different sample rate)")
    cache_loaded_3 = dataset.load_ast_embeddings_from_cache(
        sample_rate=44100,  # Different sample rate
        duration=5.0,
        model_name="ast-finetuned"
    )
    logger.info(f"Result: {'✓ Loaded' if cache_loaded_3 else '✗ Not found/incompatible'}")


def example_clear_cache(root_dir="/home/lpala/fedgfe/dataset/Audio/VEGAS"):
    """
    Example: Clear AST cache.
    """
    logger.info("=== Example 4: Clearing AST cache ===")

    # Create dataset
    dataset = VEGASDataset(
        root_dir=root_dir,
        enable_ast_cache=True
    )

    # Clear all cache
    logger.info("\nClearing all AST cache files...")
    dataset.clear_ast_cache()

    # Or clear specific configuration
    logger.info("\nClearing specific cache (16000Hz, 5.0s)...")
    dataset.clear_ast_cache(sample_rate=16000, duration=5.0, model_name="ast-finetuned")


def example_server_integration(ast_model):
    """
    Example: Integration with serverA2V for AST embedding extraction.

    This shows how to use AST cache in the server training workflow.
    """
    logger.info("=== Example 5: Server integration ===")

    # Server configuration
    sample_rate = 16000
    duration = 5.0
    model_name = "ast-finetuned"

    # Create dataset with cache
    dataset = VEGASDataset(
        root_dir="/home/lpala/fedgfe/dataset/Audio/VEGAS",
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='train',
        enable_ast_cache=True,
        ast_cache_dir="/home/lpala/fedgfe/dataset/Audio/VEGAS/ast_cache",
        load_audio=True
    )

    # Try to load from cache first
    cache_loaded = dataset.load_ast_embeddings_from_cache(
        sample_rate=sample_rate,
        duration=duration,
        model_name=model_name
    )

    if not cache_loaded:
        logger.info("Cache miss - extracting AST embeddings (this may take a while)...")

        # Extract embeddings
        embeddings = extract_ast_embeddings(dataset, ast_model, sample_rate, duration)

        # Save to cache for next time
        dataset.save_ast_embeddings_to_cache(
            embeddings=embeddings,
            sample_rate=sample_rate,
            duration=duration,
            model_name=model_name
        )

        logger.info("✓ Embeddings extracted and cached")
    else:
        logger.info("✓ Cache hit - loaded embeddings from cache (fast!)")

    # Filter for active classes
    dataset.filter_audio_embeddings_from_file()

    # Now use dataset in training...
    logger.info(f"Ready for training with {len(dataset.audio_embs)} cached embeddings")


if __name__ == "__main__":
    # Note: You need to initialize your AST model before running examples
    # For demonstration, we'll show the workflow without actual model

    logger.info("AST Cache Usage Examples")
    logger.info("=" * 60)

    # Example 1: First run - extract and cache
    # ast_model = load_your_ast_model()
    # example_with_cache_save(ast_model)

    # Example 2: Subsequent runs - load from cache (fast)
    # example_with_cache_load()

    # Example 3: Cache invalidation
    # example_cache_invalidation()

    # Example 4: Clear cache
    # example_clear_cache()

    # Example 5: Server integration
    # example_server_integration(ast_model)

    logger.info("\nTo use these examples:")
    logger.info("1. Initialize your AST model")
    logger.info("2. Run example_with_cache_save() to extract and cache embeddings")
    logger.info("3. Run example_with_cache_load() on subsequent runs (fast!)")
    logger.info("4. Cache is automatically invalidated if config changes")
