#!/usr/bin/env python3
"""
Generate images from saved embedding checkpoints.

This script loads embedding checkpoints created during federated training
and generates images using FLUX diffusion model. This allows separating
the embedding computation phase from the computationally expensive image
generation phase.

Usage:
    python generate_images_from_checkpoint.py --checkpoint path/to/checkpoint.pt
    python generate_images_from_checkpoint.py --checkpoint_dir checkpoints/embeddings/
    python generate_images_from_checkpoint.py --checkpoint_dir checkpoints/embeddings/ --batch_size 4
"""

import argparse
import os
import torch
import logging
from pathlib import Path
from tqdm import tqdm
import sys

# Add system path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from flcore.trainmodel.Audio2Visual_NoData.src.models.audio2image import SDImageModel
from diffusers import FluxPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EmbeddingImageGenerator:
    """Generate images from saved embedding checkpoints."""

    def __init__(self, diffusion_type='flux', device='cuda', batch_size=1):
        """
        Initialize image generator.

        Args:
            diffusion_type: Type of diffusion model ('flux', 'sd', etc.)
            device: Device to run on ('cuda' or 'cpu')
            batch_size: Number of images to generate in parallel
        """
        self.diffusion_type = diffusion_type
        self.device = device
        self.batch_size = batch_size
        self.diffusion_model = None

        logger.info(f"Initializing {diffusion_type} diffusion model on {device}")
        self._initialize_diffusion_model()

    def _initialize_diffusion_model(self):
        """Initialize the diffusion model based on type."""
        if self.diffusion_type == 'flux':
            logger.info("Loading FLUX pipeline...")
            self.diffusion_model = FluxPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-dev",
                torch_dtype=torch.bfloat16
            ).to(self.device)
            logger.info("FLUX pipeline loaded successfully")

        elif self.diffusion_type == 'sd':
            logger.info("Loading Stable Diffusion model...")
            self.diffusion_model = SDImageModel(
                device=self.device,
                img_pipe_name="runwayml/stable-diffusion-v1-5"
            )
            logger.info("Stable Diffusion model loaded successfully")

        else:
            raise ValueError(f"Unsupported diffusion type: {self.diffusion_type}")

    def load_checkpoint(self, checkpoint_path):
        """
        Load embedding checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Checkpoint data dict
        """
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        logger.info(f"Checkpoint info:")
        logger.info(f"  Node ID: {checkpoint['node_id']}")
        logger.info(f"  Round: {checkpoint['round']}")
        logger.info(f"  Timestamp: {checkpoint['timestamp']}")
        logger.info(f"  Total embeddings: {len(checkpoint['embeddings'])}")
        logger.info(f"  Classes: {checkpoint['metadata']['selected_classes']}")

        return checkpoint

    def generate_images_from_checkpoint(self, checkpoint_path, output_dir=None):
        """
        Generate all images from a checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file
            output_dir: Optional override for output directory

        Returns:
            List of generated image paths
        """
        checkpoint = self.load_checkpoint(checkpoint_path)
        embeddings_data = checkpoint['embeddings']
        generated_paths = []

        logger.info(f"Generating {len(embeddings_data)} images...")

        # Process in batches
        for batch_start in tqdm(range(0, len(embeddings_data), self.batch_size),
                                desc=f"Generating images (batch_size={self.batch_size})"):
            batch_end = min(batch_start + self.batch_size, len(embeddings_data))
            batch = embeddings_data[batch_start:batch_end]

            # Generate images for this batch
            for embedding_data in batch:
                image_path = self._generate_single_image(embedding_data, output_dir)
                if image_path:
                    generated_paths.append(image_path)

        logger.info(f"✓ Generated {len(generated_paths)} images")
        return generated_paths

    def _generate_single_image(self, embedding_data, output_dir=None):
        """
        Generate a single image from embedding data.

        Args:
            embedding_data: Dict with embedding and metadata
            output_dir: Optional override for output directory

        Returns:
            Path to generated image
        """
        # Determine output path
        if output_dir:
            image_filename = os.path.basename(embedding_data['image_filename'])
            image_path = os.path.join(output_dir, image_filename)
        else:
            image_path = embedding_data['image_path']

        # Create output directory
        os.makedirs(os.path.dirname(image_path), exist_ok=True)

        # Get embeddings
        t5_emb = embedding_data.get('t5_embedding')
        clip_emb = embedding_data.get('clip_embedding')

        if t5_emb is None and clip_emb is None:
            logger.warning(f"No embeddings found for {image_path}, skipping")
            return None

        # Move embeddings to device
        if t5_emb is not None:
            t5_emb = t5_emb.to(self.device)
        if clip_emb is not None:
            clip_emb = clip_emb.to(self.device)

        # Generate image
        try:
            if self.diffusion_type == 'flux':
                # FLUX uses both prompt_embeds (CLIP) and pooled_prompt_embeds (T5)
                images = self.diffusion_model(
                    prompt_embeds=clip_emb if clip_emb is not None else t5_emb,
                    pooled_prompt_embeds=t5_emb if t5_emb is not None else clip_emb,
                    num_inference_steps=50,
                    guidance_scale=7.5,
                    height=1024,
                    width=1024
                ).images

            elif self.diffusion_type == 'sd':
                # SD uses different API
                images = self.diffusion_model.generate_from_embeddings(
                    text_embeddings=clip_emb if clip_emb is not None else t5_emb
                )

            # Save image
            if images and len(images) > 0:
                images[0].save(image_path)
                logger.debug(f"Saved image: {image_path}")
                return image_path
            else:
                logger.warning(f"No image generated for {image_path}")
                return None

        except Exception as e:
            logger.error(f"Failed to generate image for {image_path}: {e}")
            return None

    def process_checkpoint_directory(self, checkpoint_dir, output_dir=None):
        """
        Process all checkpoint files in a directory.

        Args:
            checkpoint_dir: Directory containing checkpoint files
            output_dir: Optional override for output directory

        Returns:
            Dict mapping checkpoint files to generated image paths
        """
        checkpoint_files = list(Path(checkpoint_dir).glob("*.pt"))
        logger.info(f"Found {len(checkpoint_files)} checkpoint files in {checkpoint_dir}")

        results = {}
        for checkpoint_path in checkpoint_files:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing: {checkpoint_path.name}")
            logger.info(f"{'='*80}")

            generated_paths = self.generate_images_from_checkpoint(
                str(checkpoint_path),
                output_dir=output_dir
            )
            results[str(checkpoint_path)] = generated_paths

        return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate images from embedding checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to single checkpoint file'
    )

    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='checkpoints/embeddings',
        help='Directory containing checkpoint files (default: checkpoints/embeddings)'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        help='Override output directory for generated images (optional)'
    )

    parser.add_argument(
        '--diffusion_type',
        type=str,
        default='flux',
        choices=['flux', 'sd'],
        help='Diffusion model type (default: flux)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on (default: cuda if available)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='Batch size for generation (default: 1)'
    )

    args = parser.parse_args()

    # Initialize generator
    generator = EmbeddingImageGenerator(
        diffusion_type=args.diffusion_type,
        device=args.device,
        batch_size=args.batch_size
    )

    # Process checkpoints
    if args.checkpoint:
        # Single checkpoint file
        logger.info(f"Processing single checkpoint: {args.checkpoint}")
        generated_paths = generator.generate_images_from_checkpoint(
            args.checkpoint,
            output_dir=args.output_dir
        )
        logger.info(f"\n{'='*80}")
        logger.info(f"✓ Complete! Generated {len(generated_paths)} images")
        logger.info(f"{'='*80}")

    else:
        # Checkpoint directory
        logger.info(f"Processing checkpoint directory: {args.checkpoint_dir}")
        results = generator.process_checkpoint_directory(
            args.checkpoint_dir,
            output_dir=args.output_dir
        )

        # Summary
        total_images = sum(len(paths) for paths in results.values())
        logger.info(f"\n{'='*80}")
        logger.info(f"✓ Complete!")
        logger.info(f"  Processed checkpoints: {len(results)}")
        logger.info(f"  Total images generated: {total_images}")
        logger.info(f"{'='*80}")


if __name__ == '__main__':
    main()
