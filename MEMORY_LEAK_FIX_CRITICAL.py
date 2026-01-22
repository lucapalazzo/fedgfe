"""
CRITICAL MEMORY LEAK FIX - Vegas 5n-1c-real Configuration

This file contains the corrected version of get_audio_embeddings_from_dataset()
which is the PRIMARY source of memory leak in the image generation pipeline.

USAGE:
1. Replace the method in system/flcore/clients/clientA2V.py (lines 1409-1463)
2. Test with test_memory_leak_fix.py
3. Monitor memory usage across rounds

ISSUE: The original function loops over all batches but only returns the LAST batch,
       causing massive GPU memory leak as intermediate tensors are never freed.

FIX: Accumulate all batches properly and cleanup after each iteration.
"""

import torch
import logging
from torch.utils.data import DataLoader
from transformers import ASTFeatureExtractor

logger = logging.getLogger(__name__)


def get_audio_embeddings_from_dataset(self, dataset):
    """
    Generate audio embeddings for all samples in a given dataset.

    FIXED VERSION - Properly accumulates all batches and cleans up GPU memory.

    Args:
        dataset: Dataset to process (train/test/val)

    Returns:
        dict: {
            'clip': torch.Tensor of shape (total_samples, hidden_dim),
            't5': torch.Tensor of shape (total_samples, seq_len, hidden_dim),
            'class_name': list of class names
        }
        Returns None if AST model is not initialized.
    """
    # Create dataloader for the entire dataset
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # Check if AST model is initialized (needed for audio processing)
    if self.model.ast_model is None or self.model.ast_feature_extractor is None:
        logger.warning(f"Node {self.id}: AST model not initialized (using pretrained generators), cannot process audio")
        return None

    # Accumulatori per tutti i batch
    all_embeddings = {module_name: [] for module_name in self.adapters.keys()}
    all_class_names = []

    # Move AST model to device ONCE (not in every iteration)
    ast_model = self.model.ast_model.to(self.device)
    ast_model.eval()

    # Move adapters to device ONCE (not in every iteration)
    # Store original device for later restoration
    adapter_original_devices = {}
    for module_name in self.adapters.keys():
        adapter_original_devices[module_name] = next(self.adapters[module_name].parameters()).device
        self.adapters[module_name] = self.adapters[module_name].to(self.device)

    total_batches = len(dataloader)
    print(f"Node {self.id} - Processing {total_batches} batches from dataset...")

    with torch.no_grad():
        for batch_idx, samples in enumerate(dataloader):
            # Variables to cleanup in finally block
            audio_data = None
            audio_data_np = None
            audio_inputs = None
            audio_embeddings = None

            try:
                # Extract audio data from batch
                if 'audio' not in samples or not isinstance(samples['audio'], torch.Tensor):
                    logger.warning(f"Node {self.id}: Batch {batch_idx} missing audio data, skipping")
                    continue

                audio_data = samples['audio'].to(self.device)

                # Convert to numpy for AST feature extractor
                if isinstance(self.model.ast_feature_extractor, ASTFeatureExtractor):
                    audio_data_np = audio_data.cpu().numpy()

                    # Extract features using AST feature extractor
                    audio_inputs = self.model.ast_feature_extractor(
                        audio_data_np,
                        sampling_rate=16000,
                        return_tensors="pt",
                        padding=True
                    ).input_values.to(self.device, self.model.torch_dtype)

                    # Forward through AST model to get audio embeddings
                    audio_embeddings = ast_model(audio_inputs).last_hidden_state  # (batch, seq_len, feature_dim)
                else:
                    # Alternative path if not using AST feature extractor
                    # (e.g., using pretrained generators directly)
                    class_names = samples.get('class_name', [])
                    audio_embeddings_list = []
                    for class_name in class_names:
                        if class_name in self.model.generators_dict:
                            gen_emb = self.model.generators_dict[class_name].sample(
                                num_samples=1,
                                device=self.device,
                                target_sequence_length=1214
                            )
                            audio_embeddings_list.append(gen_emb)
                        else:
                            logger.warning(f"Node {self.id}: No generator for class '{class_name}'")

                    if audio_embeddings_list:
                        audio_embeddings = torch.cat(audio_embeddings_list, dim=0)
                    else:
                        logger.warning(f"Node {self.id}: Batch {batch_idx} failed to generate embeddings")
                        continue

                # Process through adapters to get text embeddings
                batch_embeddings = {}
                for module_name in self.adapters.keys():
                    adapter = self.adapters[module_name]  # Already on device
                    adapter.eval()

                    # Forward through adapter
                    output = adapter(audio_embeddings)

                    # Move to CPU immediately to free GPU memory
                    # Store in list for later concatenation
                    batch_embeddings[module_name] = output.cpu()

                # Accumulate results
                for module_name in self.adapters.keys():
                    all_embeddings[module_name].append(batch_embeddings[module_name])

                # Store class names
                batch_class_names = samples.get('class_name', [])
                all_class_names.extend(batch_class_names)

                # Progress logging
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                    print(f"  Progress: {batch_idx + 1}/{total_batches} batches processed, "
                          f"{len(all_class_names)} total samples")

            except Exception as e:
                logger.error(f"Node {self.id}: Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

            finally:
                # CRITICAL: Cleanup batch tensors to free GPU memory
                # This prevents accumulation of tensors across iterations
                if audio_data is not None:
                    del audio_data
                if audio_data_np is not None:
                    del audio_data_np
                if audio_inputs is not None:
                    del audio_inputs
                if audio_embeddings is not None:
                    del audio_embeddings
                if 'batch_embeddings' in locals():
                    for key in list(batch_embeddings.keys()):
                        del batch_embeddings[key]
                    del batch_embeddings

                # Force GPU cache cleanup every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    torch.cuda.empty_cache()

    # Check if we collected any data
    if not all_class_names:
        logger.warning(f"Node {self.id}: No samples collected from dataset")
        return None

    # Concatenate all batches
    result = {}
    for module_name in all_embeddings.keys():
        if all_embeddings[module_name]:
            # Concatenate tensors from all batches
            result[module_name] = torch.cat(all_embeddings[module_name], dim=0)
        else:
            logger.warning(f"Node {self.id}: No embeddings collected for adapter '{module_name}'")

    result['class_name'] = all_class_names

    # Final cleanup
    for module_name in all_embeddings.keys():
        del all_embeddings[module_name]
    del all_embeddings

    # Restore adapters to original devices (optional, for compatibility)
    # Comment out if you want to keep them on GPU
    # for module_name, original_device in adapter_original_devices.items():
    #     self.adapters[module_name] = self.adapters[module_name].to(original_device)

    torch.cuda.empty_cache()

    print(f"Node {self.id} - ✓ Retrieved {len(all_class_names)} audio embeddings from dataset")
    print(f"Node {self.id} - Embedding shapes: {', '.join([f'{k}: {v.shape}' for k, v in result.items() if isinstance(v, torch.Tensor)])}")

    return result


# ============================================================================
# ADDITIONAL FIXES (OPTIONAL BUT RECOMMENDED)
# ============================================================================

def generate_images_from_diffusion_FIXED(self, text_embeddings, base_embeddings=None):
    """
    FIXED VERSION of generate_images_from_diffusion with proper cleanup.

    Replace in system/flcore/servers/serverA2V.py (lines 3114-3165)
    """
    if 't5' not in text_embeddings or 'clip' not in text_embeddings:
        print('Text embeddings is missing something')
        return []

    prompt_embeds = None
    pooled_prompt_embeds = None
    imgs = None

    try:
        # Prepare embeddings based on configuration
        if self.generate_from_t5_text_embeddings and base_embeddings:
            prompt_embeds_list = []
            for class_name in text_embeddings['class_name']:
                if class_name in base_embeddings:
                    prompt_embeds_list.append(base_embeddings[class_name]['flux']['prompt_embeds'])
                else:
                    print(f"Class name {class_name} not found in base embeddings")
                    continue
            prompt_embeds = torch.stack(prompt_embeds_list).squeeze(dim=1).to(
                self.global_model.diffusion_dtype).to(self.diffusion_device)
        else:
            prompt_embeds = text_embeddings['t5'].to(
                self.global_model.diffusion_dtype).to(self.diffusion_device)

        if self.generate_from_clip_text_embeddings and base_embeddings:
            pooled_prompt_embeds_list = []
            for class_name in text_embeddings['class_name']:
                if class_name in base_embeddings:
                    pooled_prompt_embeds_list.append(base_embeddings[class_name]['flux']['pooled_prompt_embeds'])
                else:
                    print(f"Class name {class_name} not found in base embeddings")
                    continue
            pooled_prompt_embeds = torch.stack(pooled_prompt_embeds_list).squeeze(dim=1).to(
                self.global_model.diffusion_dtype).to(self.diffusion_device)
        else:
            pooled_prompt_embeds = text_embeddings['clip'].to(
                self.global_model.diffusion_dtype).to(self.diffusion_device)

        # Generate images
        if not self.generate_low_memomy_footprint:
            imgs = self.global_model.diffusion_model(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=1,
                output_type="pt",
            ).images
        else:
            imgs = self.generate_single_images_from_diffusion(prompt_embeds, pooled_prompt_embeds)

        return imgs

    finally:
        # CRITICAL: Cleanup intermediate tensors
        if prompt_embeds is not None:
            del prompt_embeds
        if pooled_prompt_embeds is not None:
            del pooled_prompt_embeds
        torch.cuda.empty_cache()


def generate_images_FIXED(self, client):
    """
    FIXED VERSION of generate_images with proper cleanup between splits.

    Replace in system/flcore/servers/serverA2V.py (lines 3201-3242)
    """
    print(f"\nGenerating images for Node {client.id} using Audio2Visual model.")

    if not os.path.exists(self.images_output_dir):
        try:
            os.makedirs(self.images_output_dir)
        except FileExistsError:
            pass

    # Get all available datasets
    node_val_dataset = client.node_data.get_val_dataset()
    node_test_dataset = client.node_data.get_test_dataset()
    node_train_dataset = client.node_data.get_train_dataset()

    # Map split names to datasets
    split_datasets = {
        'val': node_val_dataset,
        'test': node_test_dataset,
        'train': node_train_dataset
    }

    # Initialize result dictionary
    generated_images_files = {}

    # Use nodes splits for per-node generation
    generation_splits = self.nodes_test_metrics_splits

    # Generate images for configured splits
    for split_name in generation_splits:
        dataset = split_datasets.get(split_name)

        if dataset is None or len(dataset) == 0:
            print(f"Unable to get {split_name} split from node {client.id}")
            continue

        # Variables for cleanup
        text_embs = None
        embeddings = None
        generated_imgs = None

        try:
            print(f"Generating images for split: {split_name}")
            text_embs = dataset.text_embs

            # Get audio embeddings from dataset (FIXED function)
            embeddings = client.get_audio_embeddings_from_dataset(dataset)

            if embeddings is None:
                print(f"Failed to get embeddings for {split_name} split")
                continue

            # Generate images from diffusion model
            generated_imgs = self.generate_images_from_diffusion(embeddings, base_embeddings=text_embs)

            # Save generated images
            saved_files = self.save_generated_images(
                generated_imgs,
                client.id,
                embeddings,
                suffix=f'{split_name}'
            )

            generated_images_files[split_name] = saved_files

        except Exception as e:
            logger.error(f"Error generating images for split {split_name}: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # CRITICAL: Cleanup split data to prevent accumulation
            if embeddings is not None:
                for key in list(embeddings.keys()):
                    if isinstance(embeddings[key], torch.Tensor):
                        del embeddings[key]
                del embeddings

            if generated_imgs is not None:
                del generated_imgs

            # Force GPU cleanup after each split
            torch.cuda.empty_cache()

            print(f"✓ Cleaned up memory after {split_name} split")

    return generated_images_files
