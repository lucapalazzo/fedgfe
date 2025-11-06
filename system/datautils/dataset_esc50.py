"""
ESC-50 Dataset loader for Audio2Visual tasks.
Handles multimodal data loading with caching and class filtering.
Based on ESC-50: Dataset for Environmental Sound Classification
"""

import os
import json
import pickle
import hashlib
import numpy as np
import torch
import torchaudio
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ESC50Dataset(Dataset):
    """
    ESC-50 Dataset class for multimodal (audio, image) data loading.
    Supports caching, class filtering, and federated learning splits.

    ESC-50 contains 2000 environmental audio recordings organized into 50 classes.
    """

    def __init__(self,
                 root_dir: str = "/home/lpala/fedgfe/dataset/Audio/esc50-v2.0.0-full",
                 text_embedding_file: Optional[str] = "/home/lpala/fedgfe/dataset/Audio/esc50_text_embs_dict.pt",
                 audio_embedding_file: Optional[str] = None,
                 selected_classes: Optional[Union[List[str], List[int]]] = None,
                 excluded_classes: Optional[Union[List[str], List[int]]] = None,
                 split: str = "all",
                 split_ratio: float = 0.8,
                 use_folds: bool = False,
                 train_folds: List[int] = [0, 1, 2, 3],
                 test_folds: List[int] = [4],
                 node_id: Optional[int] = None,
                 enable_cache: bool = False,
                 cache_dir: str = "/tmp/esc50_cache",
                 audio_sample_rate: int = 16000,
                 audio_duration: float = 5.0,  # ESC-50 clips are 5 seconds
                 image_size: Tuple[int, int] = (224, 224),
                 transform_audio: Optional = None,
                 transform_image: Optional = None):
        """
        Initialize ESC-50 dataset.

        Args:
            root_dir: Root directory of ESC-50 dataset
            embedding_file: Path to text embeddings file
            audio_embedding_file: Path to audio embeddings file
            selected_classes: List of classes to include (str names or int labels)
            excluded_classes: List of classes to exclude (str names or int labels)
            split: 'train', 'test', or 'all'
            split_ratio: Ratio for train/test split (used if use_folds=False)
            use_folds: Whether to use official fold splits
            train_folds: List of fold indices for training (0-4)
            test_folds: List of fold indices for testing (0-4)
            node_id: Federated learning node ID for consistent splitting
            enable_cache: Whether to enable caching
            cache_dir: Directory for cache files
            audio_sample_rate: Target sample rate for audio
            audio_duration: Duration of audio clips in seconds
            image_size: Target size for images (H, W)
            transform_audio: Audio transform function
            transform_image: Image transform function
        """
        self.root_dir = root_dir
        self.embedding_file = text_embedding_file
        self.audio_embedding_file = audio_embedding_file
        self.split = split
        self.split_ratio = split_ratio
        self.use_folds = use_folds
        self.train_folds = train_folds
        self.test_folds = test_folds
        self.node_id = node_id
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir
        self.audio_sample_rate = audio_sample_rate
        self.audio_duration = audio_duration
        self.image_size = image_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load_audio = True
        self.load_image = True

        # Load class labels from JSON
        class_labels_path = os.path.join(self.root_dir, "class_labels.json")
        with open(class_labels_path, 'r') as f:
            self.CLASS_LABELS = json.load(f)

        self.LABEL_TO_CLASS = {v: k for k, v in self.CLASS_LABELS.items()}

        # Load captions if available
        self.captions = {}
        captions_path = os.path.join(self.root_dir, "captions.json")
        if os.path.exists(captions_path):
            with open(captions_path, 'r') as f:
                self.captions = json.load(f)

        # Load text embeddings if provided
        self.text_embs = None
        if text_embedding_file and os.path.exists(text_embedding_file):
            self.text_embs = torch.load(text_embedding_file, map_location=self.device)
            # Convert keys to lowercase
            self.text_embs = {k.lower(): v for k, v in self.text_embs.items()}

        # Load audio embeddings if provided
        self.audio_embs = None
        if audio_embedding_file and os.path.exists(audio_embedding_file):
            self.audio_embs = torch.load(audio_embedding_file, map_location=self.device)

        self._excluded_classes = None
        self._selected_classes = None

        # Process class selection
        self.selected_classes = self._process_class_selection(selected_classes)
        self.excluded_classes = self._process_class_selection(excluded_classes)

        self.available_classes = list(self.CLASS_LABELS.keys())

        # Create transforms
        self.transform_audio = transform_audio
        self.transform_image = transform_image or self._default_image_transform()

        # Create cache directory
        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        logger.info(f"ESC-50 Dataset initialized: {len(self.samples)} samples, "
                   f"classes: {len(self.active_classes)}, split: {split}")

    @property
    def selected_classes(self) -> Optional[List[str]]:
        return self._selected_classes

    @selected_classes.setter
    def selected_classes(self, value: Optional[Union[List[str], List[int]]]):
        self._selected_classes = self._process_class_selection(value)
        self.active_classes = self._filter_classes()
        self.samples = self._load_samples()

    @property
    def excluded_classes(self) -> Optional[List[str]]:
        return self._excluded_classes

    @excluded_classes.setter
    def excluded_classes(self, value: Optional[Union[List[str], List[int]]]):
        self._excluded_classes = self._process_class_selection(value)
        self.active_classes = self._filter_classes()
        self.samples = self._load_samples()

    def get_text_embeddings ( self, class_name ):
        if self.text_embs is not None and class_name in self.text_embs:
            return self.text_embs[class_name]
        return None

    def _process_class_selection(self, classes: Optional[Union[List[str], List[int]]]) -> Optional[List[str]]:
        """Process class selection, converting indices to names."""
        if classes is None:
            return None

        processed = []
        for cls in classes:
            if isinstance(cls, int):
                if cls in self.LABEL_TO_CLASS:
                    processed.append(self.LABEL_TO_CLASS[cls])
                else:
                    logger.warning(f"Invalid class index: {cls}")
            elif isinstance(cls, str):
                if cls in self.CLASS_LABELS:
                    processed.append(cls)
                else:
                    logger.warning(f"Invalid class name: {cls}")

        return processed if processed else None

    def _filter_classes(self) -> Dict[str, int]:
        """Filter classes based on selection and exclusion."""
        active_classes = dict(self.CLASS_LABELS)

        # Apply selection filter
        if self._selected_classes:
            active_classes = {cls: label for cls, label in active_classes.items()
                            if cls in self.selected_classes}

        # Apply exclusion filter
        if self._excluded_classes:
            active_classes = {cls: label for cls, label in active_classes.items()
                            if cls not in self.excluded_classes}

        # Remap labels to be contiguous starting from 0
        sorted_classes = sorted(active_classes.keys())
        active_classes = {cls: idx for idx, cls in enumerate(sorted_classes)}

        return active_classes

    def _default_image_transform(self):
        """Default image transforms."""
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

    def _load_samples(self) -> List[Dict]:
        """Load all samples with metadata."""
        cache_key = self._get_cache_key()
        cache_file = os.path.join(self.cache_dir, f"esc50_samples_{cache_key}.pkl")

        # Try to load from cache
        if self.enable_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    samples = pickle.load(f)
                logger.info(f"Loaded {len(samples)} samples from cache")
                return samples
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        # Load samples from fold JSON files
        samples = []

        if self.use_folds:
            # Use official fold splits
            if self.split == 'train':
                folds_to_use = self.train_folds
            elif self.split == 'test':
                folds_to_use = self.test_folds
            else:  # 'all'
                folds_to_use = [0, 1, 2, 3, 4]
        else:
            # Use all folds and split later
            folds_to_use = [0, 1, 2, 3, 4]

        for fold_idx in folds_to_use:
            fold_file = os.path.join(self.root_dir, f"fold0{fold_idx}.json")

            if not os.path.exists(fold_file):
                logger.warning(f"Fold file not found: {fold_file}")
                continue

            with open(fold_file, 'r') as f:
                fold_data = json.load(f)

            for audio_filename, class_list in fold_data.items():
                class_name = class_list[0]  # Each file has one class

                # Skip if class not in active classes
                if class_name not in self.active_classes:
                    continue

                # Extract file ID (remove .wav extension)
                file_id = audio_filename.replace('.wav', '')

                # Build paths
                audio_path = os.path.join(self.root_dir, "audio", f"fold0{fold_idx}", audio_filename)

                # Get class index for image lookup
                original_class_idx = self.CLASS_LABELS[class_name]
                image_idx = random.randint(0, 39)  # Randomly select one of the 5 images per class
                image_path = os.path.join(self.root_dir, "image", f"{class_name}_{image_idx}.png")

                # Check if files exist
                if os.path.exists(audio_path):
                    sample = {
                        'class_name': class_name,
                        'class_label': self.active_classes[class_name],
                        'original_class_label': original_class_idx,
                        'audio_filename': audio_filename,
                        'file_id': file_id,
                        'audio_path': audio_path,
                        'image_path': image_path if os.path.exists(image_path) else None,
                        'fold': fold_idx,
                        'sample_idx': len(samples),
                        'caption': self.captions.get(file_id, '')
                    }
                    samples.append(sample)

        # Apply train/test split if not using folds
        if not self.use_folds and self.split != 'all':
            samples = self._apply_split(samples)

        # Save to cache
        if self.enable_cache:
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(samples, f)
                logger.info(f"Saved {len(samples)} samples to cache")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

        return samples

    def _get_cache_key(self) -> str:
        """Generate cache key based on dataset configuration."""
        config_str = (f"{self.root_dir}_{self.active_classes}_{self.split}_{self.split_ratio}_"
                     f"{self.use_folds}_{self.train_folds}_{self.test_folds}_{self.node_id}")
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _apply_split(self, samples: List[Dict]) -> List[Dict]:
        """Apply train/test split to samples."""
        # Use reproducible random seed based on node_id
        rng = np.random.RandomState(42 + (self.node_id or 0))

        # Group samples by class for balanced splitting
        class_samples = {}
        for sample in samples:
            class_name = sample['class_name']
            if class_name not in class_samples:
                class_samples[class_name] = []
            class_samples[class_name].append(sample)

        # Split each class
        split_samples = []
        for class_name, cls_samples in class_samples.items():
            cls_samples = sorted(cls_samples, key=lambda x: x['file_id'])
            rng.shuffle(cls_samples)

            n_train = int(len(cls_samples) * self.split_ratio)

            if self.split == 'train':
                split_samples.extend(cls_samples[:n_train])
            elif self.split == 'test':
                split_samples.extend(cls_samples[n_train:])

        return split_samples

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio."""
        try:
            waveform, sample_rate = torchaudio.load(audio_path)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample if necessary
            if sample_rate != self.audio_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, self.audio_sample_rate)
                waveform = resampler(waveform)

            # Pad or truncate to target duration
            target_length = int(self.audio_duration * self.audio_sample_rate)
            if waveform.shape[1] < target_length:
                # Pad with zeros
                padding = target_length - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            elif waveform.shape[1] > target_length:
                # Truncate
                waveform = waveform[:, :target_length]

            # Apply transforms
            if self.transform_audio:
                waveform = self.transform_audio(waveform)

            return waveform.squeeze(0)  # Remove channel dimension

        except Exception as e:
            logger.error(f"Error loading audio {audio_path}: {e}")
            # Return silence as fallback
            target_length = int(self.audio_duration * self.audio_sample_rate)
            return torch.zeros(target_length)

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image."""
        if image_path is None or not os.path.exists(image_path):
            # Return black image as fallback
            return torch.zeros(3, *self.image_size)

        try:
            image = Image.open(image_path).convert('RGB')

            if self.transform_image:
                image = self.transform_image(image)

            return image

        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return black image as fallback
            return torch.zeros(3, *self.image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        output = {}

        # Load audio
        if self.load_audio:
            audio = self._load_audio(sample['audio_path'])
            output['audio'] = audio

        # Load image
        if self.load_image:
            image = self._load_image(sample['image_path'])
            output['image'] = image

        # Load text embedding if available
        if self.text_embs and sample['class_name'] in self.text_embs:
            text_emb = self.text_embs[sample['class_name']]
            output['text_emb'] = text_emb

        # Load audio embedding if available
        if self.audio_embs and sample['file_id'] in self.audio_embs:
            audio_emb = self.audio_embs[sample['file_id']]
            output['audio_emb'] = audio_emb

        # Add labels and metadata
        output.update({
            'label': torch.tensor(sample['class_label'], dtype=torch.long),
            'audio_filename': sample['audio_filename'],
            'image_filename': os.path.basename(sample['image_path']) if sample['image_path'] else '',
            'video_filename': '',  # ESC-50 doesn't have videos
            'class_name': sample['class_name'],
            'file_id': sample['file_id'],
            'sample_idx': sample['sample_idx'],
            'fold': sample['fold'],
            'caption': sample.get('caption', ''),
            'ytid': '',
            'start_second': 0,
            'end_second': self.audio_duration
        })

        return output

    def get_class_names(self) -> List[str]:
        """Get list of active class names."""
        return list(self.active_classes.keys())

    def get_class_labels(self) -> List[int]:
        """Get list of active class labels."""
        return list(self.active_classes.values())

    def get_num_classes(self) -> int:
        """Get number of active classes."""
        return len(self.active_classes)

    def get_samples_per_class(self) -> Dict[str, int]:
        """Get number of samples per class."""
        class_counts = {}
        for sample in self.samples:
            class_name = sample['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts

    def clear_cache(self):
        """Clear dataset cache."""
        if self.enable_cache and os.path.exists(self.cache_dir):
            cache_files = [f for f in os.listdir(self.cache_dir) if f.startswith('esc50_samples_')]
            for cache_file in cache_files:
                cache_path = os.path.join(self.cache_dir, cache_file)
                try:
                    os.remove(cache_path)
                    logger.info(f"Removed cache file: {cache_path}")
                except Exception as e:
                    logger.error(f"Failed to remove cache file {cache_path}: {e}")


def create_esc50_dataloader(dataset,
                           batch_size: int = 32,
                           shuffle: bool = True,
                           num_workers: int = 4,
                           pin_memory: bool = True) -> DataLoader:
    """Create a DataLoader for ESC-50 dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_esc50_collate_fn
    )


def _esc50_collate_fn(batch):
    """Custom collate function for ESC-50 batch processing."""
    # Stack tensors
    audio = torch.stack([item['audio'] for item in batch])
    image = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    # Stack text embeddings if available
    if 'text_emb' in batch[0]:
        text_embs = torch.stack([item['text_emb'] for item in batch])
    else:
        text_embs = None

    # Stack audio embeddings if available
    if 'audio_emb' in batch[0]:
        audio_embs = torch.stack([item['audio_emb'] for item in batch])
    else:
        audio_embs = None

    # Collect metadata
    metadata = {
        'class_names': [item['class_name'] for item in batch],
        'file_ids': [item['file_id'] for item in batch],
        'sample_indices': [item['sample_idx'] for item in batch],
        'folds': [item['fold'] for item in batch],
        'captions': [item['caption'] for item in batch]
    }

    output = {
        'audio': audio,
        'image': image,
        'labels': labels,
        'metadata': metadata
    }

    if text_embs is not None:
        output['text_embs'] = text_embs

    if audio_embs is not None:
        output['audio_embs'] = audio_embs

    return output
