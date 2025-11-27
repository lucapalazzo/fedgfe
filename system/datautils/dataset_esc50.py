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
from sklearn.model_selection import train_test_split

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
                 split: Optional[str] = None,
                 splits_to_load: Optional[List[str]] = None,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.2,
                 split_ratio: Optional[float] = None,
                 use_folds: bool = False,
                 train_folds: List[int] = [0, 1, 2, 3],
                 val_folds: Optional[List[int]] = None,
                 test_folds: List[int] = [4],
                 stratify: bool = True,
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
            text_embedding_file: Path to text embeddings file
            audio_embedding_file: Path to audio embeddings file
            selected_classes: List of classes to include (str names or int labels)
            excluded_classes: List of classes to exclude (str names or int labels)
            split: 'train', 'val', 'test', 'all', or None (default).
                  - None: Auto-creates train/val/test splits accessible as .train, .val, .test
                  - 'train'/'val'/'test': Returns only specified split
                  - 'all': Returns all samples together (legacy behavior)
            splits_to_load: List of splits to load data from (e.g., ['train', 'val']).
                          If None, loads based on 'split' parameter
            train_ratio: Ratio for training split (default 0.7 = 70%)
            val_ratio: Ratio for validation split (default 0.1 = 10%)
            test_ratio: Ratio for test split (default 0.2 = 20%)
            split_ratio: [DEPRECATED] Legacy parameter, use train_ratio instead.
                        If provided, sets train+val ratio (1-test_ratio)
            use_folds: Whether to use official fold splits
            train_folds: List of fold indices for training (0-4)
            val_folds: List of fold indices for validation (0-4), if None uses val_ratio from train
            test_folds: List of fold indices for testing (0-4)
            stratify: Whether to use stratified sampling for train/val/test split
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
        self.text_embedding_file = text_embedding_file
        self.audio_embedding_file = audio_embedding_file
        self.split = split

        # Handle legacy split_ratio parameter for backwards compatibility
        if split_ratio is not None:
            logger.warning("split_ratio is deprecated. Use train_ratio, val_ratio, test_ratio instead.")
            # Convert old split_ratio to new format
            # old: split_ratio=0.8 meant 80% train+val, 20% test
            self.train_ratio = train_ratio if train_ratio != 0.7 else split_ratio * (1 - val_ratio)
            self.val_ratio = val_ratio
            self.test_ratio = 1 - split_ratio
            self.split_ratio = split_ratio  # Keep for compatibility
        else:
            # Validate that ratios sum to ~1.0
            total_ratio = train_ratio + val_ratio + test_ratio
            if not (0.99 <= total_ratio <= 1.01):
                logger.warning(f"train_ratio + val_ratio + test_ratio = {total_ratio:.3f}, normalizing to 1.0")
                self.train_ratio = train_ratio / total_ratio
                self.val_ratio = val_ratio / total_ratio
                self.test_ratio = test_ratio / total_ratio
            else:
                self.train_ratio = train_ratio
                self.val_ratio = val_ratio
                self.test_ratio = test_ratio
            # Set split_ratio for backwards compatibility with _apply_split logic
            self.split_ratio = 1 - test_ratio

        # Handle splits_to_load parameter
        self.splits_to_load = splits_to_load
        if splits_to_load is not None:
            # Validate splits_to_load
            valid_splits = {'train', 'val', 'test', 'all'}
            for s in splits_to_load:
                if s not in valid_splits:
                    raise ValueError(f"Invalid split in splits_to_load: {s}. Must be one of {valid_splits}")
            logger.info(f"Loading data only from splits: {splits_to_load}")

        self.stratify = stratify
        self.use_folds = use_folds
        self.train_folds = train_folds
        self.val_folds = val_folds
        self.test_folds = test_folds
        self.node_id = node_id
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir
        self.audio_sample_rate = audio_sample_rate
        self.audio_duration = audio_duration
        self.image_size = image_size

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load_audio = True
        self.load_image = False

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
            self.text_embs_from_file = torch.load(text_embedding_file, map_location=self.device)
            self.text_embs = {k.lower(): v for k, v in self.text_embs_from_file.items()}

        # Load audio embeddings if provided
        self.audio_embs = None
        if audio_embedding_file and os.path.exists(audio_embedding_file):
            self.audio_embs = torch.load(audio_embedding_file, map_location=self.device)

        # Handle auto-split creation when split=None
        if split is None:
            logger.info("Auto-creating train/val/test splits (split=None)")
            self._create_all_splits(
                root_dir=root_dir,
                text_embedding_file=text_embedding_file,
                audio_embedding_file=audio_embedding_file,
                selected_classes=selected_classes,
                excluded_classes=excluded_classes,
                splits_to_load=splits_to_load,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                use_folds=use_folds,
                train_folds=train_folds,
                val_folds=val_folds,
                test_folds=test_folds,
                stratify=stratify,
                node_id=node_id,
                enable_cache=enable_cache,
                cache_dir=cache_dir,
                audio_sample_rate=audio_sample_rate,
                audio_duration=audio_duration,
                image_size=image_size,
                transform_audio=transform_audio,
                transform_image=transform_image
            )
            # For split=None, dataset itself contains all samples (train+val+test combined)
            # Individual splits are accessible via .train, .val, .test attributes
            self.split = 'all'  # Internally treat as 'all' for combined access

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

    def to(self, device: torch.device):
        self.device = device
        for sample in self.samples:
            if 'text_emb' in sample and sample['text_emb'] is not None:
                sample['text_emb'] = sample['text_emb'].to(device)
            if 'audio_emb' in sample and sample['audio_emb'] is not None:
                sample['audio_emb'] = sample['audio_emb'].to(device)
            if 'audio' in sample and sample['audio'] is not None:
                sample['audio'] = sample['audio'].to(device)
        return self

    def get_text_embeddings ( self, class_name ):
        if self.text_embs is not None and class_name in self.text_embs:
            return self.text_embs[class_name]
        return None

    def _create_all_splits(self, **kwargs):
        """
        Create train, val, and test splits automatically.

        This method is called when split=None to auto-create all three splits.
        The splits are exposed as .train, .val, and .test attributes.

        Args:
            **kwargs: All dataset initialization parameters
        """
        logger.info("Creating train split...")
        self.train = ESC50Dataset(
            split='train',
            **kwargs
        )

        logger.info("Creating validation split...")
        self.val = ESC50Dataset(
            split='val',
            **kwargs
        )


        logger.info("Creating test split...")
        self.test = ESC50Dataset(
            split='test',
            **kwargs
        )

        logger.info(f"Auto-split created: train={len(self.train)}, val={len(self.val)}, test={len(self.test)}")

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

        # Handle splits_to_load parameter
        if self.splits_to_load is not None:
            # Load data from multiple specified splits and combine them
            all_split_samples = []

            for split_name in self.splits_to_load:
                if split_name == 'all':
                    # Load all data
                    temp_split = 'all'
                else:
                    temp_split = split_name

                # Temporarily set split to load specific data
                original_split = self.split
                self.split = temp_split

                # Determine folds to use for this split
                if self.use_folds:
                    if temp_split == 'train':
                        folds_to_use = self.train_folds
                    elif temp_split == 'val':
                        folds_to_use = self.val_folds if self.val_folds else self.train_folds
                    elif temp_split == 'test':
                        folds_to_use = self.test_folds
                    else:  # 'all'
                        folds_to_use = [0, 1, 2, 3, 4]
                else:
                    # Will load all and split later
                    folds_to_use = [0, 1, 2, 3, 4]

                # Load samples for this split
                split_samples = self._load_samples_from_folds(folds_to_use)

                # Apply split if not using folds
                if not self.use_folds and temp_split != 'all':
                    split_samples = self._apply_split(split_samples)

                all_split_samples.extend(split_samples)

                # Restore original split
                self.split = original_split

            return all_split_samples

        # Normal loading (single split)
        if self.use_folds:
            # Use official fold splits
            if self.split == 'train':
                folds_to_use = self.train_folds
            elif self.split == 'val':
                # Use validation folds if specified, otherwise will split from train
                if self.val_folds:
                    folds_to_use = self.val_folds
                else:
                    # Will use train folds and split later
                    folds_to_use = self.train_folds
            elif self.split == 'test':
                folds_to_use = self.test_folds
            else:  # 'all'
                folds_to_use = [0, 1, 2, 3, 4]
        else:
            # Use all folds and split later
            folds_to_use = [0, 1, 2, 3, 4]

        samples = self._load_samples_from_folds(folds_to_use)

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

    def _load_samples_from_folds(self, folds_to_use: List[int]) -> List[Dict]:
        """
        Load samples from specified folds.

        Args:
            folds_to_use: List of fold indices to load

        Returns:
            List of sample dictionaries
        """
        samples = []

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

        return samples

    def _get_cache_key(self) -> str:
        """Generate cache key based on dataset configuration."""
        config_str = (f"{self.root_dir}_{self.active_classes}_{self.split}_{self.split_ratio}_"
                     f"{self.use_folds}_{self.train_folds}_{self.test_folds}_{self.node_id}")
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _apply_split(self, samples: List[Dict]) -> List[Dict]:
        """
        Apply train/val/test split to samples with optional stratification.

        Uses scikit-learn's train_test_split for stratified sampling.
        Supports custom train_ratio, val_ratio, test_ratio (default 70-10-20).
        """
        if not samples:
            return samples

        # Use reproducible random seed based on node_id
        random_state = 42 + (self.node_id or 0)

        # Extract labels for stratification
        labels = [sample['class_label'] for sample in samples]

        if self.stratify:
            # Stratified split using sklearn with custom ratios
            if self.split in ['train', 'val']:
                # First split: train+val vs test using test_ratio
                if (self.test_ratio > 0):
                    train_val_samples, test_samples = train_test_split(
                        samples,
                        test_size=self.test_ratio,
                        stratify=labels if self.stratify else None,
                        random_state=random_state
                    )
                else:
                    train_val_samples = samples
                    test_samples = []

                # If we need validation split
                if self.val_ratio > 0 and not (self.use_folds and self.val_folds):
                    # Second split: train vs val
                    train_val_labels = [s['class_label'] for s in train_val_samples]
                    # Calculate val size relative to train_val set
                    # val_ratio is relative to total, so we need to adjust
                    val_size_relative = self.val_ratio / (self.train_ratio + self.val_ratio)

                    train_samples, val_samples = train_test_split(
                        train_val_samples,
                        test_size=val_size_relative,
                        stratify=train_val_labels if self.stratify else None,
                        random_state=random_state
                    )

                    if self.split == 'train':
                        return train_samples
                    elif self.split == 'val':
                        return val_samples
                else:
                    # No validation split needed
                    if self.split == 'train':
                        return train_val_samples

            elif self.split == 'test':
                # Split to get test set using test_ratio
                if (self.test_ratio == 0):
                    return []
                
                _, test_samples = train_test_split(
                    samples,
                    test_size=self.test_ratio,
                    stratify=labels if self.stratify else None,
                    random_state=random_state
                )
                return test_samples

        else:
            # Non-stratified split (original behavior) with custom ratios
            rng = np.random.RandomState(random_state)

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

                if self.val_ratio > 0 and not (self.use_folds and self.val_folds):
                    # Three-way split: train / val / test using custom ratios
                    n_train = int(len(cls_samples) * self.train_ratio)
                    n_val = int(len(cls_samples) * self.val_ratio)

                    if self.split == 'train':
                        split_samples.extend(cls_samples[:n_train])
                    elif self.split == 'val':
                        split_samples.extend(cls_samples[n_train:n_train + n_val])
                    elif self.split == 'test':
                        split_samples.extend(cls_samples[n_train + n_val:])
                else:
                    # Two-way split: train / test using custom ratios
                    n_train = int(len(cls_samples) * self.train_ratio)

                    if self.split == 'train':
                        split_samples.extend(cls_samples[:n_train])
                    elif self.split == 'test':
                        split_samples.extend(cls_samples[n_train:])

            return split_samples

        return samples

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
        if 'audio' not in sample:
            sample['audio'] = None
            if self.load_audio:
                audio = self._load_audio(sample['audio_path'])
                sample['audio'] = audio
        output['audio'] = sample['audio']

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

    def get_class_distribution(self) -> Dict[str, float]:
        """
        Get class distribution as percentages.

        Returns:
            Dictionary mapping class names to their percentage in the dataset
        """
        counts = self.get_samples_per_class()
        total = sum(counts.values())
        return {cls: (count / total) * 100 for cls, count in counts.items()}

    def print_split_statistics(self):
        """Print statistics about the current split."""
        logger.info(f"\n=== ESC-50 Dataset Statistics ({self.split} split) ===")
        logger.info(f"Total samples: {len(self.samples)}")
        logger.info(f"Number of classes: {len(self.active_classes)}")
        logger.info(f"Stratified: {self.stratify}")

        samples_per_class = self.get_samples_per_class()
        distribution = self.get_class_distribution()

        logger.info("\nClass distribution:")
        for class_name in sorted(self.active_classes.keys()):
            count = samples_per_class.get(class_name, 0)
            percent = distribution.get(class_name, 0.0)
            logger.info(f"  {class_name}: {count} samples ({percent:.2f}%)")

        if self.use_folds:
            folds_used = []
            if self.split == 'train':
                folds_used = self.train_folds
            elif self.split == 'val' and self.val_folds:
                folds_used = self.val_folds
            elif self.split == 'test':
                folds_used = self.test_folds
            logger.info(f"\nUsing folds: {folds_used}")

    def verify_stratification(self, other_dataset: 'ESC50Dataset', tolerance: float = 0.05) -> bool:
        """
        Verify that class distributions are similar between this and another dataset.

        Args:
            other_dataset: Another ESC50Dataset to compare with
            tolerance: Maximum allowed difference in class percentages (default 5%)

        Returns:
            True if distributions are similar within tolerance
        """
        dist1 = self.get_class_distribution()
        dist2 = other_dataset.get_class_distribution()

        all_classes = set(dist1.keys()) | set(dist2.keys())

        is_stratified = True
        logger.info("\n=== Stratification Verification ===")

        for class_name in sorted(all_classes):
            pct1 = dist1.get(class_name, 0.0)
            pct2 = dist2.get(class_name, 0.0)
            diff = abs(pct1 - pct2)

            status = "✓" if diff <= tolerance * 100 else "✗"
            logger.info(f"{status} {class_name}: {pct1:.2f}% vs {pct2:.2f}% (diff: {diff:.2f}%)")

            if diff > tolerance * 100:
                is_stratified = False

        logger.info(f"\nStratification {'PASSED' if is_stratified else 'FAILED'} (tolerance: {tolerance*100:.1f}%)")
        return is_stratified

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
