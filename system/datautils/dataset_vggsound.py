"""
VGGSound Dataset loader for Audio2Visual tasks.
Handles multimodal data loading with caching and class filtering.
Based on VGGSound: A large-scale audio-visual dataset
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
import cv2
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VGGSoundDataset(Dataset):
    """
    VGGSound Dataset class for multimodal (audio, video) data loading.
    Supports caching, class filtering, and federated learning splits.

    VGGSound contains 200k+ audio-visual clips spanning 300+ sound classes.
    """

    def __init__(self,
                 root_dir: str = "/home/lpala/fedgfe/dataset/Audio/vggsound",
                 text_embedding_file: Optional[str] = "/home/lpala/fedgfe/dataset/Audio/vggsound_text_embs_dict.pt",
                 audio_embedding_file: Optional[str] = None,
                 selected_classes: Optional[Union[List[str], List[int]]] = None,
                 excluded_classes: Optional[Union[List[str], List[int]]] = None,
                 num_samples_per_class: Optional[int] = None,
                 split: Optional[str] = None,
                 splits_to_load: Optional[List[str]] = None,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.2,
                 split_ratio: Optional[float] = None,
                 use_official_split: bool = True,
                 stratify: bool = True,
                 node_id: Optional[int] = None,
                 enable_cache: bool = False,
                 cache_dir: str = "/tmp/vggsound_cache",
                 audio_sample_rate: int = 16000,
                 audio_duration: float = 10.0,  # VGGSound clips are 10 seconds
                 image_size: Tuple[int, int] = (224, 224),
                 video_fps: int = 25,
                 transform_audio: Optional = None,
                 transform_image: Optional = None,
                 transform_video: Optional = None,
                 load_audio: bool = True,
                 load_image: bool = False,
                 load_video: bool = False):
        """
        Initialize VGGSound dataset.

        Args:
            root_dir: Root directory of VGGSound dataset
            text_embedding_file: Path to text embeddings file
            audio_embedding_file: Path to audio embeddings file
            selected_classes: List of classes to include (str names or int labels)
            excluded_classes: List of classes to exclude (str names or int labels)
            num_samples_per_class: Maximum number of samples to load per class (None = all samples)
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
            use_official_split: Whether to use official train/test split from dataset
            stratify: Whether to use stratified sampling for train/val/test split
            node_id: Federated learning node ID for consistent splitting
            enable_cache: Whether to enable caching
            cache_dir: Directory for cache files
            audio_sample_rate: Target sample rate for audio
            audio_duration: Duration of audio clips in seconds
            image_size: Target size for images (H, W)
            video_fps: Target FPS for video
            transform_audio: Audio transform function
            transform_image: Image transform function
            transform_video: Video transform function
            load_audio: Whether to load audio data
            load_image: Whether to load image data
            load_video: Whether to load video data
        """
        self.root_dir = root_dir
        self.text_embedding_file = text_embedding_file
        self.audio_embedding_file = audio_embedding_file
        self.num_samples_per_class = num_samples_per_class
        self.split = split
        self.use_official_split = use_official_split

        # Handle legacy split_ratio parameter for backwards compatibility
        if split_ratio is not None:
            logger.warning("split_ratio is deprecated. Use train_ratio, val_ratio, test_ratio instead.")
            self.train_ratio = train_ratio if train_ratio != 0.7 else split_ratio * (1 - val_ratio)
            self.val_ratio = val_ratio
            self.test_ratio = 1 - split_ratio
            self.split_ratio = split_ratio
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
            self.split_ratio = 1 - test_ratio

        # Handle splits_to_load parameter
        self.splits_to_load = splits_to_load
        if splits_to_load is not None:
            valid_splits = {'train', 'val', 'test', 'all'}
            for s in splits_to_load:
                if s not in valid_splits:
                    raise ValueError(f"Invalid split in splits_to_load: {s}. Must be one of {valid_splits}")
            logger.info(f"Loading data only from splits: {splits_to_load}")

        self.stratify = stratify
        self.node_id = node_id
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir
        self.audio_sample_rate = audio_sample_rate
        self.audio_duration = audio_duration
        self.image_size = image_size
        self.video_fps = video_fps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load_audio = load_audio
        self.load_image = load_image
        self.load_video = load_video

        # Initialize class labels by scanning dataset
        self.CLASS_LABELS = {}
        self.LABEL_TO_CLASS = {}
        self._initialize_class_labels()

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
            logger.info(f"Loaded text embeddings for {len(self.text_embs)} classes")

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
                num_samples_per_class=num_samples_per_class,
                splits_to_load=splits_to_load,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                use_official_split=use_official_split,
                stratify=stratify,
                node_id=node_id,
                enable_cache=enable_cache,
                cache_dir=cache_dir,
                audio_sample_rate=audio_sample_rate,
                audio_duration=audio_duration,
                image_size=image_size,
                video_fps=video_fps,
                transform_audio=transform_audio,
                transform_image=transform_image,
                transform_video=transform_video,
                load_audio=load_audio,
                load_image=load_image,
                load_video=load_video
            )
            self.split = 'all'

        self._excluded_classes = None
        self._selected_classes = None

        # Process class selection
        self.selected_classes = self._process_class_selection(selected_classes)
        self.excluded_classes = self._process_class_selection(excluded_classes)

        self.available_classes = list(self.CLASS_LABELS.keys())

        # Create transforms
        self.transform_audio = transform_audio
        self.transform_image = transform_image or self._default_image_transform()
        self.transform_video = transform_video

        # Create cache directory
        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        logger.info(f"VGGSound Dataset initialized: {len(self.samples)} samples, "
                   f"classes: {len(self.active_classes)}, split: {split}")

    def _initialize_class_labels(self):
        """
        Initialize class labels by extracting from text embeddings or scanning dataset.
        VGGSound class names are extracted from the text embeddings file.
        """
        if self.text_embedding_file and os.path.exists(self.text_embedding_file):
            try:
                text_embs_temp = torch.load(self.text_embedding_file, map_location='cpu')
                class_names = sorted([k.lower().replace(' ', '_') for k in text_embs_temp.keys()])
                self.CLASS_LABELS = {cls: idx for idx, cls in enumerate(class_names)}
                self.LABEL_TO_CLASS = {v: k for k, v in self.CLASS_LABELS.items()}
                logger.info(f"Initialized {len(self.CLASS_LABELS)} classes from text embeddings")
                return
            except Exception as e:
                logger.warning(f"Failed to load classes from text embeddings: {e}")

        # Fallback: scan audio directories for class names
        logger.info("Scanning dataset for class names...")
        class_names_set = set()

        # Since VGGSound doesn't have explicit class folders, we'll create a minimal set
        # In practice, you may need metadata files mapping filenames to classes
        logger.warning("VGGSound requires metadata files to properly map classes. Using text embeddings is recommended.")
        self.CLASS_LABELS = {}
        self.LABEL_TO_CLASS = {}

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

    def get_text_embeddings(self, class_name):
        if self.text_embs is not None and class_name in self.text_embs:
            return self.text_embs[class_name]
        return None

    def _create_all_splits(self, **kwargs):
        """
        Create train, val, and test splits automatically.
        """
        logger.info("Creating train split...")
        self.train = VGGSoundDataset(
            split='train',
            **kwargs
        )

        logger.info("Creating validation split...")
        self.val = VGGSoundDataset(
            split='val',
            **kwargs
        )

        logger.info("Creating test split...")
        self.test = VGGSoundDataset(
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
                # Normalize class name
                cls_normalized = cls.lower().replace(' ', '_')
                if cls_normalized in self.CLASS_LABELS:
                    processed.append(cls_normalized)
                else:
                    logger.warning(f"Invalid class name: {cls}")

        return processed if processed else None

    def _filter_classes(self) -> Dict[str, int]:
        """
        Filter classes based on selection and exclusion.

        IMPORTANT: Labels are NOT remapped - original CLASS_LABELS indices are preserved.
        This ensures consistency across federated nodes and with conditional generators.

        Example:
            CLASS_LABELS = {'dog': 0, 'baby_laughter': 1, 'cap_gun': 2, 'car': 3, ...}
            selected_classes = ['cap_gun', 'car']
            Result: {'cap_gun': 2, 'car': 3}  # Original labels preserved, not remapped to [0, 1]
        """
        active_classes = dict(self.CLASS_LABELS)

        # Apply selection filter
        if self._selected_classes:
            active_classes = {cls: label for cls, label in active_classes.items()
                            if cls in self.selected_classes}

        # Apply exclusion filter
        if self._excluded_classes:
            active_classes = {cls: label for cls, label in active_classes.items()
                            if cls not in self.excluded_classes}

        # DO NOT remap labels - preserve original CLASS_LABELS indices
        # This is critical for federated learning where multiple nodes with different
        # class selections must use consistent label spaces

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
        cache_file = os.path.join(self.cache_dir, f"vggsound_samples_{cache_key}.pkl")

        # Try to load from cache
        if self.enable_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    samples = pickle.load(f)
                logger.info(f"Loaded {len(samples)} samples from cache")
                return samples
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        samples = []

        # Handle splits_to_load parameter
        if self.splits_to_load is not None:
            all_split_samples = []
            for split_name in self.splits_to_load:
                if split_name == 'all':
                    temp_split = 'all'
                else:
                    temp_split = split_name

                original_split = self.split
                self.split = temp_split

                split_samples = self._load_samples_from_split()
                all_split_samples.extend(split_samples)

                self.split = original_split

            return all_split_samples

        # Normal loading (single split)
        samples = self._load_samples_from_split()

        # Apply train/val/test split if not using official split
        if not self.use_official_split and self.split != 'all':
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

    def _load_samples_from_split(self) -> List[Dict]:
        """
        Load samples from specified split directory.
        """
        samples = []

        # Determine which directories to scan
        if self.use_official_split:
            if self.split == 'train':
                split_dirs = ['train']
            elif self.split in ['val', 'test']:
                # Official VGGSound has train/test, we'll split test for val
                split_dirs = ['test']
            else:  # 'all'
                split_dirs = ['train', 'test']
        else:
            # Load all and split later
            split_dirs = ['train', 'test']

        for split_dir in split_dirs:
            audio_dir = os.path.join(self.root_dir, split_dir, 'audios')
            video_dir = os.path.join(self.root_dir, split_dir, 'video')

            if not os.path.exists(audio_dir):
                logger.warning(f"Audio directory not found: {audio_dir}")
                continue

            # List all audio files
            audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])

            # Apply num_samples_per_class limit if specified
            class_sample_count = {}

            for audio_filename in audio_files:
                # Parse filename: {ytid}_{start_time}.wav
                file_id = audio_filename.replace('.wav', '')
                parts = file_id.split('_')

                if len(parts) < 2:
                    continue

                ytid = '_'.join(parts[:-1])
                start_second = int(parts[-1])

                # For VGGSound, we need metadata to map files to classes
                # As a workaround, we'll assign a generic class or skip class filtering
                # This requires proper metadata files in production

                # Placeholder: assign to first class if no metadata
                # In practice, you should have a CSV/JSON mapping ytid -> class_name
                class_name = None
                if self.active_classes:
                    class_name = list(self.active_classes.keys())[0]  # Placeholder
                else:
                    continue

                # Check class sample limit
                if self.num_samples_per_class is not None:
                    count = class_sample_count.get(class_name, 0)
                    if count >= self.num_samples_per_class:
                        continue
                    class_sample_count[class_name] = count + 1

                # Build paths
                audio_path = os.path.join(audio_dir, audio_filename)
                video_path = os.path.join(video_dir, audio_filename.replace('.wav', '.mp4'))

                # Check if files exist
                if os.path.exists(audio_path):
                    sample = {
                        'class_name': class_name,
                        'class_label': self.active_classes[class_name],
                        'original_class_label': self.CLASS_LABELS.get(class_name, 0),
                        'audio_filename': audio_filename,
                        'video_filename': audio_filename.replace('.wav', '.mp4'),
                        'file_id': file_id,
                        'ytid': ytid,
                        'start_second': start_second,
                        'audio_path': audio_path,
                        'video_path': video_path if os.path.exists(video_path) else None,
                        'split_dir': split_dir,
                        'sample_idx': len(samples),
                        'caption': self.captions.get(file_id, '')
                    }
                    samples.append(sample)

        logger.info(f"Loaded {len(samples)} samples from {split_dirs}")
        return samples

    def _get_cache_key(self) -> str:
        """Generate cache key based on dataset configuration."""
        config_str = (f"{self.root_dir}_{self.active_classes}_{self.split}_{self.split_ratio}_"
                     f"{self.use_official_split}_{self.node_id}_{self.num_samples_per_class}")
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _apply_split(self, samples: List[Dict]) -> List[Dict]:
        """
        Apply train/val/test split to samples with optional stratification.

        Notes:
        - train split is always present
        - val split is optional (val_ratio can be 0)
        - test split is optional (test_ratio can be 0)
        """
        if not samples:
            return samples

        # Use reproducible random seed based on node_id
        random_state = 42 + (self.node_id or 0)

        # Extract labels for stratification
        labels = [sample['class_label'] for sample in samples]

        # Validate and clean ratios to avoid floating point errors
        # sklearn requires test_size in range (0.0, 1.0) or None
        epsilon = 1e-6  # Tolerance for floating point comparison

        # Clamp ratios to valid range [0.0, 1.0]
        test_ratio_clean = max(0.0, min(1.0, self.test_ratio))
        val_ratio_clean = max(0.0, min(1.0, self.val_ratio))
        train_ratio_clean = max(0.0, min(1.0, self.train_ratio))

        # If test_ratio is effectively zero (< epsilon), treat as no test split
        has_test_split = test_ratio_clean >= epsilon
        has_val_split = val_ratio_clean >= epsilon

        if self.stratify:
            # Stratified split using sklearn with custom ratios
            if self.split in ['train', 'val']:
                # First split: train+val vs test using test_ratio
                if has_test_split:
                    train_val_samples, test_samples = train_test_split(
                        samples,
                        test_size=test_ratio_clean,
                        stratify=labels if self.stratify else None,
                        random_state=random_state
                    )
                else:
                    # No test split - all samples go to train+val
                    train_val_samples = samples
                    test_samples = []

                # If we need validation split
                if has_val_split:
                    # Second split: train vs val
                    train_val_labels = [s['class_label'] for s in train_val_samples]
                    # Calculate val size relative to train_val set
                    total_train_val = train_ratio_clean + val_ratio_clean
                    if total_train_val > epsilon:
                        val_size_relative = val_ratio_clean / total_train_val
                        # Ensure val_size_relative is in valid range
                        val_size_relative = max(epsilon, min(1.0 - epsilon, val_size_relative))

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
                        # Invalid ratios, return all samples for train
                        if self.split == 'train':
                            return train_val_samples
                        elif self.split == 'val':
                            return []
                else:
                    # No validation split needed
                    if self.split == 'train':
                        return train_val_samples
                    elif self.split == 'val':
                        # Val requested but val_ratio is 0
                        return []

            elif self.split == 'test':
                # Split to get test set using test_ratio
                if not has_test_split:
                    # No test split configured
                    return []

                _, test_samples = train_test_split(
                    samples,
                    test_size=test_ratio_clean,
                    stratify=labels if self.stratify else None,
                    random_state=random_state
                )
                return test_samples

        else:
            # Non-stratified split with custom ratios
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

                if has_val_split:
                    # Three-way split: train / val / test
                    n_train = int(len(cls_samples) * train_ratio_clean)
                    n_val = int(len(cls_samples) * val_ratio_clean)

                    if self.split == 'train':
                        split_samples.extend(cls_samples[:n_train])
                    elif self.split == 'val':
                        split_samples.extend(cls_samples[n_train:n_train + n_val])
                    elif self.split == 'test':
                        if has_test_split:
                            split_samples.extend(cls_samples[n_train + n_val:])
                        # else: return empty list for test when test_ratio=0
                else:
                    # Two-way split: train / test
                    n_train = int(len(cls_samples) * train_ratio_clean)

                    if self.split == 'train':
                        split_samples.extend(cls_samples[:n_train])
                    elif self.split == 'val':
                        # Val requested but val_ratio is 0
                        pass  # Return empty split_samples
                    elif self.split == 'test':
                        if has_test_split:
                            split_samples.extend(cls_samples[n_train:])
                        # else: return empty list for test when test_ratio=0

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

    def _load_video(self, video_path: str) -> torch.Tensor:
        """Load and preprocess video."""
        if video_path is None or not os.path.exists(video_path):
            # Return black frames as fallback
            num_frames = int(self.audio_duration * self.video_fps)
            return torch.zeros(num_frames, 3, *self.image_size)

        try:
            cap = cv2.VideoCapture(video_path)
            frames = []

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Calculate frame sampling
            target_frames = int(self.audio_duration * self.video_fps)
            frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)

            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()

                if not ret:
                    break

                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)

                # Apply transforms
                if self.transform_image:
                    frame = self.transform_image(frame)
                else:
                    frame = self._default_image_transform()(frame)

                frames.append(frame)

            cap.release()

            # Pad if needed
            while len(frames) < target_frames:
                frames.append(torch.zeros(3, *self.image_size))

            return torch.stack(frames[:target_frames])

        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            # Return black frames as fallback
            num_frames = int(self.audio_duration * self.video_fps)
            return torch.zeros(num_frames, 3, *self.image_size)

    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image (single frame from video)."""
        if image_path is None or not os.path.exists(image_path):
            return torch.zeros(3, *self.image_size)

        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform_image:
                image = self.transform_image(image)
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
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

        # Load video
        if self.load_video:
            video = self._load_video(sample['video_path'])
            output['video'] = video

        # Load image (extract from video if needed)
        if self.load_image:
            # Extract middle frame from video if available
            if self.load_video and 'video' in output:
                mid_frame = len(output['video']) // 2
                output['image'] = output['video'][mid_frame]
            else:
                # Load first frame from video
                output['image'] = self._load_image(sample['video_path'])

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
            'video_filename': sample.get('video_filename', ''),
            'image_filename': '',
            'class_name': sample['class_name'],
            'file_id': sample['file_id'],
            'sample_idx': sample['sample_idx'],
            'caption': sample.get('caption', ''),
            'ytid': sample['ytid'],
            'start_second': sample['start_second'],
            'end_second': sample['start_second'] + self.audio_duration
        })

        return output

    def get_class_names(self) -> List[str]:
        """Get list of active class names."""
        return list(self.active_classes.keys())

    def get_class_labels(self) -> List[int]:
        """Get list of active class labels."""
        return list(self.active_classes.values())

    def get_num_classes(self) -> int:
        """
        Get number of active classes.

        NOTE: This returns the COUNT of active classes, not the maximum label value.
        For the maximum label index needed for model output dimensions, use get_max_class_label().

        Example:
            selected_classes = ['cap_gun', 'car']  # labels 2, 3
            get_num_classes() -> 2 (count of classes)
            get_max_class_label() -> 3 (max label value)
        """
        return len(self.active_classes)

    def get_max_class_label(self) -> int:
        """
        Get the maximum class label value in active classes.

        This is useful for determining model output dimensions when labels are not remapped.
        Since VGGSound preserves original CLASS_LABELS indices, this may differ from get_num_classes().

        Returns:
            Maximum label value, or -1 if no active classes

        Example:
            selected_classes = ['cap_gun', 'car']  # original labels: cap_gun=2, car=3
            get_max_class_label() -> 3

            all_classes (309 classes total)
            get_max_class_label() -> 308
        """
        if not self.active_classes:
            return -1
        return max(self.active_classes.values())

    def get_collate_fn(self):
        return _vggsound_collate_fn

    def get_samples_per_class(self) -> Dict[str, int]:
        """Get number of samples per class."""
        class_counts = {}
        for sample in self.samples:
            class_name = sample['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts

    def get_class_distribution(self) -> Dict[str, float]:
        """Get class distribution as percentages."""
        counts = self.get_samples_per_class()
        total = sum(counts.values())
        return {cls: (count / total) * 100 for cls, count in counts.items()}

    def print_split_statistics(self):
        """Print statistics about the current split."""
        logger.info(f"\n=== VGGSound Dataset Statistics ({self.split} split) ===")
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

    def verify_stratification(self, other_dataset: 'VGGSoundDataset', tolerance: float = 0.05) -> bool:
        """Verify that class distributions are similar between datasets."""
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
            cache_files = [f for f in os.listdir(self.cache_dir) if f.startswith('vggsound_samples_')]
            for cache_file in cache_files:
                cache_path = os.path.join(self.cache_dir, cache_file)
                try:
                    os.remove(cache_path)
                    logger.info(f"Removed cache file: {cache_path}")
                except Exception as e:
                    logger.error(f"Failed to remove cache file {cache_path}: {e}")


def create_vggsound_dataloader(dataset,
                               batch_size: int = 32,
                               shuffle: bool = True,
                               num_workers: int = 4,
                               pin_memory: bool = True) -> DataLoader:
    """Create a DataLoader for VGGSound dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_vggsound_collate_fn
    )


def _vggsound_collate_fn(batch):
    """Custom collate function for VGGSound batch processing."""
    # Stack tensors
    audio = torch.stack([item['audio'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    # Stack video if available
    if 'video' in batch[0]:
        video = torch.stack([item['video'] for item in batch])
    else:
        video = None

    # Stack image if available
    if 'image' in batch[0]:
        image = torch.stack([item['image'] for item in batch])
    else:
        image = None

    # Stack text embeddings if available
    if 'text_emb' in batch[0]:
        text_embs = [item['text_emb'] for item in batch]
    else:
        text_embs = None

    # Stack audio embeddings if available
    if 'audio_emb' in batch[0]:
        audio_emb = torch.stack([item['audio_emb'] for item in batch])
    else:
        audio_emb = None

    # Collect metadata
    metadata = {
        'class_names': [item['class_name'] for item in batch],
        'file_ids': [item['file_id'] for item in batch],
        'sample_indices': [item['sample_idx'] for item in batch],
        'ytids': [item['ytid'] for item in batch],
        'start_seconds': [item['start_second'] for item in batch],
        'captions': [item['caption'] for item in batch]
    }

    output = {
        'audio': audio,
        'labels': labels,
        'metadata': metadata,
        'class_name': metadata['class_names'],
        'file_id': metadata['file_ids'],
        'audio_filename': [item['audio_filename'] for item in batch],
        'video_filename': [item['video_filename'] for item in batch]
    }

    if video is not None:
        output['video'] = video

    if image is not None:
        output['image'] = image

    if text_embs is not None:
        output['text_emb'] = text_embs

    if audio_emb is not None:
        output['audio_emb'] = audio_emb

    return output
