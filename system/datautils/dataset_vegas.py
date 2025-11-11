"""
VEGAS Dataset loader for Audio2Visual tasks.
Handles multimodal data loading with caching and class filtering.
"""

from genericpath import isfile
import os
import pickle
import hashlib
import pandas as pd
import numpy as np
import torch
import torchaudio
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union
import cv2
import logging
import sys
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class VEGASDataset(Dataset):
    """
    VEGAS Dataset class for multimodal (audio, video, image) data loading.
    Supports caching, class filtering, and federated learning splits.
    """

    # Class labels mapping
    CLASS_LABELS = {
        'baby_cry': 0,
        'chainsaw': 1,
        'dog': 2,
        'drum': 3,
        'fireworks': 4,
        'helicopter': 5,
        'printer': 6,
        'rail_transport': 7,
        'snoring': 8,
        'water_flowing': 9
    }

    LABEL_TO_CLASS = {v: k for k, v in CLASS_LABELS.items()}

    def __init__(self,
                 root_dir: str = "/home/lpala/fedgfe/dataset/Audio/VEGAS",
                 embedding_file = "/home/lpala/fedgfe/dataset/Audio/vegas_text_embs_dict.pt",
                 audio_embedding_file = None,
                 selected_classes: Optional[Union[List[str], List[int]]] = None,
                 excluded_classes: Optional[Union[List[str], List[int]]] = None,
                 split: str = "all",
                 split_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 stratify: bool = True,
                 use_saved_audio_embeddings: bool = True,
                 node_id: Optional[int] = None,
                 enable_cache: bool = False,
                 cache_dir: str = "/tmp/vegas_cache",
                 audio_sample_rate: int = 16000,
                 audio_duration: float = 10.0,
                 image_size: Tuple[int, int] = (224, 224),
                 video_fps: int = 25,
                 transform_audio: Optional = None,
                 transform_image: Optional = None,
                 transform_video: Optional = None,
                 load_audio: bool = True,
                 load_image: bool = False,
                 load_video: bool = False):
        """
        Initialize VEGAS dataset.

        Args:
            root_dir: Root directory of VEGAS dataset
            embedding_file: Path to text embeddings file
            audio_embedding_file: Path to audio embeddings file
            selected_classes: List of classes to include (str names or int labels)
            excluded_classes: List of classes to exclude (str names or int labels)
            split: 'train', 'val', 'test', or 'all'
            split_ratio: Ratio for train split
            val_ratio: Ratio for validation split from train set
            stratify: Whether to use stratified sampling for train/val/test split
            use_saved_audio_embeddings: Whether to use saved audio embeddings
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
        self.embedding_file = embedding_file
        self.audio_embedding_file = audio_embedding_file
        self.split = split
        self.split_ratio = split_ratio
        self.val_ratio = val_ratio
        self.stratify = stratify
        self.node_id = node_id
        self.enable_cache = enable_cache
        self.cache_dir = cache_dir
        self.audio_sample_rate = audio_sample_rate
        self.audio_duration = audio_duration
        self.image_size = image_size
        self.video_fps = video_fps

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.load_video = load_video
        self.load_image = load_image
        self.load_audio = load_audio
        self.text_embs = None
        if isfile(self.embedding_file):
            self.text_embs = torch.load(embedding_file, map_location=self.device)
            # convert keyes to lowercase
            self.text_embs = {k.lower(): v for k, v in self.text_embs.items()}
        
        self.audio_embs_from_file = None
        self._audio_embs = {}
        
        self._excluded_classes = None
        self._selected_classes = None

        # Process class selection
        self.selected_classes = self._process_class_selection(selected_classes)
        self.excluded_classes = self._process_class_selection(excluded_classes)

        self.available_classes = list(self.CLASS_LABELS.keys())

        # Load captions if available
        self.captions = {}
        captions_path = os.path.join(self.root_dir, "captions.json")
        if os.path.exists(captions_path):
            import json
            with open(captions_path, 'r') as f:
                self.captions = json.load(f)

        self.transform_audio = transform_audio
        self.transform_image = transform_image or self._default_image_transform()
        self.transform_video = transform_video

        self.use_saved_audio_embeddings = use_saved_audio_embeddings

        # Create cache directory
        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Load dataset metadata
        # self.samples = self._load_samples()
        logger.info(f"VEGAS Dataset initialized: {len(self.samples)} samples, "
                   f"classes: {list(self.active_classes.keys())}, split: {split}")

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

    @property
    def audio_embs(self) -> Dict[str, Dict]:
        return self._audio_embs
    
    @audio_embs.setter
    def audio_embs(self, value: Dict[str, Dict]):
        self._audio_embs = value

    def load_audio_embeddings_from_file(self, embedding_file: str):
        if isfile(embedding_file):
            self.audio_embedding_file = embedding_file
            self.audio_embs_from_file = torch.load(embedding_file, map_location=torch.device('cpu'),weights_only=False, mmap=True)
            logger.info(f"Loaded audio embeddings from {embedding_file}, total: {len(self.audio_embs)}")
        else:
            logger.warning(f"Audio embedding file not found: {embedding_file}")
            # filter audio embeddings based on selected/excluded classes
            
    def filter_audio_embeddings_from_file(self):
        self.audio_embs = {}
        if self.audio_embs_from_file is None:
            logger.warning("Audio embeddings not loaded from file.")
            return  
        for audio_filename, audio_data in self.audio_embs_from_file.items():
                if 'class_name' in audio_data and  audio_data['class_name'] in self.active_classes:
                    self.audio_embs[audio_filename] = audio_data
        self._load_samples()
        logger.info(f"Audio embedding filtered: {len(self.audio_embs)} samples after filtering")
    
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
        cache_file = os.path.join(self.cache_dir, f"vegas_samples_{cache_key}.pkl")

        # Try to load from cache
        if self.enable_cache and os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    samples = pickle.load(f)
                logger.info(f"Loaded {len(samples)} samples from cache")
                return samples
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")

        # Load samples from disk
        samples = []

        for class_name in self.active_classes.keys():
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                logger.warning(f"Class directory not found: {class_dir}")
                continue

            # Load video info
            video_info_path = os.path.join(class_dir, "video_info.csv")
            if os.path.exists(video_info_path):
                df = pd.read_csv(video_info_path)
            else:
                # If no video_info.csv, create dummy metadata
                audio_files = [f for f in os.listdir(os.path.join(class_dir, "audios"))
                              if f.endswith('.wav')]
                df = pd.DataFrame({
                    'YTID': [f.replace('.wav', '').replace('video_', '') for f in audio_files],
                    'start_second': [0] * len(audio_files),
                    'end_second': [self.audio_duration] * len(audio_files)
                })

            # Process each sample
            for idx, row in df.iterrows():
                video_id = f"video_{idx:05d}"

                # Check if all modalities exist
                audio_path = os.path.join(class_dir, "audios", f"{video_id}.wav")
                image_path = os.path.join(class_dir, "img", f"{video_id}.jpg")
                video_path = os.path.join(class_dir, "videos", f"{video_id}.mp4")

                if os.path.exists(audio_path) and os.path.exists(image_path):
                    audio_filename = os.path.basename(audio_path)
                    video_filename = os.path.basename(video_path) if os.path.exists(video_path) else ''
                    image_filename = os.path.basename(image_path)

                    sample = {
                        'class_name': class_name,
                        'class_label': self.active_classes[class_name],
                        'video_id': video_id,
                        'audio_path': audio_path,
                        'audio_filename': audio_filename,
                        'image_path': image_path,
                        'image_filename': image_filename,
                        'video_path': video_path if os.path.exists(video_path) else None,
                        'video_filename': video_filename,
                        'ytid': row.get('YTID', ''),
                        'start_second': row.get('start_second', 0),
                        'end_second': row.get('end_second', self.audio_duration),
                        'sample_idx': len(samples),
                        'audio_emb': self.audio_embs.get(audio_filename, None),
                        'caption': self.captions.get(video_id, '')
                    }
                    samples.append(sample)

        # Apply train/test split
        if self.split != 'all':
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
        config_str = f"{self.root_dir}_{self.active_classes}_{self.split}_{self.split_ratio}_{self.node_id}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _apply_split(self, samples: List[Dict]) -> List[Dict]:
        """
        Apply train/val/test split to samples with optional stratification.

        Uses scikit-learn's train_test_split for stratified sampling.
        """
        if not samples:
            return samples

        # Use reproducible random seed based on node_id
        random_state = 42 + (self.node_id or 0)

        # Extract labels for stratification
        labels = [sample['class_label'] for sample in samples]

        if self.stratify:
            # Stratified split using sklearn
            if self.split in ['train', 'val']:
                # First split: train+val vs test
                train_val_samples, test_samples = train_test_split(
                    samples,
                    test_size=1.0 - self.split_ratio,
                    stratify=labels if self.stratify else None,
                    random_state=random_state
                )

                # If we need validation split
                if self.val_ratio > 0:
                    # Second split: train vs val
                    train_val_labels = [s['class_label'] for s in train_val_samples]
                    # Calculate val size relative to train_val set
                    val_size_relative = self.val_ratio / self.split_ratio

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
                # Split to get test set
                _, test_samples = train_test_split(
                    samples,
                    test_size=1.0 - self.split_ratio,
                    stratify=labels if self.stratify else None,
                    random_state=random_state
                )
                return test_samples

        else:
            # Non-stratified split (original behavior)
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
                cls_samples = sorted(cls_samples, key=lambda x: x['video_id'])
                rng.shuffle(cls_samples)

                if self.val_ratio > 0:
                    # Three-way split: train / val / test
                    n_train = int(len(cls_samples) * self.split_ratio)
                    n_val = int(len(cls_samples) * self.val_ratio)

                    if self.split == 'train':
                        split_samples.extend(cls_samples[:n_train])
                    elif self.split == 'val':
                        split_samples.extend(cls_samples[n_train:n_train + n_val])
                    elif self.split == 'test':
                        split_samples.extend(cls_samples[n_train + n_val:])
                else:
                    # Two-way split: train / test
                    n_train = int(len(cls_samples) * self.split_ratio)

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
        try:
            image = Image.open(image_path).convert('RGB')

            if self.transform_image:
                image = self.transform_image(image)

            return image

        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return black image as fallback
            return torch.zeros(3, *self.image_size)

    def _load_video(self, video_path: str) -> Optional[torch.Tensor]:
        """Load and preprocess video."""
        if not video_path or not os.path.exists(video_path):
            return None

        try:
            cap = cv2.VideoCapture(video_path)
            frames = []

            # Calculate frame sampling
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            target_frames = int(self.video_fps * self.audio_duration)

            if total_frames > 0:
                frame_indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)

                for frame_idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        # Convert BGR to RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = Image.fromarray(frame)

                        # Resize frame
                        frame = frame.resize(self.image_size)

                        # Convert to tensor
                        frame_tensor = transforms.ToTensor()(frame)
                        frames.append(frame_tensor)

            cap.release()

            if frames:
                video_tensor = torch.stack(frames)  # (T, C, H, W)

                if self.transform_video:
                    video_tensor = self.transform_video(video_tensor)

                return video_tensor
            else:
                return None

        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        output = {}

        # Load multimodal data
        audio = torch.tensor(-1)
        if self.load_audio:
            audio = self._load_audio(sample['audio_path'])
            output['audio'] = audio

        image = torch.tensor(-1)
        if self.load_image:
            image = self._load_image(sample['image_path'])
            output['image'] = image

        video = torch.tensor(-1)
        if self.load_video:
            video = self._load_video(sample['video_path'])
            output['video'] = video
        
        sample_class_name = sample['class_name'].lower()

        text_emb = torch.tensor(-1)
        if sample['class_name'] in self.text_embs:
            text_emb = self.text_embs[sample['class_name']]
            output['text_emb'] = text_emb
        else:
            logger.warn(f"Text embedding not found for class: {sample['class_name']}")

        
        if self.use_saved_audio_embeddings:
            if sample['audio_filename'] in self.audio_embs:
                audio_emb = self.audio_embs[sample_class_name]['embeddings']
                output['audio_emb'] = audio_emb
            else:
                logger.warn(f"Audio embedding not found for file: {sample['audio_filename']} class: {sample['class_name']}")

        # Create output dictionary
        output.update({
            'label': torch.tensor(sample['class_label'], dtype=torch.long),
            'audio_filename': sample['audio_filename'],
            'image_filename': sample['image_filename'],
            'video_filename': sample['video_filename'],
            'class_name': sample['class_name'],
            'video_id': sample['video_id'],
            'sample_idx': sample['sample_idx'],
        })

        # Add metadata
        output.update({
            'ytid': sample['ytid'],
            'start_second': sample['start_second'],
            'end_second': sample['end_second'],
            'caption': sample.get('caption', '')
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

    def get_text_embeddings(self, class_name: str):
        """Get text embeddings for a given class name."""
        if self.text_embs is not None and class_name.lower() in self.text_embs:
            return self.text_embs[class_name.lower()]
        return None

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
        logger.info(f"\n=== VEGAS Dataset Statistics ({self.split} split) ===")
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

    def verify_stratification(self, other_dataset: 'VEGASDataset', tolerance: float = 0.05) -> bool:
        """
        Verify that class distributions are similar between this and another dataset.

        Args:
            other_dataset: Another VEGASDataset to compare with
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
            cache_files = [f for f in os.listdir(self.cache_dir) if f.startswith('vegas_samples_')]
            for cache_file in cache_files:
                cache_path = os.path.join(self.cache_dir, cache_file)
                try:
                    os.remove(cache_path)
                    logger.info(f"Removed cache file: {cache_path}")
                except Exception as e:
                    logger.error(f"Failed to remove cache file {cache_path}: {e}")


def create_vegas_dataloader(dataset: VEGASDataset,
                           batch_size: int = 32,
                           shuffle: bool = True,
                           num_workers: int = 4,
                           pin_memory: bool = True) -> DataLoader:
    """Create a DataLoader for VEGAS dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=_vegas_collate_fn
    )


def _vegas_collate_fn(batch):
    """Custom collate function for VEGAS batch processing."""
    # Separate None videos from valid ones
    valid_videos = [item['video'] for item in batch if item['video'] is not None]
    has_video = len(valid_videos) > 0

    # Stack tensors
    audio = torch.stack([item['audio'] for item in batch])
    image = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])

    # Handle videos (some might be None)
    if has_video:
        # Pad videos to same length if necessary
        max_frames = max(v.shape[0] for v in valid_videos)
        padded_videos = []
        video_mask = []

        for item in batch:
            if item['video'] is not None:
                video = item['video']
                if video.shape[0] < max_frames:
                    # Pad with last frame
                    last_frame = video[-1:].repeat(max_frames - video.shape[0], 1, 1, 1)
                    video = torch.cat([video, last_frame], dim=0)
                padded_videos.append(video)
                video_mask.append(True)
            else:
                # Create dummy video
                dummy_video = torch.zeros(max_frames, 3, 224, 224)
                padded_videos.append(dummy_video)
                video_mask.append(False)

        video = torch.stack(padded_videos)
        video_mask = torch.tensor(video_mask)
    else:
        video = None
        video_mask = None

    # Collect metadata
    metadata = {
        'class_names': [item['class_name'] for item in batch],
        'video_ids': [item['video_id'] for item in batch],
        'sample_indices': [item['sample_idx'] for item in batch],
        'ytids': [item['ytid'] for item in batch],
        'start_seconds': [item['start_second'] for item in batch],
        'end_seconds': [item['end_second'] for item in batch],
        'captions': [item.get('caption', '') for item in batch]
    }

    return {
        'audio': audio,
        'image': image,
        'video': video,
        'video_mask': video_mask,
        'labels': labels,
        'metadata': metadata
    }


# Example usage and testing
if __name__ == "__main__":
    # Test dataset loading
    print("Testing VEGAS Dataset...")

    # Test with specific classes
    dataset = VEGASDataset(
        selected_classes=['dog', 'baby_cry', 'chainsaw'],
        split='train',
        enable_cache=True
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Active classes: {dataset.get_class_names()}")
    print(f"Samples per class: {dataset.get_samples_per_class()}")

    # Test single sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Audio shape: {sample['audio'].shape}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Video shape: {sample['video'].shape if sample['video'] is not None else torch.tensor(-1)}")
    print(f"Label: {sample['label']} ({sample['class_name']})")

    # Test dataloader
    dataloader = create_vegas_dataloader(dataset, batch_size=4)
    batch = next(iter(dataloader))
    print(f"Batch audio shape: {batch['audio'].shape}")
    print(f"Batch image shape: {batch['image'].shape}")
    print(f"Batch video shape: {batch['video'].shape if batch['video'] is not None else torch.tensor(-1)}")
    print(f"Batch labels: {batch['labels']}")