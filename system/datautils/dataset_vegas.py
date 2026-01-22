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
                 num_samples_per_class: Optional[int] = None,
                 split: Optional[str] = None,
                 splits_to_load: Optional[List[str]] = None,
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.2,
                 split_ratio: Optional[float] = None,
                 use_folds: bool = False,
                 train_folds: Optional[List[int]] = None,
                 val_folds: Optional[List[int]] = None,
                 test_folds: Optional[List[int]] = None,
                 stratify: bool = True,
                 use_saved_audio_embeddings: bool = True,
                 node_split_id: Optional[int] = None,
                 num_nodes: Optional[int] = None,
                 samples_per_node: Optional[int] = None,
                 node_split_seed: int = 42,
                 enable_cache: bool = False,
                 cache_dir: str = "/tmp/vegas_cache",
                 audio_sample_rate: int = 16000,
                 audio_duration: float = 5.0,
                 image_size: Tuple[int, int] = (224, 224),
                 video_fps: int = 25,
                 transform_audio: Optional = None,
                 transform_image: Optional = None,
                 transform_video: Optional = None,
                 load_audio: bool = True,
                 load_image: bool = False,
                 load_video: bool = False,
                 ast_cache_dir: Optional[str] = None,
                 enable_ast_cache: bool = True):
        """
        Initialize VEGAS dataset.

        Args:
            root_dir: Root directory of VEGAS dataset
            embedding_file: Path to text embeddings file
            audio_embedding_file: Path to audio embeddings file
            selected_classes: List of classes to include (str names or int labels)
            excluded_classes: List of classes to exclude (str names or int labels)
            num_samples_per_class: Maximum number of samples to load per class (None = all samples)
            ast_cache_dir: Directory for AST embedding cache files (None = root_dir/ast_cache)
            enable_ast_cache: Whether to enable AST embedding caching (default True)
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
            use_folds: Whether to use fold-based splits (for custom fold structure)
            train_folds: List of fold indices for training
            val_folds: List of fold indices for validation, if None uses val_ratio from train
            test_folds: List of fold indices for testing
            stratify: Whether to use stratified sampling for train/val/test split
            use_saved_audio_embeddings: Whether to use saved audio embeddings
            node_split_id: Data split ID for this node (0-indexed). Allows multiple nodes with
                          the same class to have different data splits.
                          Example: Two nodes with class 'dog' can have node_split_id=0 and node_split_id=1
                          to get different portions of the 'dog' samples.
            num_nodes: Total number of nodes for federated split (mutually exclusive with samples_per_node)
            samples_per_node: Number of samples per node (mutually exclusive with num_nodes)
                            Both num_nodes and samples_per_node distribute samples proportionally per class
            node_split_seed: Random seed for reproducible node splits (default: 42)
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
        self.num_samples_per_class = num_samples_per_class
        self.split = split

        # AST cache configuration
        self.enable_ast_cache = enable_ast_cache
        self.ast_cache_dir = ast_cache_dir or os.path.join(root_dir, "ast_cache")
        if self.enable_ast_cache:
            os.makedirs(self.ast_cache_dir, exist_ok=True)

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
        self.train_folds = train_folds or []
        self.val_folds = val_folds
        self.test_folds = test_folds or []
        self.node_split_id = node_split_id
        self.num_nodes = num_nodes
        self.samples_per_node = samples_per_node
        self.node_split_seed = node_split_seed
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

        # Load captions if available
        self.captions = {}
        captions_path = os.path.join(self.root_dir, "captions.json")
        if os.path.exists(captions_path):
            import json
            with open(captions_path, 'r') as f:
                self.captions = json.load(f)

        # Handle auto-split creation when split=None
        if split is None:
            logger.info("Auto-creating train/val/test splits (split=None)")
            self._create_all_splits(
                root_dir=root_dir,
                embedding_file=embedding_file,
                audio_embedding_file=audio_embedding_file,
                selected_classes=selected_classes,
                excluded_classes=excluded_classes,
                num_samples_per_class=num_samples_per_class,
                splits_to_load=splits_to_load,
                train_ratio=self.train_ratio,
                val_ratio=self.val_ratio,
                test_ratio=self.test_ratio,
                use_folds=use_folds,
                train_folds=train_folds,
                val_folds=val_folds,
                test_folds=test_folds,
                stratify=stratify,
                use_saved_audio_embeddings=use_saved_audio_embeddings,
                node_split_id=node_split_id,
                num_nodes=num_nodes,
                samples_per_node=samples_per_node,
                node_split_seed=node_split_seed,
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
                load_video=load_video,
                ast_cache_dir=self.ast_cache_dir,  # Pass AST cache dir to splits
                enable_ast_cache=self.enable_ast_cache  # Pass AST cache enable flag to splits
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



        self.transform_audio = transform_audio
        self.transform_image = transform_image or self._default_image_transform()
        self.transform_video = transform_video

        self.use_saved_audio_embeddings = use_saved_audio_embeddings

        # Create cache directory
        if self.enable_cache:
            os.makedirs(self.cache_dir, exist_ok=True)

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

    def _create_all_splits(self, ast_cache_dir=None, enable_ast_cache=True, **kwargs):
        """
        Create train, val, and test splits automatically.

        This method is called when split=None to auto-create all three splits.
        The splits are exposed as .train, .val, and .test attributes.

        Args:
            ast_cache_dir: Directory for AST cache files
            enable_ast_cache: Whether to enable AST caching
            **kwargs: All other dataset initialization parameters
        """
        logger.info("Creating train split...")
        self.train = VEGASDataset(
            split='train',
            ast_cache_dir=ast_cache_dir,
            enable_ast_cache=enable_ast_cache,
            **kwargs
        )

        logger.info("Creating validation split...")
        self.val = VEGASDataset(
            split='val',
            ast_cache_dir=ast_cache_dir,
            enable_ast_cache=enable_ast_cache,
            **kwargs
        )

        logger.info("Creating test split...")
        self.test = VEGASDataset(
            split='test',
            ast_cache_dir=ast_cache_dir,
            enable_ast_cache=enable_ast_cache,
            **kwargs
        )

        logger.info(f"Auto-split created: train={len(self.train)}, val={len(self.val)}, test={len(self.test)}")

    def _get_ast_cache_config_hash(self, sample_rate: int, duration: float, model_name: str = "ast") -> str:
        """
        Generate hash for AST cache configuration.

        Args:
            sample_rate: Audio sample rate used for AST processing
            duration: Audio duration in seconds
            model_name: Name/version of AST model used

        Returns:
            Configuration hash string
        """
        config_str = f"{model_name}_{sample_rate}_{duration}"
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def _get_ast_cache_filepath(self, sample_rate: int, duration: float, model_name: str = "ast") -> str:
        """
        Get the filepath for AST embeddings cache.

        Args:
            sample_rate: Audio sample rate used for AST processing
            duration: Audio duration in seconds
            model_name: Name/version of AST model used

        Returns:
            Full path to cache file
        """
        config_hash = self._get_ast_cache_config_hash(sample_rate, duration, model_name)
        cache_filename = f"ast_embeddings_{config_hash}.pt"
        return os.path.join(self.ast_cache_dir, cache_filename)

    def _verify_ast_cache_compatibility(self, cache_file: str,
                                       expected_sample_rate: int,
                                       expected_duration: float,
                                       expected_model: str = "ast") -> bool:
        """
        Verify that cached AST embeddings are compatible with current configuration.

        Args:
            cache_file: Path to cache file
            expected_sample_rate: Expected audio sample rate
            expected_duration: Expected audio duration
            expected_model: Expected model name

        Returns:
            True if cache is compatible, False otherwise
        """
        if not os.path.exists(cache_file):
            return False

        try:
            # Load only metadata to check compatibility (use mmap for efficiency)
            cached_data = torch.load(cache_file, map_location='cpu', weights_only=False, mmap=True)

            # Check if metadata exists
            if 'metadata' not in cached_data:
                logger.warning(f"AST cache missing metadata: {cache_file}")
                return False

            metadata = cached_data['metadata']

            # Verify configuration match
            config_match = (
                metadata.get('sample_rate') == expected_sample_rate and
                metadata.get('duration') == expected_duration and
                metadata.get('model_name') == expected_model
            )

            if not config_match:
                logger.info(f"AST cache config mismatch. Expected: {expected_sample_rate}Hz, {expected_duration}s, {expected_model}. "
                          f"Got: {metadata.get('sample_rate')}Hz, {metadata.get('duration')}s, {metadata.get('model_name')}")
                return False

            # Verify embeddings structure
            if 'embeddings' not in cached_data:
                logger.warning(f"AST cache missing embeddings: {cache_file}")
                return False

            logger.info(f"AST cache verified: {cache_file} ({len(cached_data['embeddings'])} embeddings)")
            return True

        except Exception as e:
            logger.error(f"Error verifying AST cache {cache_file}: {e}")
            return False

    def load_ast_embeddings_from_cache(self, sample_rate: int = 16000,
                                       duration: float = 5.0,
                                       model_name: str = "ast") -> bool:
        """
        Load AST embeddings from cache if available and compatible.

        Args:
            sample_rate: Audio sample rate used for AST processing
            duration: Audio duration in seconds
            model_name: Name/version of AST model used

        Returns:
            True if loaded successfully, False if cache not found or incompatible
        """
        if not self.enable_ast_cache:
            logger.info("AST cache disabled")
            return False

        cache_file = self._get_ast_cache_filepath(sample_rate, duration, model_name)

        # Verify compatibility
        if not self._verify_ast_cache_compatibility(cache_file, sample_rate, duration, model_name):
            return False

        try:
            logger.info(f"Loading AST embeddings from cache: {cache_file}")

            # Load with memory mapping for efficient access
            cached_data = torch.load(cache_file, map_location='cpu', weights_only=False, mmap=True)

            # Store embeddings
            self.audio_embs_from_file = cached_data['embeddings']

            logger.info(f"Loaded {len(self.audio_embs_from_file)} AST embeddings from cache")
            logger.info(f"Cache metadata: {cached_data['metadata']}")

            return True

        except Exception as e:
            logger.error(f"Error loading AST cache {cache_file}: {e}")
            return False

    def save_ast_embeddings_to_cache(self, embeddings: Dict[str, torch.Tensor],
                                    sample_rate: int = 16000,
                                    duration: float = 5.0,
                                    model_name: str = "ast",
                                    embedding_shape: Optional[tuple] = None) -> bool:
        """
        Save AST embeddings to cache with metadata.

        Args:
            embeddings: Dictionary mapping file IDs to AST embeddings
            sample_rate: Audio sample rate used for AST processing
            duration: Audio duration in seconds
            model_name: Name/version of AST model used
            embedding_shape: Shape of embeddings (inferred if None)

        Returns:
            True if saved successfully, False otherwise
        """
        if not self.enable_ast_cache:
            logger.info("AST cache disabled, skipping save")
            return False

        if not embeddings:
            logger.warning("No embeddings to cache")
            return False

        try:
            cache_file = self._get_ast_cache_filepath(sample_rate, duration, model_name)

            # Infer embedding shape from first embedding
            if embedding_shape is None:
                first_emb = next(iter(embeddings.values()))
                if isinstance(first_emb, torch.Tensor):
                    embedding_shape = tuple(first_emb.shape)
                elif isinstance(first_emb, dict) and 'embedding' in first_emb:
                    embedding_shape = tuple(first_emb['embedding'].shape)

            # Create cache data structure
            cache_data = {
                'embeddings': embeddings,
                'metadata': {
                    'sample_rate': sample_rate,
                    'duration': duration,
                    'model_name': model_name,
                    'embedding_shape': embedding_shape,
                    'num_embeddings': len(embeddings),
                    'creation_time': pd.Timestamp.now().isoformat(),
                    'dataset_root': self.root_dir
                }
            }

            logger.info(f"Saving {len(embeddings)} AST embeddings to cache: {cache_file}")

            # Save with atomic write (write to temp file then rename)
            temp_file = cache_file + ".tmp"
            torch.save(cache_data, temp_file)

            # Atomic rename
            if os.path.exists(cache_file):
                os.remove(cache_file)
            os.rename(temp_file, cache_file)

            logger.info(f"AST embeddings cached successfully: {cache_file}")
            logger.info(f"Cache metadata: {cache_data['metadata']}")

            return True

        except Exception as e:
            logger.error(f"Error saving AST cache: {e}")
            # Clean up temp file if it exists
            temp_file = self._get_ast_cache_filepath(sample_rate, duration, model_name) + ".tmp"
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            return False

    def clear_ast_cache(self, sample_rate: Optional[int] = None,
                       duration: Optional[float] = None,
                       model_name: Optional[str] = None):
        """
        Clear AST embedding cache files.

        Args:
            sample_rate: If specified, only clear cache for this sample rate
            duration: If specified, only clear cache for this duration
            model_name: If specified, only clear cache for this model

        If all args are None, clears all AST cache files.
        """
        if not self.enable_ast_cache:
            logger.warning("AST cache disabled")
            return

        if not os.path.exists(self.ast_cache_dir):
            logger.info("AST cache directory does not exist")
            return

        try:
            if sample_rate is not None and duration is not None:
                # Clear specific cache file
                cache_file = self._get_ast_cache_filepath(
                    sample_rate,
                    duration,
                    model_name or "ast"
                )
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                    logger.info(f"Removed AST cache: {cache_file}")
                else:
                    logger.info(f"AST cache not found: {cache_file}")
            else:
                # Clear all AST cache files
                cache_files = [f for f in os.listdir(self.ast_cache_dir)
                             if f.startswith('ast_embeddings_') and f.endswith('.pt')]

                for cache_file in cache_files:
                    cache_path = os.path.join(self.ast_cache_dir, cache_file)
                    os.remove(cache_path)
                    logger.info(f"Removed AST cache: {cache_path}")

                logger.info(f"Cleared {len(cache_files)} AST cache file(s)")

        except Exception as e:
            logger.error(f"Error clearing AST cache: {e}")

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
        """
        Filter classes based on selection and exclusion.

        IMPORTANT: Labels are NOT remapped - original CLASS_LABELS indices are preserved.
        This ensures consistency across federated nodes and with conditional generators.

        Example:
            CLASS_LABELS = {'baby_cry': 0, 'chainsaw': 1, 'dog': 2, 'drum': 3, ...}
            selected_classes = ['dog', 'drum']
            Result: {'dog': 2, 'drum': 3}  # Original labels preserved, not remapped to [0, 1]
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

                # Load samples for this split
                split_samples = self._load_samples_from_disk()

                # Apply node split BEFORE train/val/test split
                if self.node_split_id is not None and (self.num_nodes is not None or self.samples_per_node is not None):
                    split_samples = self._apply_node_split(split_samples)

                # Apply split if not 'all'
                if temp_split != 'all':
                    split_samples = self._apply_split(split_samples)

                all_split_samples.extend(split_samples)

                # Restore original split
                self.split = original_split

            return all_split_samples

        # Normal loading (single split)
        samples = self._load_samples_from_disk()

        # CRITICAL FIX: Apply node split BEFORE train/val/test split
        # This ensures that samples_per_node is applied to the full dataset first,
        # then train/val/test ratios are applied to the node's subset
        if self.node_split_id is not None and (self.num_nodes is not None or self.samples_per_node is not None):
            samples = self._apply_node_split(samples)

        # Apply train/test split to the node's samples
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

    def _load_samples_from_disk(self) -> List[Dict]:
        """
        Load samples from disk for all active classes.

        Returns:
            List of sample dictionaries
        """
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

            # Track samples for this class
            class_samples = []

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

                    # Extract file_id (remove .wav extension) - used for caching audio embeddings
                    file_id = audio_filename.replace('.wav', '')

                    sample = {
                        'class_name': class_name,
                        'class_label': self.active_classes[class_name],
                        'video_id': video_id,
                        'audio_path': audio_path,
                        'audio_filename': audio_filename,
                        'file_id': file_id,
                        'image_path': image_path,
                        'image_filename': image_filename,
                        'video_path': video_path if os.path.exists(video_path) else None,
                        'video_filename': video_filename,
                        'ytid': row.get('YTID', ''),
                        'start_second': row.get('start_second', 0),
                        'end_second': row.get('end_second', self.audio_duration),
                        'sample_idx': len(samples) + len(class_samples),
                        'audio_emb': self.audio_embs.get(audio_filename, None),
                        'caption': self.captions.get(video_id, '')
                    }
                    class_samples.append(sample)

            # Apply num_samples_per_class limit per class if specified
            # IMPORTANT: Skip this limit if samples_per_node is set, as samples_per_node takes priority
            if self.samples_per_node is None:
                if self.num_samples_per_class is not None and self.num_samples_per_class > 0 and len(class_samples) > self.num_samples_per_class:
                    logger.info(f"Limiting class '{class_name}' from {len(class_samples)} to {self.num_samples_per_class} samples")
                    class_samples = class_samples[:self.num_samples_per_class]
            else:
                logger.debug(f"Skipping num_samples_per_class limit for class '{class_name}' because samples_per_node={self.samples_per_node} is set")

            samples.extend(class_samples)

        return samples

    def _get_cache_key(self) -> str:
        """Generate cache key based on dataset configuration."""
        config_str = f"{self.root_dir}_{self.active_classes}_{self.split}_{self.split_ratio}_{self.node_split_id}_{self.num_samples_per_class}"
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _apply_split(self, samples: List[Dict]) -> List[Dict]:
        """
        Apply train/val/test split to samples with optional stratification.

        Uses scikit-learn's train_test_split for stratified sampling.
        Supports custom train_ratio, val_ratio, test_ratio (default 70-10-20).

        Notes:
        - train split is always present
        - val split is optional (val_ratio can be 0)
        - test split is optional (test_ratio can be 0)
        """
        if not samples:
            return samples

        # Use reproducible random seed based on node_split_id
        random_state = 42 + (self.node_split_id or 0)

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
                if has_val_split and not (self.use_folds and self.val_folds):
                    # Second split: train vs val
                    train_val_labels = [s['class_label'] for s in train_val_samples]
                    # Calculate val size relative to train_val set
                    # val_ratio is relative to total, so we need to adjust
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
                cls_samples = sorted(cls_samples, key=lambda x: x['video_id'])
                rng.shuffle(cls_samples)

                if has_val_split and not (self.use_folds and self.val_folds):
                    # Three-way split: train / val / test using custom ratios
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
                    # Two-way split: train / test using custom ratios
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

    def _apply_node_split(self, samples: List[Dict]) -> List[Dict]:
        """
        Apply node-based splitting for federated learning.
        Distributes samples across nodes while maintaining class balance.

        Supports two modes:
        1. num_nodes: Divide all samples equally among N nodes
        2. samples_per_node: Each node gets exactly K samples per class
           In this mode, node_split_id determines which portion of samples to use.
           Example: node_split_id=0 gets samples [0:samples_per_node],
                    node_split_id=1 gets samples [samples_per_node:2*samples_per_node]

        Args:
            samples: List of sample dictionaries

        Returns:
            List of samples assigned to this node
        """
        if not samples:
            return samples

        # Validate parameters
        if self.num_nodes is not None and self.samples_per_node is not None:
            raise ValueError("Cannot specify both num_nodes and samples_per_node. Use one or the other.")

        if self.num_nodes is not None and self.node_split_id >= self.num_nodes:
            raise ValueError(f"node_split_id ({self.node_split_id}) must be less than num_nodes ({self.num_nodes})")

        # Use reproducible random seed based on node_split_seed
        rng = np.random.RandomState(self.node_split_seed)

        # Group samples by class for balanced splitting
        class_samples = {}
        for sample in samples:
            class_name = sample['class_name']
            if class_name not in class_samples:
                class_samples[class_name] = []
            class_samples[class_name].append(sample)

        # Shuffle each class independently with reproducible seed
        for class_name in class_samples.keys():
            class_samples[class_name] = sorted(class_samples[class_name], key=lambda x: x['video_id'])
            rng.shuffle(class_samples[class_name])

        node_samples = []

        if self.num_nodes is not None:
            # Mode 1: Divide samples equally among num_nodes
            for class_name, cls_samples in class_samples.items():
                n_samples = len(cls_samples)
                samples_per_node = n_samples // self.num_nodes
                remainder = n_samples % self.num_nodes

                # Calculate start and end indices for this node
                # Distribute remainder samples to first nodes
                if self.node_split_id < remainder:
                    start_idx = self.node_split_id * (samples_per_node + 1)
                    end_idx = start_idx + samples_per_node + 1
                else:
                    start_idx = self.node_split_id * samples_per_node + remainder
                    end_idx = start_idx + samples_per_node

                node_samples.extend(cls_samples[start_idx:end_idx])

                logger.info(f"Node split {self.node_split_id}/{self.num_nodes} - Class {class_name}: "
                          f"{end_idx - start_idx} samples (total: {n_samples})")

        elif self.samples_per_node is not None:
            # Mode 2: Each node gets exactly samples_per_node samples per class
            for class_name, cls_samples in class_samples.items():
                n_samples = len(cls_samples)

                # Calculate start index for this node based on node_split_id
                start_idx = self.node_split_id * self.samples_per_node
                end_idx = min(start_idx + self.samples_per_node, n_samples)

                if start_idx >= n_samples:
                    logger.warning(f"Node split {self.node_split_id} - Class {class_name}: "
                                 f"Not enough samples (requested {self.samples_per_node}, available {n_samples})")
                    continue

                node_samples.extend(cls_samples[start_idx:end_idx])

                actual_samples = end_idx - start_idx
                logger.info(f"Node split {self.node_split_id} - Class {class_name}: "
                          f"{actual_samples} samples (requested: {self.samples_per_node})")

        logger.info(f"Node split {self.node_split_id} total samples after split: {len(node_samples)}")

        return node_samples

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


        # Load audio embedding from cache if available
        sample_id = f'{sample['file_id']}:{sample_class_name}'
        if self.audio_embs and sample_id in self.audio_embs:
            audio_emb = self.audio_embs[sample_id]
            output['audio_emb'] = audio_emb

        # Create output dictionary
        output.update({
            'label': torch.tensor(sample['class_label'], dtype=torch.long),
            'audio_filename': sample['audio_filename'],
            'image_filename': sample['image_filename'],
            'video_filename': sample['video_filename'],
            'class_name': sample['class_name'],
            'video_id': sample['video_id'],
            'file_id': sample['file_id'],
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
        """
        Get number of active classes.

        NOTE: This returns the COUNT of active classes, not the maximum label value.
        For the maximum label index needed for model output dimensions, use get_max_class_label().

        Example:
            selected_classes = ['dog', 'drum']  # labels 2, 3
            get_num_classes() -> 2 (count of classes)
            get_max_class_label() -> 3 (max label value)
        """
        return len(self.active_classes)

    def get_max_class_label(self) -> int:
        """
        Get the maximum class label value in active classes.

        This is useful for determining model output dimensions when labels are not remapped.
        Since VEGAS preserves original CLASS_LABELS indices, this may differ from get_num_classes().

        Returns:
            Maximum label value, or -1 if no active classes

        Example:
            selected_classes = ['dog', 'drum']  # original labels: dog=2, drum=3
            get_max_class_label() -> 3

            all_classes (10 classes total)
            get_max_class_label() -> 9
        """
        if not self.active_classes:
            return -1
        return max(self.active_classes.values())

    def get_collate_fn(self):
        """Get the collate function for this dataset."""
        return _vegas_collate_fn

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

    def to(self, device: Union[str, torch.device]) -> 'VEGASDataset':
        if isinstance(device, str):
            device = torch.device(device)

        self.device = device
        logger.debug(f"Moving dataset tensors to device: {device}")

        # # Move text embeddings
        # if self.text_embs is not None:
        #     for key in self.text_embs.keys():
        #         if isinstance(self.text_embs[key], torch.Tensor):
        #             self.text_embs[key] = self.text_embs[key].to(device)
        #         elif isinstance(self.text_embs[key], dict):
        #             # Handle nested dictionaries (e.g., {'sd': tensor, 'flux': tensor})
        #             for subkey in self.text_embs[key].keys():
        #                 if isinstance(self.text_embs[key][subkey], torch.Tensor):
        #                     self.text_embs[key][subkey] = self.text_embs[key][subkey].to(device)

        # Move audio embeddings from file
        # if self.audio_embs_from_file is not None:
        #     for key in self.audio_embs_from_file.keys():
        #         if isinstance(self.audio_embs_from_file[key], torch.Tensor):
        #             self.audio_embs_from_file[key] = self.audio_embs_from_file[key].to(device)
        #         elif isinstance(self.audio_embs_from_file[key], dict):
        #             for subkey in self.audio_embs_from_file[key].keys():
        #                 if isinstance(self.audio_embs_from_file[key][subkey], torch.Tensor):
        #                     self.audio_embs_from_file[key][subkey] = self.audio_embs_from_file[key][subkey].to(device)

        # Move filtered audio embeddings
        # if self._audio_embs:
        #     for key in self._audio_embs.keys():
        #         if isinstance(self._audio_embs[key], torch.Tensor):
        #             self._audio_embs[key] = self._audio_embs[key].to(device)
        #         elif isinstance(self._audio_embs[key], dict):
        #             for subkey in self._audio_embs[key].keys():
        #                 if isinstance(self._audio_embs[key][subkey], torch.Tensor):
        #                     self._audio_embs[key][subkey] = self._audio_embs[key][subkey].to(device)

        # Move tensors in splits if they exist
        if hasattr(self, 'train') and self.train is not None:
            self.train.to(device)
        if hasattr(self, 'val') and self.val is not None:
            self.val.to(device)
        if hasattr(self, 'test') and self.test is not None:
            self.test.to(device)

        logger.debug(f"Successfully moved all dataset tensors to {device}")
        return self


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
    # Build output dictionary
    output = {}

    # Stack labels (always present)
    labels = torch.stack([item['label'] for item in batch])
    output['labels'] = labels

    # Stack audio if available
    if 'audio' in batch[0]:
        audio = torch.stack([item['audio'] for item in batch])
        output['audio'] = audio

    # Stack images if available
    if 'image' in batch[0]:
        image = torch.stack([item['image'] for item in batch])
        output['image'] = image

    # Stack text embeddings if available
    if 'text_emb' in batch[0]:
        text_embs = batch[0]['text_emb']  # Dict with sd/flux keys
        output['text_emb'] = text_embs

    # Stack audio embeddings if available
    if 'audio_emb' in batch[0]:
        audio_emb = torch.stack([item['audio_emb'] for item in batch])
        output['audio_emb'] = audio_emb

    # Handle videos (conditionally, some might be None)
    if 'video' in batch[0]:
        valid_videos = [item['video'] for item in batch if isinstance(item['video'], torch.Tensor) and item['video'].numel() > 1]
        has_video = len(valid_videos) > 0

        if has_video:
            # Pad videos to same length if necessary
            max_frames = max(v.shape[0] for v in valid_videos)
            padded_videos = []
            video_mask = []

            for item in batch:
                if isinstance(item['video'], torch.Tensor) and item['video'].numel() > 1:
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

            output['video'] = torch.stack(padded_videos)
            output['video_mask'] = torch.tensor(video_mask)
        else:
            output['video'] = None
            output['video_mask'] = None

    # Collect metadata
    metadata = {
        'class_names': [item['class_name'] for item in batch],
        'video_ids': [item['video_id'] for item in batch],
        'file_ids': [item['file_id'] for item in batch],
        'sample_indices': [item['sample_idx'] for item in batch],
        'ytids': [item['ytid'] for item in batch],
        'start_seconds': [item['start_second'] for item in batch],
        'end_seconds': [item['end_second'] for item in batch],
        'captions': [item.get('caption', '') for item in batch]
    }

    # Add metadata to output
    output['metadata'] = metadata

    # Add commonly accessed fields at top level for easy access
    output['class_name'] = metadata['class_names']
    output['file_id'] = metadata['file_ids']
    output['audio_filename'] = [item['audio_filename'] for item in batch]

    return output


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