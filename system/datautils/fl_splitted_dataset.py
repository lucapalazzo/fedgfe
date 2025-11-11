import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Union, Dict, Any


class FLSplittedDataset(Dataset):
    """
    Dataset class for handling federated learning pre-split datasets.

    This class handles loading of npz files that contain federated data splits,
    automatically unwrapping data from 'data' key and supporting both
    dictionary format (samples/labels) and tuple format (x/y).
    """

    def __init__(
        self,
        dataset_path: str,
        node_id: int,
        is_train: bool = True,
        prefix: str = "",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        device: str = 'cpu'
    ):
        """
        Initialize FLSplittedDataset.

        Args:
            dataset_path: Path to the dataset directory
            node_id: ID of the federated node
            is_train: Whether to load training or test data
            prefix: Prefix for npz files
            transform: Transform to apply to samples
            target_transform: Transform to apply to targets
            device: Device to load data on
        """
        self.dataset_path = dataset_path
        self.node_id = node_id
        self.is_train = is_train
        self.prefix = prefix
        self.transform = transform
        self.target_transform = target_transform
        self.device = device

        # Load the data
        self.data = self._load_data()
        self._process_data()

    def _load_data(self) -> Dict[str, Any]:
        """Load data from npz file."""
        split_dir = 'train' if self.is_train else 'test'
        data_dir = os.path.join(self.dataset_path, split_dir)

        # Try different file naming patterns
        file_patterns = [
            f"{self.prefix}{self.node_id}.npz",
            f"{self.node_id}.npz"
        ]

        data_file = None
        for pattern in file_patterns:
            candidate_file = os.path.join(data_dir, pattern)
            if os.path.isfile(candidate_file):
                data_file = candidate_file
                break

        if data_file is None:
            raise FileNotFoundError(f"No data file found for node {self.node_id} in {data_dir}")

        # Load the npz file
        with open(data_file, 'rb') as f:
            loaded_data = np.load(f, allow_pickle=True)

            # Check if data is wrapped in 'data' key
            if 'data' in loaded_data and loaded_data['data'].dtype == object:
                # Unwrap the data
                actual_data = loaded_data['data'].item()
                if isinstance(actual_data, dict):
                    return actual_data
                else:
                    # If not a dict, wrap it back
                    return {'data': actual_data}
            else:
                # Data is not wrapped, convert to dict
                return {key: loaded_data[key] for key in loaded_data.keys()}

    def _process_data(self):
        """Process loaded data into appropriate format."""
        # Check data format and convert tensors
        if "samples" in self.data.keys():
            # Dictionary format (samples/labels/masks)
            self.format = "dict"
            self._convert_dict_format()
        elif "x" in self.data.keys() and "y" in self.data.keys():
            # Tuple format (x/y)
            self.format = "tuple"
            self._convert_tuple_format()
        else:
            raise ValueError(f"Unsupported data format. Keys: {list(self.data.keys())}")

    def _convert_dict_format(self):
        """Convert dictionary format data to tensors."""
        for key in self.data.keys():
            if key != "samples" and np.issubdtype(self.data[key].dtype, np.number):
                self.data[key] = torch.Tensor(self.data[key])

        # Convert samples if they are numeric
        if np.issubdtype(self.data['samples'].dtype, np.number):
            self.data['samples'] = torch.Tensor(self.data['samples'])

    def _convert_tuple_format(self):
        """Convert tuple format data to list of tuples."""
        X = torch.Tensor(self.data['x']).type(torch.float32)
        y = torch.Tensor(self.data['y']).type(torch.int64)
        self.tuple_data = [(x, y) for x, y in zip(X, y)]

    def __len__(self) -> int:
        """Return the length of the dataset."""
        if self.format == "dict":
            return len(self.data['samples'])
        else:  # tuple format
            return len(self.tuple_data)

    def __getitem__(self, idx: int):
        """Get a sample from the dataset."""
        if self.format == "dict":
            return self._get_dict_item(idx)
        else:  # tuple format
            return self._get_tuple_item(idx)

    def _get_dict_item(self, idx: int):
        """Get item in dictionary format."""
        sample = self.data['samples'][idx]

        # Create label dictionary with all non-sample keys
        label = {}
        for k in self.data.keys():
            if k != 'samples':
                label[k] = self.data[k][idx]

        # Handle image format (move channel dimension if needed)
        if isinstance(sample, torch.Tensor) and len(sample.shape) == 3 and sample.shape[0] > 3:
            sample = sample.moveaxis(2, 0)

        # Apply transforms
        if self.transform is not None:
            sample = self._apply_transform(sample, idx)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return sample, label

    def _get_tuple_item(self, idx: int):
        """Get item in tuple format."""
        sample, label = self.tuple_data[idx]

        # Handle image format (move channel dimension if needed)
        if isinstance(sample, torch.Tensor) and len(sample.shape) == 3 and sample.shape[0] > 3:
            sample = sample.moveaxis(2, 0)

        # Apply transforms
        if self.transform is not None:
            sample = self._apply_transform(sample, idx)

        if self.target_transform is not None:
            label = self.target_transform(label)

        return sample, label

    def _apply_transform(self, sample: torch.Tensor, idx: int) -> torch.Tensor:
        """Apply transform to sample with size checking."""
        if hasattr(self.transform, 'transforms') and len(self.transform.transforms) > 0:
            # Get the last transform to check expected size
            last_transform = self.transform.transforms[-1]

            # Check if we need to transform based on size mismatch
            if hasattr(last_transform, 'size') and len(sample.shape) >= 2:
                expected_size = last_transform.size[0] if hasattr(last_transform.size, '__getitem__') else last_transform.size

                if sample.shape[-1] != expected_size:
                    transformed_sample = self.transform(sample)

                    # Expand single channel to 3 channels if needed
                    if len(transformed_sample.shape) == 3 and transformed_sample.shape[0] == 1:
                        transformed_sample = transformed_sample.expand(3, transformed_sample.shape[1], transformed_sample.shape[2])

                    return transformed_sample

        # If no specific size check needed, apply transform directly
        return self.transform(sample)

    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the loaded data."""
        info = {
            'format': self.format,
            'length': len(self),
            'node_id': self.node_id,
            'is_train': self.is_train,
            'data_keys': list(self.data.keys())
        }

        if self.format == "dict":
            info['sample_shape'] = self.data['samples'][0].shape if len(self.data['samples']) > 0 else None

            # Add information about labels/masks
            for key in self.data.keys():
                if key != 'samples':
                    info[f'{key}_shape'] = self.data[key].shape if hasattr(self.data[key], 'shape') else f"type_{type(self.data[key])}"
        else:
            info['sample_shape'] = self.tuple_data[0][0].shape if len(self.tuple_data) > 0 else None
            info['label_shape'] = self.tuple_data[0][1].shape if len(self.tuple_data) > 0 else None

        return info

    def to(self, device: str):
        """Move dataset to specified device."""
        self.device = device
        return self

    def get_labels(self) -> Optional[torch.Tensor]:
        """
        Get all labels from the dataset.

        Returns:
            Tensor containing all labels if available in 'labels' key,
            None if labels are not available
        """
        if self.format == "dict" and 'labels' in self.data:
            return self.data['labels']
        elif self.format == "tuple":
            # Extract labels from tuple data
            return torch.stack([y for _, y in self.tuple_data])
        return None

    def get_masks(self) -> Optional[torch.Tensor]:
        """
        Get all masks from the dataset.

        Returns:
            Tensor containing all masks if available in 'masks' key,
            None if masks are not available
        """
        if self.format == "dict" and 'masks' in self.data:
            return self.data['masks']
        
        if self.format == "dict" and 'semantic_masks' in self.data:
            return self.data['semantic_masks']
        
        return None

    def get_num_labels(self) -> int:
        """
        Get the number of labels in the dataset.

        Returns:
            Number of labels, or 0 if labels are not available
        """
        labels = self.get_labels()
        return len(labels) if labels is not None else 0

    def get_num_masks(self) -> int:
        """
        Get the number of masks in the dataset.

        Returns:
            Number of masks, or 0 if masks are not available
        """
        masks = self.get_masks()
        return len(masks) if masks is not None else 0

    def get_unique_labels(self) -> Optional[torch.Tensor]:
        """
        Get the unique labels present in the dataset.

        Returns:
            Tensor containing unique labels sorted in ascending order,
            None if labels are not available
        """
        labels = self.get_labels()
        if labels is not None:
            return torch.unique(labels, sorted=True)
        return None

    def get_num_tasks(self) -> int:
        """
        Get the total number of tasks (classification + segmentation).

        This unified method returns the sum of:
        - Classification tasks from get_num_classification_tasks()
        - Segmentation tasks from get_num_segmentation_tasks()

        Returns:
            Total number of tasks (sum of classification and segmentation tasks),
            or 0 if neither labels nor semantic_masks are available
        """
        num_classification_tasks = self.get_num_classification_tasks()
        num_segmentation_tasks = self.get_num_segmentation_tasks()

        return num_classification_tasks + num_segmentation_tasks

    def get_num_classification_tasks(self) -> int:
        """
        Get the number of classification tasks from the label shape.

        For multi-label/multi-task scenarios, this returns the number of tasks
        (i.e., the size of the last dimension of the labels tensor).
        For single-label scenarios, this returns 1.

        Returns:
            Number of classification tasks based on label shape,
            or 0 if labels are not available
        """
        labels = self.get_labels()
        if labels is None:
            return 0

        # If labels are multi-dimensional (e.g., [N, num_tasks]), return num_tasks
        if len(labels.shape) > 1:
            return labels.shape[-1]
        # If labels are 1D, it's a single task
        return 1

    def get_num_segmentation_tasks(self) -> int:
        """
        Get the number of segmentation tasks from the semantic mask shape.

        For multi-task segmentation scenarios, this returns the number of tasks
        (i.e., the size of dimension 1 of the semantic_masks tensor: [N, num_tasks, H, W]).
        For single-task scenarios, this returns 1.

        Returns:
            Number of segmentation tasks based on semantic_masks shape,
            or 0 if semantic_masks are not available
        """
        if self.format == "dict" and 'semantic_masks' in self.data:
            masks = self.data['semantic_masks']

            # Expected shape: [N, num_tasks, H, W] for multi-task
            # or [N, H, W] for single-task
            if len(masks.shape) == 4:
                # Multi-task: [N, num_tasks, H, W]
                return masks.shape[1]
            elif len(masks.shape) == 3:
                # Single-task: [N, H, W]
                return 1
            else:
                # Unexpected shape
                return 0

        return 0

    def get_samples_per_label(self) -> Optional[Dict[int, int]]:
        """
        Get the number of samples for each unique label in the dataset.
        For single-task scenarios only.

        Returns:
            Dictionary mapping label (int) to count (int),
            None if labels are not available
        """
        labels = self.get_labels()
        if labels is None:
            return None

        # For multi-dimensional labels, only work on single task
        if len(labels.shape) > 1:
            labels = labels[:, 0]

        # Count occurrences of each label
        unique_labels, counts = torch.unique(labels, return_counts=True)

        # Convert to dictionary
        samples_per_label = {
            int(label.item()): int(count.item())
            for label, count in zip(unique_labels, counts)
        }

        return samples_per_label

    def get_task_label_stats(self) -> Optional[Dict[int, Dict[str, Any]]]:
        """
        Get simple and direct label statistics for each classification task.

        This is a simplified version that returns essential statistics per task
        in an easy-to-use format.

        Returns:
            Dictionary mapping task_id -> statistics dict, None if labels not available

        Example:
            {
                0: {
                    'num_classes': 3,
                    'label_counts': {0: 150, 1: 200, 2: 175},
                    'total_samples': 525
                },
                1: {
                    'num_classes': 2,
                    'label_counts': {0: 300, 1: 225},
                    'total_samples': 525
                }
            }
        """
        labels = self.get_labels()
        if labels is None:
            return None

        num_tasks = self.get_num_classification_tasks()
        if num_tasks == 0:
            return None

        task_stats = {}

        # Handle single task case
        if num_tasks == 1:
            if len(labels.shape) == 1:
                task_labels = labels
            else:
                task_labels = labels[:, 0]

            unique_labels, counts = torch.unique(task_labels, return_counts=True)
            label_counts = {
                int(label.item()): int(count.item())
                for label, count in zip(unique_labels, counts)
            }

            # Number of classes is max_label_id + 1
            max_label_id = int(unique_labels.max().item())

            task_stats[0] = {
                'num_classes': max_label_id + 1,
                'label_counts': label_counts,
                'total_samples': int(counts.sum().item())
            }
        else:
            # Handle multi-task case
            for task_id in range(num_tasks):
                task_labels = labels[:, task_id]
                unique_labels, counts = torch.unique(task_labels, return_counts=True)

                label_counts = {
                    int(label.item()): int(count.item())
                    for label, count in zip(unique_labels, counts)
                }

                # Number of classes is max_label_id + 1
                max_label_id = int(unique_labels.max().item())

                task_stats[task_id] = {
                    'num_classes': max_label_id + 1,
                    'label_counts': label_counts,
                    'total_samples': int(counts.sum().item())
                }

        return task_stats

    def get_samples_per_label_per_task(self) -> Optional[Dict[int, Dict[int, int]]]:
        """
        Get the number of samples for each unique label for each classification task.

        Returns:
            Dictionary mapping task_id -> {label -> count},
            None if labels are not available

        Example:
            {
                0: {0: 150, 1: 200, 2: 175},  # Task 0: class 0 has 150 samples, etc.
                1: {0: 100, 1: 250, 2: 175}   # Task 1: class 0 has 100 samples, etc.
            }
        """
        labels = self.get_labels()
        if labels is None:
            return None

        num_tasks = self.get_num_classification_tasks()
        if num_tasks == 0:
            return None

        task_stats = {}

        # Handle single task case
        if num_tasks == 1:
            if len(labels.shape) == 1:
                task_labels = labels
            else:
                task_labels = labels[:, 0]

            unique_labels, counts = torch.unique(task_labels, return_counts=True)
            task_stats[0] = {
                int(label.item()): int(count.item())
                for label, count in zip(unique_labels, counts)
            }
        else:
            # Handle multi-task case
            for task_id in range(num_tasks):
                task_labels = labels[:, task_id]
                unique_labels, counts = torch.unique(task_labels, return_counts=True)

                task_stats[task_id] = {
                    int(label.item()): int(count.item())
                    for label, count in zip(unique_labels, counts)
                }

        return task_stats

    def get_label_statistics(self) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive label statistics for all tasks in the dataset.

        Returns:
            Dictionary containing detailed statistics, None if labels are not available

        Example return:
            {
                'num_tasks': 2,
                'num_samples': 1000,
                'per_task': {
                    0: {
                        'num_classes': 3,
                        'samples_per_class': {0: 150, 1: 200, 2: 175},
                        'class_percentages': {0: 15.0, 1: 20.0, 2: 17.5},
                        'unique_classes': [0, 1, 2],
                        'min_samples': 150,
                        'max_samples': 200,
                        'is_balanced': False
                    },
                    1: {...}
                }
            }
        """
        labels = self.get_labels()
        if labels is None:
            return None

        num_tasks = self.get_num_classification_tasks()
        num_samples = len(labels)
        samples_per_task = self.get_samples_per_label_per_task()

        if samples_per_task is None:
            return None

        statistics = {
            'num_tasks': num_tasks,
            'num_samples': num_samples,
            'per_task': {}
        }

        for task_id, label_counts in samples_per_task.items():
            unique_classes = sorted(label_counts.keys())
            counts = list(label_counts.values())
            min_samples = min(counts)
            max_samples = max(counts)

            # Number of classes is max_label_id + 1
            max_label_id = max(unique_classes)

            # Calculate class percentages
            class_percentages = {
                label: (count / num_samples) * 100
                for label, count in label_counts.items()
            }

            # Check if balanced (using 10% threshold)
            is_balanced = (max_samples - min_samples) / max_samples < 0.1 if max_samples > 0 else True

            statistics['per_task'][task_id] = {
                'num_classes': max_label_id + 1,
                'samples_per_class': label_counts,
                'class_percentages': class_percentages,
                'unique_classes': unique_classes,
                'min_samples': min_samples,
                'max_samples': max_samples,
                'is_balanced': is_balanced
            }

        return statistics