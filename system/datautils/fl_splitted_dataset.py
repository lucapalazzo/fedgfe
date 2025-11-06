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