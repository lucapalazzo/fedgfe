import numpy as np
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader, Dataset, random_split, ConcatDataset
from datautils.flnode_dataset import FLNodeDataset
from datautils.fl_splitted_dataset import FLSplittedDataset
from sklearn.model_selection import train_test_split

import torch
import os

import wandb
class NodeData():
    def __init__(self, args, node_id = -1, dataset_split_id = -1, transform=None, target_transform=None,
                 custom_train_dataset=None, custom_test_dataset=None, custom_val_dataset=None,
                 dataset=None, split_ratio=0.8, val_ratio=0.1, stratify=True, collate_fn=None, **kwargs):
        self.id = node_id
        self.split_id = dataset_split_id
        self.args = args
        self.kwargs = kwargs
        self.train_data = None
        self.train_samples = 0
        self.train_dataloader = None
        self.val_data = None
        self.val_samples = 0
        self.val_dataloader = None
        self.test_dataloader = None
        self.test_data = None
        self.test_samples = 0
        self.dataset_name = args.dataset
        self.dataset = dataset
        self.num_classes = args.num_classes
        self.labels = None
        self.labels_count = None
        self.labels_percent = None
        self.transform = transform
        self.target_transform = target_transform
        self.collate_fn = collate_fn

        # Split configuration
        self.split_ratio = split_ratio
        self.val_ratio = val_ratio
        self.stratify = stratify

        self.classification_tasks_count = 0
        self.segmentation_task_count = 0

        # Support for custom datasets that inherit from Dataset
        self.custom_train_dataset = custom_train_dataset
        self.custom_test_dataset = custom_test_dataset
        self.custom_val_dataset = custom_val_dataset

        # If custom datasets are provided, use them directly
        if custom_train_dataset is not None:
            self.train_dataset = custom_train_dataset
            self.train_samples = len(custom_train_dataset) if hasattr(custom_train_dataset, '__len__') else 0
            self.use_custom_dataset = True
        else:
            self.train_dataset = None
            self.use_custom_dataset = False

        if custom_test_dataset is not None:
            self.test_dataset = custom_test_dataset
            self.test_samples = len(custom_test_dataset) if hasattr(custom_test_dataset, '__len__') else 0
            self.use_custom_dataset = True
        else:
            self.test_dataset = None

        if custom_val_dataset is not None:
            self.val_dataset = custom_val_dataset
            self.val_samples = len(custom_val_dataset) if hasattr(custom_val_dataset, '__len__') else 0
            self.use_custom_dataset = True
        else:
            self.val_dataset = None

        # Check if we should use FLSplittedDataset for federated splits
        self.use_fl_splitted = self._should_use_fl_splitted_dataset()

        # Handle automatic splitting if dataset is provided without custom splits
        if self.dataset is not None and custom_train_dataset is None:
            self._split_dataset_with_stratification()

        self.device = args.device

    def _split_dataset_with_stratification(self):
        """Split dataset into train/val/test with optional stratification."""
        if self.dataset is None:
            return

        # Get collate_fn from dataset if not already set
        if self.collate_fn is None and hasattr(self.dataset, 'get_collate_fn'):
            self.collate_fn = self.dataset.get_collate_fn()

        dataset_size = len(self.dataset)

        # Calculate split sizes
        test_size = int(dataset_size * (1.0 - self.split_ratio))
        train_val_size = dataset_size - test_size
        val_size = int(dataset_size * self.val_ratio)
        train_size = train_val_size - val_size

        if self.stratify:
            try:
                # Try to extract labels for stratification
                labels = []
                for i in range(len(self.dataset)):
                    sample = self.dataset[i]
                    if isinstance(sample, (tuple, list)) and len(sample) > 1:
                        label = sample[1]
                    elif isinstance(sample, dict) and 'label' in sample:
                        label = sample['label']
                    elif isinstance(sample, dict) and 'labels' in sample:
                        label = sample['labels']
                    else:
                        # Cannot extract labels, fall back to random split
                        raise ValueError("Cannot extract labels for stratification")

                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    labels.append(label)

                # Create indices for stratified split
                indices = list(range(dataset_size))

                # First split: train+val vs test
                train_val_indices, test_indices = train_test_split(
                    indices,
                    test_size=test_size,
                    stratify=[labels[i] for i in indices],
                    random_state=42 + self.id
                )

                # Second split: train vs val
                if val_size > 0:
                    train_val_labels = [labels[i] for i in train_val_indices]
                    train_indices, val_indices = train_test_split(
                        train_val_indices,
                        test_size=val_size,
                        stratify=train_val_labels,
                        random_state=42 + self.id
                    )
                else:
                    train_indices = train_val_indices
                    val_indices = []

                # Create subset datasets
                from torch.utils.data import Subset

                dataset_train = getattr ( self.dataset, 'train', None )
                dataset_test = getattr ( self.dataset, 'test', None )
                dataset_val = getattr ( self.dataset, 'val', None )

                if dataset_train != None:
                    self.train_dataset = dataset_train
                else:
                    self.train_dataset = Subset(self.dataset, train_indices)

                if dataset_test != None:
                    self.test_dataset = dataset_test
                else:
                    self.test_dataset = Subset(self.dataset, test_indices)
                if val_size > 0 and dataset_val != None:
                    self.val_dataset = dataset_val
                elif val_size > 0:
                    self.val_dataset = Subset(self.dataset, val_indices)

                self.train_samples = len(self.train_dataset)
                self.test_samples = len(self.test_dataset)
                self.val_samples = len(self.val_dataset) if val_size > 0 else 0

                print(f"Stratified split: train={self.train_samples}, val={self.val_samples}, test={self.test_samples}")

            except (ValueError, Exception) as e:
                # Fall back to random split if stratification fails
                print(f"Stratification failed ({e}), using random split")
                self._random_split()
        else:
            self._random_split()

    def _random_split(self):
        """Perform random split without stratification."""
        dataset_size = len(self.dataset)

        # Calculate split sizes
        test_size = int(dataset_size * (1.0 - self.split_ratio))
        train_val_size = dataset_size - test_size
        val_size = int(dataset_size * self.val_ratio)
        train_size = train_val_size - val_size

        if val_size > 0:
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.dataset,
                [train_size, val_size, test_size]
            )
            self.val_samples = len(self.val_dataset)
        else:
            self.train_dataset, self.test_dataset = random_split(
                self.dataset,
                [train_val_size, test_size]
            )
            self.val_samples = 0

        self.train_samples = len(self.train_dataset)
        self.test_samples = len(self.test_dataset)

        print(f"Random split: train={self.train_samples}, val={self.val_samples}, test={self.test_samples}")

    def _should_use_fl_splitted_dataset(self):
        """Check if we should use FLSplittedDataset instead of legacy loading."""
        if self.use_custom_dataset:
            return False

        # Check if federated dataset directory structure exists
        dataset_dir = getattr(self.args, 'dataset_dir_prefix', '')
        dataset_path = os.path.join(dataset_dir, 'dataset', self.dataset_name)

        # Check if train/test split directories exist
        train_dir = os.path.join(dataset_path, 'train')
        test_dir = os.path.join(dataset_path, 'test')

        return (os.path.exists(train_dir) and os.path.exists(test_dir) and
                any(f.endswith('.npz') for f in os.listdir(train_dir)) if os.path.isdir(train_dir) else False)

    def _create_fl_splitted_datasets(self):
        """Create FLSplittedDataset instances for train and test."""
        if not self.use_fl_splitted:
            return False

        dataset_dir = getattr(self.args, 'dataset_dir_prefix', '')
        dataset_path = os.path.join(dataset_dir, 'dataset', self.dataset_name)

        try:
            # Create train dataset
            if self.train_dataset is None:
                self.train_dataset = FLSplittedDataset(
                    dataset_path=dataset_path,
                    node_id=self.split_id,
                    is_train=True,
                    transform=self.transform,
                    target_transform=self.target_transform,
                    device=self.device
                )
                self.train_samples = len(self.train_dataset)
                print(f"Created FLSplittedDataset for train: {self.train_samples} samples")

            # Create test dataset
            if self.test_dataset is None:
                self.test_dataset = FLSplittedDataset(
                    dataset_path=dataset_path,
                    node_id=self.split_id,
                    is_train=False,
                    transform=self.transform,
                    target_transform=self.target_transform,
                    device=self.device
                )
                self.test_samples = len(self.test_dataset)
                print(f"Created FLSplittedDataset for test: {self.test_samples} samples")

            return True
        except Exception as e:
            print(f"Failed to create FLSplittedDataset: {e}")
            return False

    def __str__(self):
        return "Node %d split id %d dataset %s train samples %d val samples %d test samples %d" % (
            self.id, self.split_id, self.dataset_name, self.train_samples, self.val_samples, self.test_samples )

    def to(self, device):
        """
        Set the target device for this NodeData.

        IMPORTANT: This method does NOT move bulk data to GPU.
        - For Dataset objects (train_dataset, etc.): we call their .to() method which
          should only set device preference, not move data
        - For preloaded data (train_data, etc.): we keep it on CPU to prevent OOM

        Data is moved to GPU batch-by-batch by the DataLoader during training.
        """
        self.device = device

        # Propagate device to Dataset objects (modern approach)
        # These should only set device preference, not move data
        if self.train_dataset is not None and hasattr(self.train_dataset, 'to'):
            self.train_dataset.to(device)
        if self.val_dataset is not None and hasattr(self.val_dataset, 'to'):
            self.val_dataset.to(device)
        if self.test_dataset is not None and hasattr(self.test_dataset, 'to'):
            self.test_dataset.to(device)

        # MEMORY LEAK FIX: Do NOT move train_data/val_data/test_data to GPU!
        # These are legacy preloaded tensors that should stay on CPU.
        # Only batches from dataloader should be moved to GPU during training.
        # The original code (lines 275-304) was causing massive GPU memory leaks
        # by loading entire datasets into GPU memory.

        return self
    
    def classification_labels_count(self):
        dataloader = self.load_train_data(1)
        if self.train_dataset == None:
            return None
        
        tasks_count = 0
        for _,l in dataloader:
            if type(l) == dict:
                if 'labels' in l:
                    if len(l['labels'].shape) == 1:
                        tasks_count = 1
                    else:
                        tasks_count = l['labels'].shape[1]
                else:
                    tasks_count = 1
            else:
                if len(l.shape) == 1:
                    tasks_count = 1
                else:
                    tasks_count = l.shape[1]
            break
        return tasks_count
    
    def segmentation_mask_count(self):
        dataloader = self.load_train_data(1)
        if self.train_dataset == None:
            return 0
        
        return self.train_dataset.get_num_segmentation_tasks()
        mask_count = 0
        for _,l in dataloader:
            if type(l) == dict:
                    if 'semantic_masks' in l:
                        mask_count = l['semantic_masks'].shape[1]
                    elif 'masks' in l:
                        shape = len(l['masks'].shape)
                        if shape == 4:
                            for s in range(1,shape):
                                if l['masks'].shape[s] < 32:
                                    mask_count = l['masks'].shape[s]
                                    break
                        else:
                            mask_count = 1
            else:
                mask_count += l.shape[2]
            break
        return mask_count
    
    def load_train_data(self, batch_size, dataset_limit=0, prefix = "",dataset_dir_prefix= ""):
        # Check if we can use existing dataloader
        if self.train_dataloader is not None and self.train_dataloader.batch_size == batch_size:
            return self.train_dataloader

        # Try to use FLSplittedDataset if applicable
        if self.use_fl_splitted and self.train_dataset is None:
            if self._create_fl_splitted_datasets():
                self.train_dataloader = DataLoader(
                    self.train_dataset,
                    batch_size,
                    drop_last=False,
                    shuffle=True,
                    collate_fn=self.collate_fn
                )
                return self.train_dataloader

        # If using a custom dataset (like VEGASDataset), directly create dataloader
        if self.train_dataset is not None:
            print(f"Creating dataloader for train dataset for client {self.id} with {len(self.train_dataset)} samples")
            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size,
                drop_last=False,
                shuffle=True,
                collate_fn=self.collate_fn
            )
            return self.train_dataloader

        # Original implementation for npz files
        if self.train_data == None:
            print("Loading train data for client %d dataset id %d" % ( self.id, self.split_id ) )
            self.train_data = read_client_data(self.dataset, self.split_id, is_train=True,dataset_limit=dataset_limit, prefix=prefix, dataset_dir_prefix=dataset_dir_prefix)
            if self.train_data == None:
                return None
            memory_footprint = 0
            if type(self.train_data) == dict:
                for k,v in self.train_data.items():
                    if type(v) == torch.Tensor:
                        memory_footprint += v.element_size() * v.nelement()
                self.train_samples = len(self.train_data['samples'])
            else:
                self.train_samples = len(self.train_data)
                for i in range(len(self.train_data)):
                    memory_footprint += self.train_data[i][0].element_size() * self.train_data[i][0].nelement() + self.train_data[i][1].element_size() * self.train_data[i][1].nelement()
            print("Client %d train data memory footprint: %d" % (self.id, memory_footprint))
        else:
            # self.train_dataset = FLNodeDataset(self.train_data, transform=self.transform, target_transform=self.target_transform, device=self.device)
            if self.train_dataset == None:
                self.train_dataset = FLNodeDataset(self.train_data, transform=self.transform, target_transform=self.target_transform)
            if self.train_dataloader == None:
                self.train_dataloader = DataLoader(
                    self.train_dataset,
                    batch_size,
                    drop_last=False,
                    shuffle=True,
                    collate_fn=self.collate_fn
                )
            elif self.train_dataloader.batch_size != batch_size:
                self.train_dataloader = DataLoader(
                    self.train_dataset,
                    batch_size,
                    drop_last=False,
                    shuffle=True,
                    collate_fn=self.collate_fn
                )

        return self.train_dataloader

    def load_val_data(self, batch_size, dataset_limit=0, prefix="", dataset_dir_prefix=""):
        """Load validation data and create dataloader."""
        # Check if we can use existing dataloader
        if self.val_dataloader is not None and self.val_dataloader.batch_size == batch_size:
            return self.val_dataloader

        # If using a custom dataset (like VEGASDataset), directly create dataloader
        if self.val_dataset is not None:
            print(f"Creating dataloader for val dataset for client {self.id} with {len(self.val_dataset)} samples")
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size,
                drop_last=False,
                shuffle=False,
                collate_fn=self.collate_fn
            )
            return self.val_dataloader

        # If no validation dataset exists, return None
        print(f"No validation dataset available for client {self.id}")
        return None

    def load_test_data(self, batch_size, dataset_limit=0,dataset_dir_prefix= ""):
        # Check if we can use existing dataloader
        if self.test_dataloader is not None and self.test_dataloader.batch_size == batch_size:
            return self.test_dataloader

        # Try to use FLSplittedDataset if applicable (test dataset might be created along with train)
        if self.use_fl_splitted and self.test_dataset is None:
            if self._create_fl_splitted_datasets():
                self.test_dataloader = DataLoader(
                    self.test_dataset,
                    batch_size,
                    drop_last=False,
                    shuffle=False,
                    collate_fn=self.collate_fn
                )
                return self.test_dataloader

        # If using a custom dataset (like VEGASDataset), directly create dataloader
        if self.test_dataset is not None:
            print(f"Creating dataloader for test dataset for client {self.id} with {len(self.test_dataset)} samples")
            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size,
                drop_last=False,
                shuffle=False,
                collate_fn=self.collate_fn
            )
            return self.test_dataloader

        # Original implementation for npz files
        if self.test_data == None:
            print("Loading test data for client %d dataset id %d" % ( self.id, self.split_id ) )
            self.test_data = read_client_data(self.dataset, self.split_id, is_train=False,dataset_limit=dataset_limit, dataset_dir_prefix=dataset_dir_prefix)
            if self.test_data == None:
                return None
            memory_footprint = 0
            if type(self.train_data) == dict:
                for k,v in self.train_data.items():
                    if type(v) == torch.Tensor:
                        memory_footprint += v.element_size() * v.nelement()
                self.train_samples = len(self.train_data['samples'])
            else:
                for i in range(len(self.test_data)):
                    memory_footprint += self.test_data[i][0].element_size() * self.test_data[i][0].nelement() + self.test_data[i][1].element_size() * self.test_data[i][1].nelement()
            print("Client %d test data memory footprint: %d" % (self.id, memory_footprint))
            self.test_samples = len(self.test_data)
        if self.test_dataset == None:
            self.test_dataset = FLNodeDataset(self.test_data, transform=self.transform, target_transform=self.target_transform)

        if self.test_dataloader == None:
            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size,
                drop_last=False,
                shuffle=True,
                collate_fn=self.collate_fn
            )
        elif self.test_dataloader.batch_size != batch_size:
            self.test_dataloader = DataLoader(
                self.test_dataset,
                batch_size,
                drop_last=False,
                shuffle=True,
                collate_fn=self.collate_fn
            )
        return self.test_dataloader

    def get_train_dataset(self, dataset_limit=0):
        """
        Get the training dataset without creating a DataLoader.
        Used by DDP mixin for creating custom distributed DataLoaders.
        """
        # Try to use FLSplittedDataset if applicable
        if self.use_fl_splitted and self.train_dataset is None:
            self._create_fl_splitted_datasets()

        # If using a custom dataset, return it directly
        if self.use_custom_dataset and self.train_dataset is not None:
            return self.train_dataset

        # If we have a dataset from FLSplittedDataset, return it
        if self.train_dataset is not None:
            return self.train_dataset

        # Original implementation for npz files
        if self.train_data is None:
            self.train_data = read_client_data(self.dataset, self.split_id, is_train=True, dataset_limit=dataset_limit)
            if self.train_data is None:
                return None

            if type(self.train_data) == dict:
                self.train_samples = len(self.train_data['samples'])
            else:
                self.train_samples = len(self.train_data)

        if self.train_dataset is None:
            self.train_dataset = FLNodeDataset(self.train_data, transform=self.transform, target_transform=self.target_transform)

        return self.train_dataset

    def get_val_dataset(self, dataset_limit=0):
        """
        Get the validation dataset without creating a DataLoader.
        Used by DDP mixin for creating custom distributed DataLoaders.
        """
        # If using a custom dataset, return it directly
        if self.use_custom_dataset and self.val_dataset is not None:
            return self.val_dataset

        # If we have a validation dataset, return it
        if self.val_dataset is not None:
            return self.val_dataset

        # No validation dataset available
        return None

    def get_test_dataset(self, dataset_limit=0):
        """
        Get the test dataset without creating a DataLoader.
        Used by DDP mixin for creating custom distributed DataLoaders.
        """
        # Try to use FLSplittedDataset if applicable (might be created along with train)
        if self.use_fl_splitted and self.test_dataset is None:
            self._create_fl_splitted_datasets()

        # If using a custom dataset, return it directly
        if self.use_custom_dataset and self.test_dataset is not None:
            return self.test_dataset

        # If we have a dataset from FLSplittedDataset, return it
        if self.test_dataset is not None:
            return self.test_dataset

        # Original implementation for npz files
        if self.test_data is None:
            self.test_data = read_client_data(self.dataset, self.split_id, is_train=False, dataset_limit=dataset_limit)
            if self.test_data is None:
                return None

            self.test_samples = len(self.test_data)

        if self.test_dataset is None:
            self.test_dataset = FLNodeDataset(self.test_data, transform=self.transform, target_transform=self.target_transform)

        return self.test_dataset 

    def unload_train_data(self):
        self.train_data = None
        self.train_dataset = None

    def unload_val_data(self):
        self.val_data = None
        self.val_dataset = None

    def unload_test_data(self):
        self.test_data = None
        self.test_dataset = None

    def dataset_stats_get(self):

        train_tasks, train_stats = self.train_stats_get()
        test_tasks, test_stats = self.test_stats_get()
        return train_stats, test_stats
    
    def stats_get(self, data = None):

        if data == None:
            return [],[]
        
        if type(data) == Dataset:
            print ("Dataset object provided, cannot compute stats")

        if isinstance(data, FLSplittedDataset ):
            # self.classification_labels_count = data.get_samples_per_label_per_task()
            self.classification_tasks_count = data.get_num_classification_tasks()
            self.segmentation_task_count = data.get_num_segmentation_tasks()
            self.task_label_stats = data.get_task_label_stats()
            return self.classification_tasks_count, self.task_label_stats

        
        if isinstance(data, Dataset):
            # è un'istanza di Dataset (o di una sua sottoclasse)
            dataset_length = len(data)
        else:
            labels = data

        if 'labels' in data:
            labels = data['labels']
        else:
            labels = data
            return [],[]

        if len(data['labels'].shape) == 1:
            labels = data['labels'].unsqueeze(1)
        else:
            labels = data['labels']

        self.classification_tasks_count = labels.shape[1]

        self.classification_labels_count = [0]*self.classification_tasks_count
        self.labels_percent = [0]*self.classification_tasks_count

        for task_labels in range(self.classification_tasks_count):
            if self.classification_tasks_count > 1:
                unique_labels = torch.unique(labels[:,task_labels]).long()
            else:
                unique_labels = torch.unique(labels).long()
            self.classification_labels_count[task_labels] = dict(zip(unique_labels.long().numpy(), [0]*len(unique_labels.long().numpy())))

            for label in unique_labels:
                count = torch.sum(labels[:,task_labels] == label)
                self.classification_labels_count[task_labels][label.long().item()] = count.item()

            # self.classification_labels_count[classification_label] = dict(zip(labels[classification_label], [0]*len(labels)))
            # if type(self.train_data) == dict:
            #     if 'labels' in self.train_data:
            #         for l in self.train_data['labels']:
            #             if self.classification_labels_count > 1:
            #                 self.labels_count[classification_label][l[classification_label].long().item()] += 1
            #             else:
            #                 self.labels_count[classification_label][l.long().item()] += 1
            # else:
            #     for _,l in self.train_data:
            #         self.labels_count[l.item()] += 1
            self.labels_percent = {k: v*100/self.train_samples for k,v in self.classification_labels_count[task_labels].items()}

        return self.classification_labels_count, self.labels_percent
    
    def train_stats_get(self):
        if self.train_dataset != None:
            return self.stats_get( self.train_dataset )
        if self.train_data == None:
            self.load_train_data(1)

        return self.stats_get( self.train_data )

    def val_stats_get(self):
        """Get statistics for validation dataset."""
        if self.val_dataset != None:
            return self.stats_get( self.val_dataset )
        if self.val_data == None:
            # Try to load validation data
            val_loader = self.load_val_data(1)
            if val_loader is None:
                return [], []

        return self.stats_get( self.val_data ) if self.val_data is not None else ([], [])

    def test_stats_get(self):
        if self.test_dataset != None:
            return self.stats_get( self.test_dataset )
        if self.test_data == None:
            self.load_test_data(1)

        return self.stats_get( self.test_data )
       
    def stats_wandb_log(self):
        train_labels_count, labels_percent = self.train_stats_get()
        test_labels_count, test_labels_percent = self.test_stats_get()

        if train_labels_count == 0 or train_labels_count == 0:
            return
        # labels = len(labels_count)
        # data = []

        # for label in range(labels):
        #     data.append([label, labels_count[label], test_labels_count[label]])
        tasks_count = len(train_labels_count)
        for task_labels in range(tasks_count):
            task_data = []
            for label in train_labels_count[task_labels].keys():
                if label not in test_labels_count[task_labels]:
                    test_labels_count[task_labels][label] = 0
                else:
                    test_labels_count[task_labels][label] = test_labels_count[task_labels][label]
                task_data.append([label, train_labels_count[task_labels][label], test_labels_count[task_labels][label]])
            # data = [[label, train_labels_count[task_labels][label], test_labels_count[task_labels][label]] for label in train_labels_count[task_labels].keys()]
            table = wandb.Table(data=task_data, columns=["class", "train_count", "test_count"])

            train_bar = wandb.plot.bar(table, "class", "train_count", title=f"Client {self.id} task {task_labels} class count")
            test_bar = wandb.plot.bar(table, "class", "test_count", title=f"Client {self.id} task {task_labels} class count")
            # wandb.log({f"dataset/client_{self.id}_labels_{task_labels}" : {train_bar, test_bar}})
            # wandb.log({f"dataset/client_{self.id}_labels_{task_labels}" : 
            #            { wandb.plot.bar(table, "class", "train_count", title=f"Client {self.id} task {task_labels} class count"),
            #              wandb.plot.bar(table, "class", "test_count", title=f"Client {self.id} task {task_labels} class count")}})
                                                                                         
            # wandb.log({f"dataset/client_{self.id}_labels_{task_labels}" : wandb.plot.bar(table, "class",
            #     "train_count", title=f"Client {self.id} task {task_labels} class count")})
            # wandb.log({f"dataset/client_{self.id}_labels_{task_labels}" : wandb.plot.bar(table, "class",
            #     "test_count", title=f"Client {self.id} task {task_labels} class count")})
        # data = [[label, count] for (label, count) in train_labels_count.items()]
        # table = wandb.Table(data=data, columns=["class", "count"])
        # wandb.log({f"dataset/client_{self.id}_train_labels" : wandb.plot.bar(table, "class",
        #     "count", title=f"Client {self.id} Train class count")})
        
        # data = [[label, count] for (label, count) in test_labels_count.items()]
        # table = wandb.Table(data=data, columns=["class", "count"])
        # wandb.log({f"dataset/client_{self.id}_test_labels" : wandb.plot.bar(table, "class",
        #    "count", title=f"Client {self.id} Test class count")})
        # for k,v in labels_count.items():
        #     wandb.log({f"dataset/client_{self.id}_train_labels": v, "class":k})
        # labels_count, labels_percent = self.test_stats_get()
        # for k,v in labels_count.items():
        #     wandb.log({f"dataset/client_{self.id}_test_labels": v, "class":k})

    def stats_dump(self):
        # return
        if self.train_data == None:
            self.load_train_data(1)
        elif self.test_data == None:
            self.load_test_data(1)

        if self.dataset_stats_get() == None:
            print("No data to dump")
            return
        
        train_stats, test_stats = self.dataset_stats_get()
        if type(train_stats) == tuple:
            for task in range(len(train_stats[0])):
                print(f"Task {task} labels count train {train_stats[0][task]} test {test_stats[0][task]}")
        if type(train_stats) == dict:
            for task_id in train_stats.keys():
                print(f"Task {task_id} labels count train {train_stats[task_id]['num_classes']} test {test_stats[task_id]['num_classes']}")
        else:
            print("Dataset %d stats: %s" % (self.id,train_stats))
        # print("Labels percent: %s" % labels_percent)

    def labels_get(self):
        if self.labels == None:
            self.labels = set([l.item() for t,l in self.train_data])
        return self.labels

    def get_label_percentage_per_task(self, is_train=True):
        """
        Calcola la percentuale di incidenza per ogni label per ogni task.

        Args:
            is_train: Se True, calcola le statistiche sul training set, altrimenti sul test set

        Returns:
            Una lista di dizionari, uno per ogni task, dove ogni dizionario mappa
            label -> percentuale (0-100)

        Esempio:
            [
                {0: 25.5, 1: 30.2, 2: 44.3},  # Task 0
                {0: 15.0, 1: 50.0, 2: 35.0}   # Task 1
            ]
        """
        # Determina quale dataset usare
        dataset = self.train_dataset if is_train else self.test_dataset

        # Se il dataset è FLSplittedDataset, usa il suo metodo nativo
        if isinstance(dataset, FLSplittedDataset):
            return dataset.get_label_percentage_per_task()

        # Altrimenti, ottieni le statistiche attraverso il metodo esistente
        if is_train:
            labels_count, _ = self.train_stats_get()
        else:
            labels_count, _ = self.test_stats_get()

        # Se non ci sono dati o stats_get ha restituito liste vuote
        if not labels_count or labels_count == [] or labels_count == [0]:
            return []

        # Verifica se labels_count è una lista di dizionari (multi-task) o un singolo dizionario
        if not isinstance(labels_count, list):
            labels_count = [labels_count]

        # Calcola le percentuali per ogni task
        percentages_per_task = []

        for task_idx, task_labels_count in enumerate(labels_count):
            # Se task_labels_count non è un dizionario, salta
            if not isinstance(task_labels_count, dict):
                continue

            # Calcola il totale dei campioni per questo task
            total_samples = sum(task_labels_count.values())

            # Calcola le percentuali per ogni label
            task_percentages = {
                label: (count * 100.0 / total_samples) if total_samples > 0 else 0.0
                for label, count in task_labels_count.items()
            }

            percentages_per_task.append(task_percentages)

        return percentages_per_task

    def print_label_percentages(self, is_train=True):
        """
        Stampa le percentuali di incidenza per ogni label per ogni task in modo leggibile.

        Args:
            is_train: Se True, mostra le statistiche sul training set, altrimenti sul test set
        """
        dataset_type = "Train" if is_train else "Test"
        percentages = self.get_label_percentage_per_task(is_train)

        if not percentages:
            print(f"No {dataset_type.lower()} data available")
            return

        print(f"\n{dataset_type} Label Distribution for Node {self.id} (Split {self.split_id}):")
        print("=" * 70)

        for task_idx, task_percentages in enumerate(percentages):
            print(f"\nTask {task_idx}:")
            print("-" * 50)

            # Ordina le label per percentuale decrescente
            sorted_labels = sorted(task_percentages.items(), key=lambda x: x[1], reverse=True)

            for label, percentage in sorted_labels:
                bar_length = int(percentage / 2)  # Scala la barra a 50 caratteri max
                bar = "#" * bar_length
                print(f"  Label {label:3d}: {percentage:6.2f}% [{bar}]")

        print("=" * 70)

    @staticmethod
    def merge_datasets_from_nodes(node_data_list, split_type='train', return_dataset=False):
        """
        Merge datasets from multiple NodeData instances.

        Args:
            node_data_list: List of NodeData instances to merge
            split_type: Type of split to merge ('train', 'val', or 'test')
            return_dataset: If True, returns the merged Dataset; if False, returns list of datasets

        Returns:
            ConcatDataset if return_dataset=True, else list of individual datasets

        Example:
            # Merge training data from 3 nodes
            merged_dataset = NodeData.merge_datasets_from_nodes(
                [node0, node1, node2],
                split_type='train',
                return_dataset=True
            )
        """
        datasets = []

        for node_data in node_data_list:
            if split_type == 'train':
                dataset = node_data.get_train_dataset()
            elif split_type == 'val':
                dataset = node_data.get_val_dataset()
            elif split_type == 'test':
                dataset = node_data.get_test_dataset()
            else:
                raise ValueError(f"Invalid split_type: {split_type}. Must be 'train', 'val', or 'test'")

            if dataset is not None:
                datasets.append(dataset)
            else:
                print(f"Warning: Node {node_data.id} has no {split_type} dataset")

        if not datasets:
            raise ValueError(f"No {split_type} datasets found in provided NodeData instances")

        if return_dataset:
            return ConcatDataset(datasets)
        else:
            return datasets

    @staticmethod
    def create_merged_dataloader(node_data_list, split_type='train', batch_size=32,
                                 shuffle=True, num_workers=0, drop_last=False, collate_fn=None, **kwargs):
        """
        Create a DataLoader from merged datasets of multiple NodeData instances.

        Args:
            node_data_list: List of NodeData instances to merge
            split_type: Type of split to merge ('train', 'val', or 'test')
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle the data
            num_workers: Number of workers for data loading
            drop_last: Whether to drop the last incomplete batch
            collate_fn: Optional custom collate function
            **kwargs: Additional arguments to pass to DataLoader

        Returns:
            DataLoader with merged datasets

        Example:
            # Create merged training dataloader from multiple nodes
            merged_loader = NodeData.create_merged_dataloader(
                [node0, node1, node2],
                split_type='train',
                batch_size=32,
                shuffle=True
            )

            # Use in training loop
            for batch in merged_loader:
                # Training code...
                pass
        """
        merged_dataset = NodeData.merge_datasets_from_nodes(
            node_data_list,
            split_type=split_type,
            return_dataset=True
        )

        # Use collate_fn from first node if not provided
        if collate_fn is None and len(node_data_list) > 0:
            collate_fn = node_data_list[0].collate_fn

        dataloader = DataLoader(
            merged_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=collate_fn,
            **kwargs
        )

        total_samples = len(merged_dataset)
        num_nodes = len(node_data_list)
        print(f"Created merged {split_type} dataloader: {total_samples} samples from {num_nodes} nodes")

        return dataloader

    def merge_with_nodes(self, other_nodes, split_type='train', batch_size=32,
                        shuffle=True, num_workers=0, drop_last=False, **kwargs):
        """
        Instance method to merge this node's dataset with other nodes' datasets.

        Args:
            other_nodes: List of other NodeData instances to merge with
            split_type: Type of split to merge ('train', 'val', or 'test')
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle the data
            num_workers: Number of workers for data loading
            drop_last: Whether to drop the last incomplete batch
            **kwargs: Additional arguments to pass to DataLoader

        Returns:
            DataLoader with merged datasets including this node

        Example:
            # Node 0 merges its data with nodes 1 and 2
            merged_loader = node0.merge_with_nodes(
                [node1, node2],
                split_type='train',
                batch_size=32
            )
        """
        all_nodes = [self] + list(other_nodes)
        return NodeData.create_merged_dataloader(
            all_nodes,
            split_type=split_type,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            **kwargs
        )

    @staticmethod
    def create_mixed_split_dataloader(node_data_list, split_configs, batch_size=32,
                                     shuffle=True, num_workers=0, drop_last=False, collate_fn=None, **kwargs):
        """
        Create a DataLoader from mixed splits of multiple NodeData instances.
        Allows combining different splits from different nodes.

        Args:
            node_data_list: List of NodeData instances
            split_configs: List of split types corresponding to each node
                          e.g., ['train', 'val', 'train'] means:
                          - Use train split from node_data_list[0]
                          - Use val split from node_data_list[1]
                          - Use train split from node_data_list[2]
            batch_size: Batch size for DataLoader
            shuffle: Whether to shuffle the data
            num_workers: Number of workers for data loading
            drop_last: Whether to drop the last incomplete batch
            collate_fn: Optional custom collate function
            **kwargs: Additional arguments to pass to DataLoader

        Returns:
            DataLoader with merged datasets from specified splits

        Example:
            # Combine train from node0, val from node1, test from node2
            mixed_loader = NodeData.create_mixed_split_dataloader(
                [node0, node1, node2],
                split_configs=['train', 'val', 'test'],
                batch_size=32
            )
        """
        if len(node_data_list) != len(split_configs):
            raise ValueError("node_data_list and split_configs must have the same length")

        datasets = []

        for node_data, split_type in zip(node_data_list, split_configs):
            if split_type == 'train':
                dataset = node_data.get_train_dataset()
            elif split_type == 'val':
                dataset = node_data.get_val_dataset()
            elif split_type == 'test':
                dataset = node_data.get_test_dataset()
            else:
                raise ValueError(f"Invalid split_type: {split_type}. Must be 'train', 'val', or 'test'")

            if dataset is not None:
                datasets.append(dataset)
                print(f"Adding {split_type} split from Node {node_data.id}: {len(dataset)} samples")
            else:
                print(f"Warning: Node {node_data.id} has no {split_type} dataset")

        if not datasets:
            raise ValueError("No datasets found with the specified split configurations")

        merged_dataset = ConcatDataset(datasets)

        # Use collate_fn from first node if not provided
        if collate_fn is None and len(node_data_list) > 0:
            collate_fn = node_data_list[0].collate_fn

        dataloader = DataLoader(
            merged_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            collate_fn=collate_fn,
            **kwargs
        )

        total_samples = len(merged_dataset)
        num_nodes = len(node_data_list)
        print(f"Created mixed split dataloader: {total_samples} samples from {num_nodes} nodes")

        return dataloader