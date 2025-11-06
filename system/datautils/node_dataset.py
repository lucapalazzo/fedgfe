import numpy as np
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader, Dataset, random_split
from datautils.flnode_dataset import FLNodeDataset
from datautils.fl_splitted_dataset import FLSplittedDataset

import torch
import os

import wandb
class NodeData():
    def __init__(self, args, node_id = -1, dataset_split_id = -1, transform=None, target_transform=None,
                 custom_train_dataset=None, custom_test_dataset=None, dataset=None, **kwargs):
        self.id = node_id
        self.split_id = dataset_split_id
        self.args = args
        self.kwargs = kwargs
        self.train_data = None
        self.train_samples = 0
        self.train_dataloader = None
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

        # Support for custom datasets that inherit from Dataset
        self.custom_train_dataset = custom_train_dataset
        self.custom_test_dataset = custom_test_dataset

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
        else:
            self.test_dataset = None

        # Check if we should use FLSplittedDataset for federated splits
        self.use_fl_splitted = self._should_use_fl_splitted_dataset()
        
        if self.dataset is not None:
            self.train_dataset, self.test_dataset = random_split(self.dataset, [int(0.8*len(self.dataset)), len(self.dataset) - int(0.8*len(self.dataset))])

        self.device = args.device

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
        return "Node %d split id %d dataset %s train samples %d test samples %d" % ( self.id, self.split_id, self.dataset, self.train_samples, self.test_samples )

    def to(self, device):
        self.device = device
        if self.train_dataset != None:
            self.train_dataset.to(device)
        if self.test_dataset != None:
            self.test_dataset.to(device)
        if self.train_data != None:
            if type(self.train_data) == dict:
                for k,v in self.train_data.items():
                    if type(v) == torch.Tensor:
                        self.train_data[k] = v.to(device)
            else:
                for i in range(len(self.train_data)):
                    self.train_data[i] = (self.train_data[i][0].to(device), self.train_data[i][1].to(device))
        if self.test_data != None:
            if type(self.test_data) == dict:
                for k,v in self.test_data.items():
                    if type(v) == torch.Tensor:
                        self.test_data[k] = v.to(device)
            else:
                for i in range(len(self.test_data)):
                    self.test_data[i] = (self.test_data[i][0].to(device), self.test_data[i][1].to(device))
        return self
    
    def classification_labels_count(self):
        dataloader = self.load_train_data(1)
        if self.train_data == None:
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
        if self.train_data == None:
            return None
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
                self.train_dataloader = DataLoader(self.train_dataset, batch_size, drop_last=False, shuffle=True)
                return self.train_dataloader

        # If using a custom dataset (like VEGASDataset), directly create dataloader
        if self.train_dataset is not None:
            print(f"Loading custom train dataset for client {self.id} with {self.train_samples} samples")
            self.train_dataloader = DataLoader(self.train_dataset, batch_size, drop_last=False, shuffle=True)
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
                self.train_dataloader = DataLoader(self.train_dataset, batch_size, drop_last=False, shuffle=True)
            elif self.train_dataloader.batch_size != batch_size:
                self.train_dataloader = DataLoader(self.train_dataset, batch_size, drop_last=False, shuffle=True)

        return self.train_dataloader
   
    def load_test_data(self, batch_size, dataset_limit=0,dataset_dir_prefix= ""):
        # Check if we can use existing dataloader
        if self.test_dataloader is not None and self.test_dataloader.batch_size == batch_size:
            return self.test_dataloader

        # Try to use FLSplittedDataset if applicable (test dataset might be created along with train)
        if self.use_fl_splitted and self.test_dataset is None:
            if self._create_fl_splitted_datasets():
                self.test_dataloader = DataLoader(self.test_dataset, batch_size, drop_last=False, shuffle=False)
                return self.test_dataloader

        # If using a custom dataset (like VEGASDataset), directly create dataloader
        if self.test_dataset is not None:
            print(f"Loading custom test dataset for client {self.id} with {self.test_samples} samples")
            self.test_dataloader = DataLoader(self.test_dataset, batch_size, drop_last=False, shuffle=False)
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
            self.test_dataloader = DataLoader(self.test_dataset, batch_size, drop_last=False, shuffle=True)
        elif self.test_dataloader.batch_size != batch_size:
            self.test_dataloader = DataLoader(self.test_dataset, batch_size, drop_last=False, shuffle=True)
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

    def unload_test_data(self):
        self.test_data = None
        self.test_dataset = None

    def dataset_stats_get(self):

        train_stats = self.train_stats_get()
        test_stats = self.test_stats_get()
        task_count = len(train_stats[0])

        # print ( "Dataset stats: ")
        # for task in range(task_count):
        #     print ( f"Task {task} train: {train_stats[0][task]} test: {test_stats[0][task]}" )
        return train_stats, test_stats
    
    def stats_get(self, data = None):

        if data == None:
            return [],[]
        
        if type(data) == Dataset:
            print ("Dataset object provided, cannot compute stats")
        
        if isinstance(data, Dataset):
            # è un'istanza di Dataset (o di una sua sottoclasse)
            print("Dataset object provided, cannot compute stats")
            dataset_length = len(data)
            print(f"Dataset length: {dataset_length}")
            print(f"")
            return [],[]
            if 'labels' in data:
                labels = data['labels']
            else:
                return [],[]
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
        if self.dataset != None:
            print("Using Dataset obkect id %d" % (self.id))
        elif self.train_data == None:
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
        else:
            print("Dataset %d stats: %s" % (self.id,train_stats))
        # print("Labels percent: %s" % labels_percent)

    def labels_get(self):
        if self.labels == None:
            self.labels = set([l.item() for t,l in self.train_data])
        return self.labels