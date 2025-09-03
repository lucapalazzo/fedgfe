from utils.data_utils import read_client_data
from torch.utils.data import DataLoader, Dataset
from datautils.flnode_dataset import FLNodeDataset

import torch

import wandb
class NodeData():
    def __init__(self, args, id = -1, transform=None, target_transform=None, **kwargs):
        self.id = id
        self.dataset_id = id
        self.args = args
        self.kwargs = kwargs   
        self.train_data = None
        self.train_samples = 0
        self.train_dataloader = None
        self.test_dataloader = None
        self.test_data = None
        self.test_samples = 0
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.labels = None
        self.labels_count = None
        self.labels_percent = None
        self.transform = transform
        self.target_transform = target_transform
        self.train_dataset = None
        self.test_dataset = None
        self.device = args.device

    def to(self, device):
        self.device = device
        if self.train_dataset != None:
            self.train_dataset.to(device)
        if self.test_dataset != None:
            self.test_dataset.to(device)
        if self.train_data != None:
            if type(self.train_data) == dict:
                for k,v in self.train_data.items():
                    self.train_data[k] = v.to(device)
            else:
                for i in range(len(self.train_data)):
                    self.train_data[i] = (self.train_data[i][0].to(device), self.train_data[i][1].to(device))
        if self.test_data != None:
            if type(self.test_data) == dict:
                for k,v in self.test_data.items():
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
        if self.train_data == None:
            print("Loading train data for client %d dataset id %d" % ( self.id, self.dataset_id ) )
            self.train_data = read_client_data(self.dataset, self.dataset_id, is_train=True,dataset_limit=dataset_limit, prefix=prefix, dataset_dir_prefix=dataset_dir_prefix)
            if self.train_data == None:
                return None
            memory_footprint = 0
            if type(self.train_data) == dict:
                for k,v in self.train_data.items():
                    memory_footprint += v.element_size() * v.nelement()
                self.train_samples = len(self.train_data['samples'])
            else:
                self.train_samples = len(self.train_data)
                for i in range(len(self.train_data)):
                    memory_footprint += self.train_data[i][0].element_size() * self.train_data[i][0].nelement() + self.train_data[i][1].element_size() * self.train_data[i][1].nelement()
            print("Client %d train data memory footprint: %d" % (self.id, memory_footprint))
         
        # self.train_dataset = FLNodeDataset(self.train_data, transform=self.transform, target_transform=self.target_transform, device=self.device)
        if self.train_dataset == None:
            self.train_dataset = FLNodeDataset(self.train_data, transform=self.transform, target_transform=self.target_transform)
        if self.train_dataloader == None:
            self.train_dataloader = DataLoader(self.train_dataset, batch_size, drop_last=False, shuffle=True)
        elif self.train_dataloader.batch_size != batch_size:
            self.train_dataloader = DataLoader(self.train_dataset, batch_size, drop_last=False, shuffle=True)
        return self.train_dataloader
   
    def load_test_data(self, batch_size, dataset_limit=0,dataset_dir_prefix= ""):
        if self.test_data == None:
            print("Loading test data for client %d dataset id %d" % ( self.id, self.dataset_id ) )
            self.test_data = read_client_data(self.dataset, self.dataset_id, is_train=False,dataset_limit=dataset_limit, dataset_dir_prefix=dataset_dir_prefix)
            if self.test_data == None:
                return None
            memory_footprint = 0
            if type(self.train_data) == dict:
                for k,v in self.train_data.items():
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
        
        if type(data) == dict:
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
        if self.train_data == None:
            self.load_train_data(1)
        
        return self.stats_get( self.train_data )
    
    def test_stats_get(self):
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
        if self.test_data == None:
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