from utils.data_utils import read_client_data
from torch.utils.data import DataLoader, Dataset
from datautils.flnode_dataset import FLNodeDataset

import wandb
class NodeData():
    def __init__(self, args, id = -1, transform=None, target_transform=None, **kwargs):
        self.id = id
        self.args = args
        self.kwargs = kwargs   
        self.train_data = None
        self.train_samples = 0
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
        return self
    
    def load_train_data(self, batch_size, dataset_limit=0):
        if self.train_data == None:
            print("Loading train data for client %d" % self.id)
            self.train_data = read_client_data(self.dataset, self.id, is_train=True,dataset_limit=dataset_limit)
            self.train_samples = len(self.train_data)
        self.train_dataset = FLNodeDataset(self.train_data, transform=self.transform, target_transform=self.target_transform, device=self.device)
        return DataLoader(self.train_dataset, batch_size, drop_last=False, shuffle=True)
   
    def load_test_data(self, batch_size, dataset_limit=0):
        if self.test_data == None:
            print("Loading test data for client %d" % self.id)
            self.test_data = read_client_data(self.dataset, self.id, is_train=False,dataset_limit=dataset_limit)
            self.test_samples = len(self.test_data)
        self.test_dataset = FLNodeDataset(self.test_data, transform=self.transform, target_transform=self.target_transform, device=self.device)
        return DataLoader(self.test_dataset, batch_size, drop_last=False, shuffle=True)


    def unload_train_data(self):
        self.train_data = None
        self.train_dataset = None

    def unload_test_data(self):
        self.test_data = None
        self.test_dataset = None
         
    def stats_get(self):
        # labels = self.labels_get()
        self.load_train_data(1)

        labels = list(range(self.num_classes))
        if self.labels_count == None or self.labels_percent == None:
            self.labels_count = dict(zip(labels, [0]*len(labels)))
            for _,l in self.train_data:
                self.labels_count[l.item()] += 1
            self.labels_percent = {k: v*100/self.train_samples for k,v in self.labels_count.items()}

        return self.labels_count, self.labels_percent
    
    def test_stats_get(self):
        # labels = self.labels_get()
        self.load_test_data(1)
        labels = list(range(self.num_classes))
        if self.labels_count == None or self.labels_percent == None:
            self.labels_count = dict(zip(labels, [0]*len(labels)))
            for _,l in self.test_data:
                self.labels_count[l.item()] += 1
            self.labels_percent = {k: v*100/self.test_samples for k,v in self.labels_count.items()}

        return self.labels_count, self.labels_percent
    
    # def stats_wandb_define(self):
    #     wandb.define_metric(f"dataset/client_{self.id}_train_labels", step_metric="class")
    #     wandb.define_metric(f"dataset/client_{self.id}_test_labels", step_metric="class")
       
    def stats_wandb_log(self):
        labels_count, labels_percent = self.stats_get()
        test_labels_count, test_labels_percent = self.test_stats_get()
        # labels = len(labels_count)
        # data = []

        # for label in range(labels):
        #     data.append([label, labels_count[label], test_labels_count[label]])
        data = [[label, count] for (label, count) in labels_count.items()]
        table = wandb.Table(data=data, columns=["class", "count"])
        wandb.log({f"dataset/client_{self.id}_train_labels" : wandb.plot.bar(table, "class",
            "count", title=f"Client {self.id} Train class count")})
        
        data = [[label, count] for (label, count) in test_labels_count.items()]
        table = wandb.Table(data=data, columns=["class", "count"])
        wandb.log({f"dataset/client_{self.id}_test_labels" : wandb.plot.bar(table, "class",
           "count", title=f"Client {self.id} Test class count")})
        # for k,v in labels_count.items():
        #     wandb.log({f"dataset/client_{self.id}_train_labels": v, "class":k})
        # labels_count, labels_percent = self.test_stats_get()
        # for k,v in labels_count.items():
        #     wandb.log({f"dataset/client_{self.id}_test_labels": v, "class":k})

    def stats_dump(self):
        labels_count, labels_percent = self.stats_get()
        print("Dataset %d stats: %s" % (self.id,labels_count))
        # print("Labels percent: %s" % labels_percent)

    def labels_get(self):
        if self.labels == None:
            self.labels = set([l.item() for t,l in self.train_data])
        return self.labels