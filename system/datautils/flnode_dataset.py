
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch

class FLNodeDataset(Dataset):
    def __init__(self, data, transform=None, target_transform=None, device = 'cpu', **kwargs):
        self.kwargs = kwargs   
        self.train_data = None
        self.train_samples = 0
        self.test_data = None
        self.test_samples = 0
        self.labels = None
        self.labels_count = None
        self.labels_percent = None
        self.transform = transform
        self.target_transform = target_transform
        self.data = data
        self.device = device

    def to(self, device):
        self.device = device
        return self
    
    def __len__(self):
        if type(self.data) == dict:
            return len(self.data['samples'])

        return len(self.data)
    
    def __getitem__(self, idx):
        if type(self.data) == dict:
            sample = self.data['samples'][idx].to(self.device)
            label = {}
            for k in self.data.keys():
                if k != 'samples':
                    label[k] = self.data[k][idx].to(self.device)
        else:
            sample = self.data[idx][0].to(self.device)
            label = self.data[idx][1].to(self.device)
        if sample.shape[0] > 3:
            sample = sample.moveaxis(2,0)
        if self.transform != None:
            sample_transforms = len(self.transform.transforms)
            if sample.shape[2] != self.transform.transforms[sample_transforms-1].size[0]:
                # print ( "Transforming sample ", idx, " with shape ", data.shape)
                # data = data.to('cpu')
                transformed_data = self.transform(sample)
                if (transformed_data.shape[0] == 1):
                    transformed_data = transformed_data.expand(3, transformed_data.shape[1], transformed_data.shape[2])
                # data.to(self.device)
                if type(self.data) == dict:
                    temp = list(self.data['samples'])
                    # temp[idx] = transformed_data.detach().cpu()
                    # self.data['samples'] = torch.Tensor(temp)
                    # self.data['samples'][idx] = torch.Tensor(transformed_data.detach().cpu()
                    sample = transformed_data
                else:
                    temp = list(self.data[idx])
                    temp[0] = transformed_data.detach().cpu()
                    self.data[idx] = tuple(temp)
                    sample = transformed_data
            else:
                t = 1
            # data = data.to(self.device)
        return sample, label