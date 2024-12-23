
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

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
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx][0].to(self.device)
        label = self.data[idx][1].to(self.device)
        if data.shape[0] > 3:
            data = data.moveaxis(2,0)
        if self.transform != None:
            sample_transforms = len(self.transform.transforms)
            if data.shape[2] != self.transform.transforms[sample_transforms-1].size[0]:
                # print ( "Transforming sample ", idx, " with shape ", data.shape)
                # data = data.to('cpu')
                transformed_data = self.transform(data)
                if (transformed_data.shape[0] == 1):
                    transformed_data = transformed_data.expand(3, transformed_data.shape[1], transformed_data.shape[2])
                # data.to(self.device)
                temp = list(self.data[idx])
                temp[0] = transformed_data.detach().cpu()
                self.data[idx] = tuple(temp)
                data = transformed_data
            else:
                t = 1
            # data = data.to(self.device)
        return data, label