from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import random
from pathlib import Path
import re
from PIL import Image
import torchvision.transforms.functional as F

class ShenzhenDataset(Dataset):
    def __init__(self, dataset = None, is_train=True, test_ratio = 0.2, validate_ratio=0, train=None, transform=None, download=False, root=".", dataset_path = "dataset", datafile = "Data Chest X-Ray RSUA (Validated)/Split_Data_RSUA_Paths_k3.xlsx"):
        self.is_train = is_train
        if train is not None:
            self.is_train = train

        self.dataset_path = dataset_path
        self.datafile = datafile
        self.transform = transform
        self.labels = []
        self.traindata = None
        self.testdata = None
        self.valdata = None
        self.mask_path = "mask"
        self.image_path = "img"

        if dataset is not None and isinstance(dataset, ShenzhenDataset):
            self.samples = dataset.samples
            self.traindata_ids = dataset.traindata_ids
            self.testdata_ids = dataset.testdata_ids
            self.valdata_ids = dataset.valdata_ids
            self.traindata_len = dataset.traindata_len
            self.testdata_len = dataset.testdata_len
            self.validatedata_len = dataset.validatedata_len
            self.sample_ids = dataset.sample_ids
            # self.labels = dataset.labels
            return

        self.data_path = Path(self.dataset_path+"/Shenzhen")
        self.data_dir = Path(self.dataset_path)
        self.samples_data_dir = Path(self.dataset_path+"/img")
        self.masks_data_dir = Path(self.dataset_path+"/mask")

        self.samples = []
        for sample_file in self.samples_data_dir.iterdir():
            if sample_file.suffix == '.png':
                mask_file = self.masks_data_dir / (sample_file.stem + '.png')
                if mask_file.exists():
                    self.samples.append((sample_file, mask_file))
                else:
                    print('No mask for sample %s' %sample_file)

        self.traindata_len = int(len(self.samples) * (1 - test_ratio - validate_ratio))
        self.testdata_len = int(len(self.samples) * test_ratio) + 1
        self.validatedata_len = int(len(self.samples) * validate_ratio)
        self.sample_ids = list(range(len(self.samples)))
        sample_ids = self.sample_ids
        self.traindata_ids = random.sample(sample_ids, self.traindata_len)
        sample_ids = [i for i in sample_ids if i not in self.traindata_ids]

        self.testdata_ids = random.sample(sample_ids, self.testdata_len)
        sample_ids = [i for i in sample_ids if i not in self.testdata_ids]
        self.valdata_ids = random.sample(sample_ids, self.validatedata_len)
        sample_ids = [i for i in sample_ids if i not in self.valdata_ids]
    
    def get_sample(self, sample_type, index):
        if sample_type == 'train':
            sample_file, mask_file = self.samples[self.traindata_ids[index]]
        elif sample_type == 'test':
            sample_file, mask_file = self.samples[self.testdata_ids[index]]

        print ( f"{index}", end=" " )
        # print ( "%s Sample %s mask %s" %(sample_type, str(sample_file), str(mask_file)))

        sample_tensor = read_image_as_tensor(sample_file)
        mask_tensor = read_image_as_tensor(mask_file)

        # print ( "Image %s mask %s" %(str(sample_tensor.shape), str(mask_tensor.shape)))
        downsampled_image = self.transform(sample_tensor)
        downsampled_mask = self.transform(mask_tensor)

        # print ( "Image %s mask %s" %(str(downsampled_image.shape), str(downsampled_mask.shape)))

        sample = [downsampled_image, downsampled_mask]
        return sample
    
    def get_data(self, datatype):
        if datatype == 'train':
            self.traindata = []
            data = self.traindata
            ids = self.traindata_ids
            
        elif datatype == 'test':
            self.testdata = []
            data = self.testdata
            ids = self.testdata_ids
        elif datatype == 'validate':
            self.valdata = []
            data = self.valdata
            ids = self.valdata_ids
        for index in ids:
            sample = self.get_sample(index)
            data.append(sample) 
        return data

    # def get_label(self, path):
    #     splitted = path.split('/')
    #     label = splitted[-3]
    #     label_id = self.labels.index(label)
    #     return label_id
    
    # def populate_labels(self):
    #     for index,col in self.data.iterrows():
    #         splitted = col['images_path'].split('/')
    #         label = splitted[-3]
    #         if label not in self.labels:
    #             self.labels.append(label)
    #             print("Added %d col %s" %(index, label))
    #         # mask = col['masks_path']
    #         # path = Path(self.dataset_path+'/'+col['images_path'])
    #         # mask_path = col['masks_path']
    #         # image_path = col['images_path']
    #         # image_path = re.sub(r'Data Thorax DICOM RSUA \(Validated\)', 'Data Chest X-Ray RSUA (Validated)', image_path)
    #         # mask_path = re.sub(r'Data Thorax DICOM RSUA \(Validated\)', 'Data Chest X-Ray RSUA (Validated)', mask_path)
    #         # # image_path = re.sub(r'.npy', '.bmp', image_path)
    #         # image_path = Path(self.dataset_path+'/'+image_path)
    #         # mask_path = Path(self.dataset_path+'/'+mask_path)
    #         # # print("index %d col %s %s" %(index, mask, path))
    #         # if not path.exists():
    #         #     print("Path %s does not exist" %path)
    #         # else:
    #         #     image_length = os.path.getsize(path)
    #         #     print ( "File %s size %d" %(path, image_length))

    #     return self.labels


    def __len__(self):
        if self.is_train:
            return self.traindata_len
        else:
            return self.testdata_len
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_train:
            item = self.get_sample('train',idx)

        elif self.is_train == False:
            item = self.get_sample('test', idx)

        return item

        splitted = col['images_path'].split('/')
        label = splitted[-3]
        if label not in labels:
            labels.append(label)
            label_id = len(labels)-1
            print("Added %d col %s" %(label_id, label))
        else:
            label_id = labels.index(label)
        mask_path = col['masks_path']
        image_path = col['images_path']
        image_path = re.sub(r'Data Thorax DICOM RSUA \(Validated\)', 'Data Chest X-Ray RSUA (Validated)', image_path)
        mask_path = re.sub(r'Data Thorax DICOM RSUA \(Validated\)', 'Data Chest X-Ray RSUA (Validated)', mask_path)
        return self.dataset[idx]

    def __delitem__(self, key):
        pass

    def __setitem__(self, key, value):
        pass

    def __contains__(self, item):
        pass

    def __iter__(self):
        pass

    def __reversed__(self):
        pass

def read_image_as_tensor(image_path):
    image = Image.open(image_path)
    image_tensor = F.to_tensor(image)
    return image_tensor