from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import random
from pathlib import Path
import re

class ChestXrayDataset(Dataset):
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

        if dataset is not None and isinstance(dataset, ChestXrayDataset):
            self.data = dataset.data
            self.traindata_ids = dataset.traindata_ids
            self.testdata_ids = dataset.testdata_ids
            self.valdata_ids = dataset.valdata_ids
            self.traindata_len = dataset.traindata_len
            self.testdata_len = dataset.testdata_len
            self.validatedata_len = dataset.validatedata_len
            self.sample_ids = dataset.sample_ids
            self.labels = dataset.labels
            return

        datafile_path = dataset_path + "/" + datafile
        if not os.path.exists(datafile_path):
            print ("File %s does not exist" %(dataset_path+"/"+datafile))
            return
        self.data = pd.read_excel(datafile_path, index_col=0)
        self.traindata_len = int(len(self.data) * (1 - test_ratio - validate_ratio))
        self.testdata_len = int(len(self.data) * test_ratio) + 1
        self.validatedata_len = int(len(self.data) * validate_ratio)
        self.sample_ids = list(self.data.index)
        sample_ids = self.sample_ids
        self.traindata_ids = random.sample(sample_ids, self.traindata_len)
        sample_ids = [i for i in sample_ids if i not in self.traindata_ids]

        self.testdata_ids = random.sample(sample_ids, self.testdata_len)
        sample_ids = [i for i in sample_ids if i not in self.testdata_ids]
        self.valdata_ids = random.sample(sample_ids, self.validatedata_len)
        sample_ids = [i for i in sample_ids if i not in self.valdata_ids]

        self.populate_labels()
    
    def get_sample(self, index):
        col = self.data.iloc[index]
        mask_path = col['masks_path']
        image_path = col['images_path']
        image_path = re.sub(r'Data Thorax DICOM RSUA \(Validated\)', 'Data Chest X-Ray RSUA (Validated)', image_path)
        mask_path = re.sub(r'Data Thorax DICOM RSUA \(Validated\)', 'Data Chest X-Ray RSUA (Validated)', mask_path)
        image_sample = np.load(self.dataset_path+'/'+image_path)
        mask_sample = np.load(self.dataset_path+'/'+mask_path)
        image_label = self.get_label(image_path)
        # sample = {'image': image_sample, 'label': image_label, 'mask': mask_sample}
        sample = [image_sample, image_label, mask_sample]
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

    def get_label(self, path):
        splitted = path.split('/')
        label = splitted[-3]
        label_id = self.labels.index(label)
        return label_id
    
    def populate_labels(self):
        for index,col in self.data.iterrows():
            splitted = col['images_path'].split('/')
            label = splitted[-3]
            if label not in self.labels:
                self.labels.append(label)
                print("Added %d col %s" %(index, label))
            # mask = col['masks_path']
            # path = Path(self.dataset_path+'/'+col['images_path'])
            # mask_path = col['masks_path']
            # image_path = col['images_path']
            # image_path = re.sub(r'Data Thorax DICOM RSUA \(Validated\)', 'Data Chest X-Ray RSUA (Validated)', image_path)
            # mask_path = re.sub(r'Data Thorax DICOM RSUA \(Validated\)', 'Data Chest X-Ray RSUA (Validated)', mask_path)
            # # image_path = re.sub(r'.npy', '.bmp', image_path)
            # image_path = Path(self.dataset_path+'/'+image_path)
            # mask_path = Path(self.dataset_path+'/'+mask_path)
            # # print("index %d col %s %s" %(index, mask, path))
            # if not path.exists():
            #     print("Path %s does not exist" %path)
            # else:
            #     image_length = os.path.getsize(path)
            #     print ( "File %s size %d" %(path, image_length))

        return self.labels


    def __len__(self):
        if self.is_train:
            return self.traindata_len
        else:
            return self.testdata_len
        return len(self.data)

    def __getitem__(self, idx):
        if self.is_train:
            if self.traindata is None:
                self.get_data('train')

            item = self.traindata[idx]
        elif self.is_train == False:
            if self.testdata is None:
                self.get_data('test')
            item = self.testdata[idx]

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