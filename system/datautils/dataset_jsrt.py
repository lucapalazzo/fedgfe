from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import os
import random
from pathlib import Path
from datautils.jsrtparser.jsrt import Jsrt, JsrtImage
import re
from collections import OrderedDict

class JSRTDataset(Dataset):
    def __init__(self, dataset = None, is_train=True, test_ratio = 0.2, validate_ratio=0, train=None, transform=None, download=False, root=".", dataset_path = "dataset", datafile = "All247images/"):
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

        if dataset is not None and isinstance(dataset, JSRTDataset):
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

        jsrt = Jsrt()
        jsrt.load_images(images_path=dataset_path)

        
        if not os.path.exists(datafile_path):
            print ("File %s does not exist" %(dataset_path+"/"+datafile))
            return
        
        self.data = jsrt._has_nodule_image_list + jsrt._non_nodule_image_list
        self.fix_data(self.data)
        # self.data.append(jsrt._non_nodule_image_list)

        # self.data = pd.read_excel(datafile_path, index_col=0)
        self.traindata_len = int(len(self.data) * (1 - test_ratio - validate_ratio))
        self.testdata_len = int(len(self.data) * test_ratio) + 1
        self.validatedata_len = int(len(self.data) * validate_ratio)
        self.sample_ids =  range(len(self.data))
        sample_ids = self.sample_ids
        self.traindata_ids = random.sample(sample_ids, self.traindata_len)
        sample_ids = [i for i in sample_ids if i not in self.traindata_ids]

        self.testdata_ids = random.sample(sample_ids, self.testdata_len)
        sample_ids = [i for i in sample_ids if i not in self.testdata_ids]
        self.valdata_ids = random.sample(sample_ids, self.validatedata_len)
        sample_ids = [i for i in sample_ids if i not in self.valdata_ids]

        self.populate_labels()
    
    def get_sample(self, index):
        image = self.data[index]

        sample_tensor = torch.from_numpy(image.image.astype(np.float32))
        sample_tensor = sample_tensor/4095
        sample_tensor = sample_tensor.unsqueeze(0).expand(3, -1, -1)

        image_sample = self.transform(sample_tensor)
        image_label = self.labels[index]
        # sample = {'image': image_sample, 'label': image_label, 'mask': mask_sample}
        sample = [image_sample, image_label]
        print ( f"Id {index} Benign: {image._malignant_or_benign} Nodule: {image._image_type} Zone: {image._position} {image_label}")
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

    def get_images_labels(self, images):
        image_types = OrderedDict()
        image_zones = OrderedDict()
        image_nodules = OrderedDict()
        for image in images:
            image_type = image._malignant_or_benign
            image_zone = image._position
            image_nodule = image._image_type
            if image_type not in image_types:
                image_types[image_type]=1
            else:
                image_types[image_type] += 1
            if image_zone not in image_zones:
                image_zones[image_zone]=1
            else:
                image_zones[image_zone] += 1
            if image_nodule not in image_nodules:
                image_nodules[image_nodule]=1
            else:
                image_nodules[image_nodule] += 1

        print (f"Image Types: {image_types}")
        print (f"Image Zones: {image_zones}")
        print (f"Image Nodules: {image_nodules}")
        return image_types, image_zones, image_nodules  


    def fix_data(self, data):
        for image in data:
            image._malignant_or_benign = 'benign' if image._malignant_or_benign == None else image._malignant_or_benign
            image._image_type = 'non-nodule' if image._image_type == None else image._image_type
            image._position = 'unknown' if image._position == None else image._position

    def populate_labels(self):
        image_types, images_zones, image_nodules = self.get_images_labels(self.data)
        images_types_keys = list(image_types.keys())
        images_zones_keys = list(images_zones.keys())
        images_nodules_keys = list(image_nodules.keys())

        for image in self.data:
            image_type = image._malignant_or_benign
            image_zone = image._position
            image_nodule = image._image_type
            label = [images_types_keys.index(image_type), images_nodules_keys.index(image_nodule), images_zones_keys.index(image_zone)]
            self.labels.append(label)
            print("Label %s" %(label))
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

        return item, {'image': item[0], 'label': item[1]}

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