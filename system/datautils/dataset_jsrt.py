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
from torchvision import transforms
from pathlib import Path
from PIL import Image
import copy

# Labels
# Type ['malignant', 'benign']
# Nodule {'has nodule': 154, 'non-nodule': 93}
# Zones Image zones 
# 0 'l.lower lobe(S8)',
# 1 'r.upper lobe(S1)', 
# 2 'l.upper lobe(S3)',
# 3 'l.upper lobe',
# 4 'r.middl e lobe',
# 5 'r.upper lobe(S3)',
# 6 'r.lower lobe(S7)',
# 7 'l.lower lobe(S9)',
# 8 'r.lower lobe(S9)',
# 9 'r.lower lobe(S6)',
# 10 'r.upper lobe',
# 11 'r.lower lobe',
# 12 'r.upper lobe(S2)',
# 13 'r.middl e lobe(S4)',
# 14 'l.upper lobe(S1+2)',
# 15 'r.lower lobe(S8)',
# 16 'left lu ng',
# 17 'l.upper lobe(S4)',
# 18 'right l ung',
# 19 'l.lower lobe(S10)',
# 20 'r.lower lobe(S9-10)',
# 21 'l.lower lobe',
# 22 'r.upper lobe(S2-S3)',
# 23 'l.lower lobe(S6)',
# 24 'l.lower lobe(S6-S8)',
# 25 'r.lower lobe(S10)',
# 26 'l.upper lobe(S5)',
# 27 'r.middl e lobe(S5)',
# 28 'unknown']

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
       
        self.semantic_segmentation_labels_path = dataset_path + "/SemanticSegmentation/label"
        self.semantic_segmentation_image_path = dataset_path + "/SemanticSegmentation/org"
        self.semantic_segmentation_images = []
        self.semantic_segmentation_labels = []

        if Path(self.semantic_segmentation_labels_path).exists() == True:
            self.semantic_segmentation_labels = os.listdir(self.semantic_segmentation_labels_path)


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

        self.jsrt = Jsrt()
        self.jsrt.load_images(images_path=dataset_path)


        
        if not os.path.exists(datafile_path):
            print ("File %s does not exist" %(dataset_path+"/"+datafile))
            return
        
        self.data = self.jsrt._has_nodule_image_list + self.jsrt._non_nodule_image_list
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

    def get_semantic_segmentation_image(self, index):
        image = self.data[index]
        image_path = Path(image.image_path)
        image_index = re.search(r'\d+', image_path.name).group()
        segmentation_mask_file = self.semantic_segmentation_labels_path + "/case" + image_index + "_label.png"
        segmentation_image_file = self.semantic_segmentation_image_path + "/case" + image_index + ".bmp"
        image = Image.open(segmentation_image_file)
        image = torch.from_numpy(np.array(image))
        # image = self.transform(image)
        
        # Cuore 85
        # Polmone 255
        # Esterno campo polmonare 170
        # Esterno corpo 0
        colors = [255, 170, 85, 0]

        mask = Image.open(segmentation_mask_file)
        mask = np.array(mask)
        orginal_mask_tensor = mask
        mask = torch.from_numpy(mask).unsqueeze(0).repeat(4,1,1)
        
        for i in range(mask.shape[0]):
            if colors[i] == 0:
                mask[i] = torch.logical_not(mask[i])
            else:
                mask[i] *= (mask[i] == colors[i])
                mask[i][mask[i] == colors[i]] = 1
        mask = self.transform(mask)
        return image, mask    


    def get_sample(self, index):
        image = self.data[index]

        semantic_image, semantic_mask = self.get_semantic_segmentation_image(index)

        sample_tensor = torch.from_numpy(image.image.astype(np.float32))
        sample_tensor = sample_tensor/4095
        sample_tensor = sample_tensor.unsqueeze(0).expand(3, -1, -1)


        if ( self.transform is not None):
            target_size = 0
            original_size = sample_tensor.shape[1]
            image_sample = self.transform(sample_tensor)
            for t in range(len(self.transform.transforms)):
                if isinstance(self.transform.transforms[t], transforms.Resize):
                    target_size = self.transform.transforms[t].size
                    break
            if target_size != 0:
                nodule_size = 0 if image.nodule_size == None else image.nodule_size*target_size/original_size
                nodule_position_x = int(image.x*target_size/original_size)
                nodule_position_y = int(image.y*target_size/original_size)
                # self.labels[index][3] = nodule_size
                # self.labels[index][4] = nodule_position_x
                # self.labels[index][5] = nodule_position_y

        image_label = self.labels[index]
        # tensor_image_label = []
        # tensor_image_label.append(torch.tensor(image_label[0]))
        # tensor_image_label.append(torch.tensor(image_label[1]))
        # tensor_image_label.append(torch.tensor(image_label[2]))
        # tensor_image_label.append(torch.tensor(image_label[3]))
        # sample = {'image': image_sample, 'label': image_label, 'mask': mask_sample}
        sample = [image_sample, image_label, semantic_mask]
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

        print ( "Image types %s" % images_types_keys )
        print ( "Image zones %s" % images_zones_keys )
        print ( "Image nodules %s" % images_nodules_keys )

        for image in self.data:
            image_type = image._malignant_or_benign
            image_zone = image._position
            image_nodule = image._image_type
            nodule_size = 0 if image.nodule_size == None else image.nodule_size
            nodule_position_x = image.x
            nodule_position_y = image.y
            # label = [images_types_keys.index(image_type), images_nodules_keys.index(image_nodule), images_zones_keys.index(image_zone), nodule_size, nodule_position_x, nodule_position_y]
            label = [images_types_keys.index(image_type), images_nodules_keys.index(image_nodule), images_zones_keys.index(image_zone) ]
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
            ids = self.traindata_ids

            item = self.traindata[idx]
        elif self.is_train == False:
            if self.testdata is None:
                self.get_data('test')
            item = self.testdata[idx]
            ids = self.testdata_ids

        image_info = copy.deepcopy(self.data[idx])

        image_info_dict = image_info.__dict__
        del image_info_dict['resized_image']
        del image_info_dict['image']

        for k in list(image_info_dict):
            if image_info_dict[k] == None:
                print ( "Found none" )
                # del image_info_dict[k]
                image_info_dict[k] = -1
        
        # sample_info = {'id': ids[idx] }
        sample_info = ids[idx]

        return item, {'image': item[0], 'label': item[1], 'semantic_masks': item[2], 'sample_info': sample_info }
        # return item, {'image': item[0], 'label': item[1], 'semantic_masks': item[2] }

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