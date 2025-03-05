# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from datautils.dataset_utils import check, separate_data, split_data, save_file
from datautils.dataset_darwin import DarwinDataset



random.seed(1)
np.random.seed(1)
num_clients = 20
num_classes = 10
dir_path = "JSRT/"


# Allocate data to users
def generate_darwin(args, dir_path, num_clients, num_classes, niid, balance, partition, alpha=0.1, class_per_client = 2, darwin_path = "dataset/ChestXRaySegmentation/Darwin", sourcedir = ""):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(args, config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return
        
    transform = None
    image_size = args.dataset_image_size
    transform = transforms.Compose([ transforms.ToPILImage()])
    if args.dataset_transform:
        transform.transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    if image_size != -1:
        transform.transforms.append(transforms.Resize(image_size))
    transform.transforms.append(transforms.ToTensor())

    trainset = DarwinDataset(root=dir_path+"rawdata", train=True,  transform=transform, dataset_path=darwin_path)
    testset = DarwinDataset( dataset=trainset, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        # trainset, batch_size=len(trainset), shuffle=False)
        trainset, batch_size=64, shuffle=False)
    testloader = torch.utils.data.DataLoader(
        # testset, batch_size=len(testset), shuffle=False)
        testset, batch_size=64, shuffle=False)
    
    train_samples = []
    test_samples = []
    train_masks = []
    test_masks = []
    for _, train_data in enumerate(trainloader, 0):
        traindata, trainmask = train_data[0], train_data[1]
        train_samples.append(traindata.cpu().detach().numpy())
        train_masks.append(trainmask.cpu().detach().numpy())
    for _, test_data in enumerate(testloader, 0):
        test_sample, test_mask = test_data[0], test_data[1]
        test_samples.append(test_sample.cpu().detach().numpy())
        test_masks.append(test_mask.cpu().detach().numpy())

    dataset_image = []
    # dataset_label = []
    dataset_mask = []

    for train_sample in train_samples:
        if len(dataset_image) == 0:
            dataset_image = train_sample
        else:
            dataset_image = np.concatenate((dataset_image, train_sample))
    for test_sample in test_samples:
        if len(dataset_image) == 0:
            dataset_image = test_sample
        else:
            dataset_image = np.concatenate((dataset_image, test_sample))
    for train_mask in train_masks:
        if len(dataset_mask) == 0:
            dataset_mask = train_mask
        else:
            dataset_mask = np.concatenate((dataset_mask, train_mask))
    for test_mask in test_masks:
        if len(dataset_mask) == 0:
            dataset_mask = test_mask
        else:
            dataset_mask = np.concatenate((dataset_mask, test_mask))

    dataset_union = { 'samples': dataset_image, 'masks': dataset_mask }

    # dataset_image = np.array(dataset_image)
    # dataset_mask = np.array(dataset_mask)
    data = (dataset_image, dataset_mask)


    dataset_classes = 0


    X, y, client_data, statistic = separate_data(data, num_clients, dataset_classes, dataset_classes,  
                                    niid, balance, partition, class_per_client, dataset_union=dataset_union)
    train_data, test_data = split_data(X, y, client_data = client_data)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition, alpha=alpha)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    args = None

    generate_darwin(args, dir_path, num_clients, num_classes, niid, balance, partition )