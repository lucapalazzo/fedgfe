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
from datautils.dataset_chectxray import ChestXrayDataset



random.seed(1)
np.random.seed(1)
num_clients = 20
num_classes = 10
dir_path = "Cifar10/"


# Allocate data to users
def generate_chestxray(args, dir_path, num_clients, num_classes, niid, balance, partition, alpha=0.1, class_per_client = 2):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(args, config_path, train_path, test_path, num_clients, num_classes, niid, balance, partition):
        return
        
    # Get Cifar10 data
    transform = None
    image_size = args.dataset_image_size
    transform = transforms.Compose(
        [transforms.ToTensor()])
    if args.dataset_transform:
        transform.append([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    if image_size != -1:
        transform.append(transforms.Compose([transforms.Resize(image_size)]))

    trainset = ChestXrayDataset(root=dir_path+"rawdata", train=True, download=True, transform=transform)
    testset = ChestXrayDataset( dataset= trainset, train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets, trainset.masks = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets, testset.masks = test_data

    dataset_image = []
    dataset_label = []
    dataset_mask = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_mask.extend(trainset.masks.cpu().detach().numpy())
    dataset_mask.extend(testset.masks.cpu().detach().numpy())

    dataset_union = { 'samples': dataset_image, 'labels': dataset_label, 'masks': dataset_mask }

    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    dataset_classes = len(trainset.labels)

    X, y, client_data, statistic = separate_data((dataset_image, dataset_label), num_clients, dataset_classes,  
                                    niid, balance, partition, class_per_client, alpha=alpha, dataset_union=dataset_union)
    train_data, test_data = split_data(X, y, client_data = client_data)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, niid, balance, partition, alpha=alpha)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None
    args = None

    generate_chestxray(args, dir_path, num_clients, num_classes, niid, balance, partition )