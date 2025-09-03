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
from datautils.dataset_jsrt import JSRTDataset



random.seed(1)
np.random.seed(1)
num_clients = 20
num_classes = 10
dir_path = "JSRT/"


# Allocate data to users
def generate_jsrt(args, dir_path, num_clients, num_classes, niid, balance, partition, alpha=0.1, class_per_client = 2, jsrt_path = "dataset/JSRT/"):
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
    # transform = transforms.Compose([ transforms.ToPILImage()])
    transform = transforms.Compose([])
    if args.dataset_transform:
        transform.transforms.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    if image_size != -1:
        transform.transforms.append(transforms.Resize(image_size))
    # transform.transforms.append(transforms.ToTensor())

    trainset = JSRTDataset(root=dir_path+"rawdata", train=True,  transform=transform, dataset_path=jsrt_path)
    testset = JSRTDataset( dataset=trainset, train=False, transform=transform,dataset_path=jsrt_path)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        traindata, traintargets, trainmasks, traininfo = train_data[0][0], train_data[0][1], train_data[1]['semantic_masks'], train_data[1]['sample_info']
    for _, test_data in enumerate(testloader, 0):
        testdata, testtargets, testmasks, testinfo = test_data[0][0], test_data[0][1], test_data[1]['semantic_masks'], test_data[1]['sample_info']

    dataset_train_data = trainset.traindata
    dataset_train_data_ids = trainset.traindata_ids
    dataset_test_data = testset.testdata
    dataset_test_data_ids = testset.testdata_ids
    dataset_data = trainset.data

    dataset_image = []
    dataset_label = []
    dataset_mask = []
    dataset_info = []

    dataset_image.extend(traindata.cpu().detach().numpy())
    dataset_image.extend(testdata.cpu().detach().numpy())
    dataset_mask.extend(trainmasks.cpu().detach().numpy())
    dataset_mask.extend(testmasks.cpu().detach().numpy())
    dataset_info.extend(traininfo.cpu().detach().numpy())
    dataset_info.extend(testinfo.cpu().detach().numpy())
    # dataset_label.extend(trainset.targets.cpu().detach().numpy())
    # train_nodules = torch.stack(traintargets[3]).permute(1,0)
    # test_nodules = torch.stack(testtargets[3]).permute(1,0)
    traintargets = torch.stack(traintargets).permute(1,0)
    testtargets = torch.stack(testtargets).permute(1,0)
    
    # dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_label = torch.cat( (traintargets, testtargets),0)
    # dataset_label.append(testtargets)

    chosen_label = 0
    num_classes = len(np.unique(dataset_label[chosen_label]))

    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)
    dataset_mask = np.array(dataset_mask)
    dataset_info = np.array(dataset_info)

    dataset_classes = len(traintargets)
    dataset_classes = 2

    dataset_union = { 'samples': dataset_image, 'labels': dataset_label, 'semantic_masks': dataset_mask, 'info': dataset_info }


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

    generate_jsrt(args, dir_path, num_clients, num_classes, niid, balance, partition )