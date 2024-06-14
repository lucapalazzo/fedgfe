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

import torch
import math
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import wandb
from utils.data_utils import read_client_data
from datautils.node_dataset import NodeData
from modelutils.modelwrapper import FLModel
from torchvision import transforms
from sklearn.metrics import confusion_matrix


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, train_data = None, test_data = None, val_data = None, **kwargs):
        self.args = args
        self.model = FLModel(args, id)

        # self.model = copy.deepcopy(args.model)
        self.starting_model = self.model.inner_model
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data
        self.dataset_image_size = args.dataset_image_size
        self.transform = None
        self.round_available_count = []
        self.federation_clients = None
        if args.dataset_image_size != -1:
            self.transform = transforms.Compose(
                [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.Resize(self.dataset_image_size)])
       
        self.node_data = NodeData(args, self.id, transform=self.transform, **kwargs)

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.local_learning_rate = args.local_learning_rate
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.dataset_limit = args.dataset_limit
        self.loss_weighted = args.loss_weighted
        self.loss_weights = None

        self.round = -1

        self.federation_size = 0

        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.privacy = args.privacy
        self.dp_sigma = args.dp_sigma

        self.model.loss = nn.CrossEntropyLoss()

            
        if args.model_optimizer == 'Adam':
            self.model.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.local_learning_rate)
        elif args.model_optimizer == 'AdamW':
            self.model.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.local_learning_rate)
        else:
            self.model.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.local_learning_rate)
            
        self.loss = self.model.loss
        self.optimizer = self.model.optimizer
        
        self.learning_rate_schedule = args.learning_rate_schedule
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer, 
            # gamma=args.learning_rate_schedule_gamma,
            patience=2
        )

    def check_batch(self, x, y):
        if len(x) <= 1:
            return False
        return True
    
    def route(self, available_clients = None): 
        if self.routing is not None:
            self.routing.federation_clients = self.federation_clients
            if self.routing.model is None:
                self.routing.model = self.model 
            return self.routing.route( available_clients )
        
    def load_train_data(self, batch_size=None,dataset_limit=0):
        if batch_size == None:
            batch_size = self.batch_size
        return self.node_data.load_train_data(batch_size, dataset_limit)
        if self.train_data == None:
            print("Loading train data for client %d" % self.id)
            self.train_data = read_client_data(self.dataset, self.id, is_train=True,dataset_limit=dataset_limit)
            self.train_samples = len(self.train_data)
        return DataLoader(self.train_data, batch_size, drop_last=True, shuffle=True)

    def load_test_data(self, batch_size=None,dataset_limit=0):
        if batch_size == None:
            batch_size = self.batch_size
        return self.node_data.load_test_data(batch_size, dataset_limit)
        if self.test_data == None:
            print("Loading test data for client %d" % self.id)
            self.test_data = read_client_data(self.dataset, self.id, is_train=False,dataset_limit=dataset_limit)
            self.test_samples = len(self.test_data)
        return DataLoader(self.test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, model):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model, target):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()
            # target_param.grad = param.grad.clone()

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def get_scores_data(self, testloader = None, model = None, on_train = False):
        if testloader == None:
            if on_train == False:
                testloader = self.load_test_data()
            else:
                testloader = self.load_train_data()

        test_acc, test_num, auc, y_true, y_prob = self.test_metrics_data(testloader, model)
        return test_acc, test_num, auc, y_true, y_prob
        

    def get_confusion_matrix(self, testloader = None, model = None):

        test_acc, test_num, auc, y_true, y_pred = self.get_scores_data(testloader, model)
        y_pred = F.softmax(torch.tensor(y_pred), dim=1).numpy()
        y_pred = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(y_true, y_pred)
        return cm
    
    def test_metrics(self, test_client = None, on_train = False, model = None):
        client = self
        if test_client != None:
            client = test_client
        if on_train == True:
            testloader = client.load_train_data()
        else:
            testloader = client.load_test_data()
        
        test_acc, test_num, auc, test_y_true, test_y_prob = self.test_metrics_data(testloader, model ) 

        return test_acc, test_num, auc, test_y_true, test_y_prob
    
    def test_metrics_data(self, dataloader, test_model = None):

        test_acc = 0
        test_num = 0
        y_pred = []
        y_prob = []
        y_true = []
        a = []   
        model = self.model

        if ( test_model != None):
            model = test_model

        model.to(self.device)
        model.eval()
        self.node_data.to(self.device)
        with torch.no_grad():
            for x, y in dataloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)
                # if isinstance(output, dict):
                #     output = output['logits']
                
                predictions = torch.argmax(output, dim=1)

                test_acc += (torch.sum(predictions == y)).item()
                test_num += y.shape[0]

                if torch.isnan(output).any().item():
                    if not self.no_wandb:
                        wandb.log({f'warning/{self.id}': torch.isnan(output)})
                    # print(f'warning for client {self.id} in round {self.round}:', torch.isnan(output))
                    print(f'warning for client {self.id} in round {self.round}:', "output contains nan")

                prob = F.softmax(output, dim=1) 
                # y_prob.append(prob.detach().cpu().numpy()) 
                y_prob.append(output.detach().cpu().numpy()) 
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(y.detach().cpu().numpy())

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        # a = np.concatenate(prob, axis=0)
        if len(y_prob) > 0:
            y_prob = np.concatenate(y_prob, axis=0)
            y_true = np.concatenate(y_true, axis=0)
            prob = prob.detach().cpu().numpy()
        # else:
        #     print ( "No data for client ", self.node_data.id)


        # auc = metrics.roc_auc_score(y_true, y_prob[:,1], average='micro')
        # auc = metrics.roc_auc_score(y_true, prob, average='micro')
        auc = 0
        
        return test_acc, test_num, auc, y_true, y_prob

    def train_metrics(self, trainloader=None):
        if ( trainloader == None):
            trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        self.model.to(self.device)
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                if ( torch.isnan(output).any()):
                    print ( "Output NAN")

                loss = self.model.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]


        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num
    def train_metrics_other(self, test_client = None):
        if ( test_client == None and test_client.id != self.id):
            return
        trainloader = test_client.load_train_data()
        return self.train_metrics(trainloader)

    def test_metrics_other(self, test_client = None):
        if ( test_client == None and test_client.id != self.id):
            return
        
        testloaderfull = test_client.load_test_data()
       
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.starting_model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc, y_true, y_prob

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
