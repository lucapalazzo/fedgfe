from collections import defaultdict
import copy
import sys
import sklearn
import torch
import torch.nn as nn
import numpy as np
import time

import wandb
from flcore.clients.clientbase import Client
from sklearn.preprocessing import label_binarize
from sklearn.metrics import balanced_accuracy_score
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight

class clientRewind(Client):
    def __init__(self, args, id, train_samples, test_samples,is_strong = False, id_by_type=-1, rewind_epochs = 0, rewind_interval = 0, rewind_ratio = 0,**kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.node_routes = []
        self.rewind_previous_node = []
        self.rewind_epochs = rewind_epochs
        self.rewind_interval = rewind_interval
        self.rewind_ratio = rewind_ratio
        self.id_by_type = id_by_type
        self.train_loader = None
        self.no_wandb = args.no_wandb
        self.train_dataloader = None
        self.test_dataloader = None

        self.train_model = self.model
        self.train_model_id = id
        self.starting_model = self.model
        self.logits = None
        # self.global_logits = None # quesi non sono più qelli globali ma quelli ricevuti dall'altro client
        # self.loss_mse = nn.MSELoss()

        self.lamda = args.lamda
        self.dataset = args.dataset
        self.loss_weightes = None


    def train(self, client_device = None, rewind_train_node = None):
        node_trainloader = self.load_train_data()
        # if ( self.loss_weighted and self.loss_weightes == None ):
        if ( self.loss_weighted and self.loss_weightes == None ):
            classes = [y[1].item() for x, y in enumerate(self.train_data)]
            unique = np.unique(classes)
            loss_weights = (compute_class_weight(class_weight='balanced', classes=unique, y=classes))
            print ( f"Node {self.id} setting loss weights to {loss_weights}")
            self.loss_weights = torch.Tensor(loss_weights).to(self.device)
            self.loss = nn.CrossEntropyLoss(weight=self.loss_weights)
        # unique, counts = np.unique(self.train_data[1], return_counts=True)


        trainloader = node_trainloader
        start_time = time.time()
        device = self.device
        if (client_device != None):
            device = client_device
        # self.model.to(self.device)
        print ( "Training model from node %d on dataset %d " % ( self.train_model_id, self.id ) )
        self.model = self.train_model
        self.loss_weights.to(device)
        self.model.train().to(device)

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        if ( self.rewind_epochs > 0 and rewind_train_node != None ):
            rewind_steps = self.rewind_epochs
        else:
            rewind_steps = int ( max_local_epochs * self.rewind_ratio )
        max_local_epochs = max_local_epochs - rewind_steps
        
        for step in range(max_local_epochs):
            if ( step == max_local_epochs//2 and rewind_train_node != None ):
                self.rewind_metrics()
                print ( "Halfway step: %d steps on node %d, rewinding to node %d for %d steps" % ( step, self.id, rewind_train_node.id, rewind_steps ) )
                self.rewind_train ( rewind_steps, rewind_train_node, device )
                self.rewind_metrics()
            trainloader = node_trainloader

            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(device)
                else:
                    x = x.to(device)
                y = y.to(device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x).to(device)
                loss = self.loss(output, y).to(device)

                # if self.global_logits != None:
                #     logit_new = copy.deepcopy(output.detach())
                #     for i, yy in enumerate(y):
                #         y_c = yy.item()
                #         if type(self.global_logits[y_c]) != type([]):
                #             logit_new[i, :] = self.global_logits[y_c].data
                #     loss += self.loss_mse(logit_new, output) * self.lamda

                # for i, yy in enumerate(y):
                #     y_c = yy.item()
                #     logits[y_c].append(output[i, :].detach().data)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.rewind_metrics()

    def rewind_train(self, rewind_epochs = 0, rewind_train_node = None, device = 0):
        if ( rewind_epochs == 0 or rewind_train_node == None):
            return
        dataloader = rewind_train_node.load_train_data()
        start_time = time.time() 
        for step in range(rewind_epochs):
            for i, (x, y) in enumerate(dataloader):
                if type(x) == type([]):
                    x[0] = x[0].to(device)
                else:
                    x = x.to(device)
                y = y.to(device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        # self.logits = agg_func(logits)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_logits(self, global_logits):
        self.global_logits = copy.deepcopy(global_logits)

    def train_metrics(self):
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
                loss = self.loss(output, y)

                # if self.global_logits != None:
                #     logit_new = copy.deepcopy(output.detach())
                #     for i, yy in enumerate(y):
                #         y_c = yy.item()
                #         if type(self.global_logits[y_c]) != type([]):
                #             logit_new[i, :] = self.global_logits[y_c].data
                #     loss += self.loss_mse(logit_new, output) * self.lamda
                    
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num
    
    def rewind_metrics(self):
        losses, train_num = self.train_metrics()
        loss = losses / train_num
        if not self.no_wandb:
            wandb.log({f"rewind/rewind_phase_loss_{self.id}": loss})

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
                output = self.model(x)

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
# https://github.com/yuetan031/fedlogit/blob/main/lib/utils.py#L205
def agg_func(logits):
    """
    Returns the average of the weights.
    """

    for [label, logit_list] in logits.items():
        if len(logit_list) > 1:
            logit = 0 * logit_list[0].data
            for i in logit_list:
                logit += i.data
            logits[label] = logit / len(logit_list)
        else:
            logits[label] = logit_list[0]

    return logits