from collections import defaultdict
import copy
import sys
import sklearn
import torch
import torch.nn as nn
import numpy as np
import time
import math

import wandb
from flcore.clients.clientbase import Client
from sklearn.preprocessing import label_binarize
from sklearn.metrics import balanced_accuracy_score
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torchvision import  transforms

class clientRewind(Client):
    def __init__(self, args, id, train_samples, test_samples,is_strong = False, id_by_type=-1, rewind_epochs = 0, rewind_interval = 0, rewind_ratio = 0,**kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.node_routes = []
        self.rewind_previous_node_id = []
        self.rewind_previous_model_id = []
        self.rewind_previous_node = []
        self.rewind_previous_node_loss = []
        self.rewind_epochs = rewind_epochs
        self.rewind_interval = rewind_interval
        self.rewind_ratio = args.rewind_ratio
        self.rewind_donkey = args.rewind_donkey
        self.rewind_donkey_count = args.rewind_donkey_count
        self.rewind_learning_rate_schedule = args.rewind_learning_rate_schedule
        self.rewind_strategy = args.rewind_strategy
        self.rewind_learning_rate_decay = args.rewind_learning_rate_decay
        self.rewind_learning_rate_decay_ratio = args.rewind_learning_rate_decay_ratio
        self.rewind_learning_rate_keep = args.rewind_learning_rate_keep
        self.rewind_end_epoch_ratio = args.rewind_end_epoch_ratio
        self.id_by_type = id_by_type
        self.train_loader = None
        self.no_wandb = args.no_wandb
        self.train_dataloader = None
        self.test_dataloader = None



        self.starting_model = self.model
        self.starting_loss = self.loss
        self.train_model = self.model
        self.next_train_model = self.model
        self.train_model_id = id
        self.next_train_model_id = id

        self.logits = None
        # self.global_logits = None # quesi non sono più qelli globali ma quelli ricevuti dall'altro client
        # self.loss_mse = nn.MSELoss()

        self.lamda = args.lamda
        self.dataset = args.dataset
        self.node_data_losses = []

        self.rewind_step = 0
        self.focal_loss = sigmoid_focal_loss
        self.test_std = []
        self.test_std_on_train = []

        self.transform = transforms.Compose(
            [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
             transforms.Resize(224)])

    def train(self, client_device = None, rewind_train_node = None, ):
        node_trainloader = self.load_train_data()
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # if self.next_train_model_id != self.id:
        #     self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

        # if ( self.loss_weighted and self.loss_weightes == None ):
        if self.round == 0:
            self.node_data.stats_dump()
        if ( self.loss_weighted and self.loss_weights == None ):

            loss_weights = [0] * self.num_classes
            # lbls = set([l.item() for t,l in self.train_data])
            unique =  [label for label in self.node_data.labels_get()]
            classes = [y[1].item() for x, y in enumerate(self.node_data.train_data)]
            
            # unique = np.unique(classes)
            class_count = len(unique)
            
            lw = (compute_class_weight(class_weight='balanced', classes=unique, y=classes))
            for i in range(class_count):
                class_index = unique[i]
                loss_weights[class_index] = lw[i]
            ## I pesi per la loss sono legati al dataset e quindi al nodo e non al modello
            self.loss_weights = torch.Tensor(loss_weights).to(self.device)

        # Cambia la loss del modello se se il nodo ha impostato i pesi per le classi
        if self.loss_weights != None:        
            # self.model.loss = nn.CrossEntropyLoss(weight=self.loss_weights)
            self.model.loss.weight = self.loss_weights
        # unique, counts = np.unique(self.train_data[1], return_counts=True)

        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        # for g in self.optimizer.param_groups:
        #     g['lr'] = 0.1
        # self.learning_rate_scheduler.base_lrs = [0.1]
        trainloader = node_trainloader
        start_time = time.time()
        device = self.device
        if (client_device != None):
            device = client_device
        # self.model.to(self.device)
        
        if self.loss_weights != None and self.round == 0:
            print ( f"Node {self.id} setting loss weights to {self.loss_weights}")
        # self.node_data.stats_dump()
        if len(self.rewind_previous_node) > 0:
            print ( "Rewind previous:  node %s dataset %s losses %s " % (self.rewind_previous_node_id, self.rewind_previous_model_id, self.rewind_previous_node_loss))
            # print ( "Model previous trained on ", )
            # # self.rewind_previous_node[-1].node_data.stats_dump()
            # print ( "Previous losses ", )

        self.rewind_previous_model_id.append(self.next_train_model_id)
        # self.train_model = copy.deepcopy(self.next_train_model)
        self.train_model = self.next_train_model
        self.train_model_id = self.next_train_model_id
        self.model = self.train_model
        # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.model.train().to(device)

        print ( "\n--------\nNode %d: training model %d (%d) on dataset %d " % ( self.id, self.train_model.id, self.model.id, self.node_data.id ) )
        print ( "Training on model %s loss %s optimizer %s" % ( hex(id(self.model.inner_model)), hex(id (self.model.loss)), hex(id(self.model.optimizer)) ) )

        if self.train_model_id == self.id:
            print ( f"Node {self.id} training model {self.train_model_id} on self dataset")

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        if ( self.rewind_epochs > 0 and rewind_train_node != None ):
            rewind_epochs = self.rewind_epochs
        else:
            rewind_epochs = int ( max_local_epochs * self.rewind_ratio )
        local_epochs = max_local_epochs - rewind_epochs
        
        rewind_nodes_count = len(self.rewind_previous_node)
        epoch_start_lr = []
        epoch_end_lr = []
        starting_lr = self.local_learning_rate
        self.model.optimizer.param_groups[0]['lr'] = self.local_learning_rate
        print ( "Epoch starting LR ", starting_lr)
        for step in range(local_epochs):
            if ( ( self.rewind_strategy == "halfway" or self.rewind_strategy == "interval" or self.rewind_strategy == "atend_pre"  ) and len(self.rewind_previous_node) > 0 ):
                self.rewind(step, max_local_epochs, rewind_epochs, rewind_nodes_count)

            trainloader = node_trainloader

            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(device)
                    # x[0] = self.transform(x[0])
                else:
                    x = x.to(device)
                    # x= self.transform(x)
                y = y.to(device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x).to(device)
                loss = self.model.loss(output, y).to(device)
                # floss = self.focal_loss(output, y, alpha=0.25, gamma=2, reduction='mean')
                # print ( f"Step {step} Loss {loss} Focal Loss {floss}")
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

                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()

            if ( self.rewind_learning_rate_schedule == True ):
                epoch_start_lr.append( self.learning_rate_scheduler.get_last_lr() )
                self.learning_rate_scheduler.step()
                epoch_end_lr.append( self.learning_rate_scheduler.get_last_lr() )
        
            loss, num = self.train_metrics()
            print ( f"{loss/num} ", end='')
        if ( self.rewind_strategy == "atend" and len(self.rewind_previous_node) > 0 ):
            self.rewind(step, max_local_epochs, rewind_epochs, rewind_nodes_count)
        if self.rewind_learning_rate_decay == True:
            print ( "\nRestoring LR to ", starting_lr)
            self.model.optimizer.param_groups[0]['lr'] = starting_lr
        # print("lr ", *epoch_start_lr, sep=" " )
        # print("lr ", *epoch_end_lr, sep=" " )
        # if self.learning_rate_scheduler != None:
        #     self.learning_rate_scheduler.step()
        # self.rewind_metrics()
        print()
        if len(self.rewind_previous_node) > 0:
            rewind_node = self.rewind_previous_node[-1]
            local_loss, rw_loss = self.rewind_train_metrics(rewind_node)
            if not self.no_wandb:
                wandb.log({f"train/model_{self.model.id}/atend_loss_on_local": local_loss, "round": self.round})
                wandb.log({f"train/model_{self.model.id}/atend_loss_on_previous": rw_loss, "round": self.round})


    def prepare_rewind(self, max_local_epochs, rewind_train_node = None):
        if ( self.rewind_epochs > 0 and rewind_train_node != None ):
            rewind_epochs = self.rewind_epochs
        else:
            rewind_epochs = int ( max_local_epochs * self.rewind_ratio )
        local_epochs = max_local_epochs - rewind_epochs
        
        rewind_nodes_count = len(self.rewind_previous_node)
        return rewind_epochs, local_epochs, rewind_nodes_count
    
    def rewind(self, step, max_local_epochs = 0, rewind_epochs = 0, rewind_node_count = 0, device = 0):

        if self.rewind_donkey:
            rewind_nodes = unique_node ( self.rewind_previous_node[-self.rewind_donkey_count::] )
        else:
            rewind_nodes = [self.rewind_previous_node[-1]]
        rewind_node_count = len(rewind_nodes)

        if rewind_epochs == 0:
            return
        
        rewind_start_epoch = -1
        if ( self.rewind_strategy == "atend_pre" ):
            rewind_ending_epochs_count = math.ceil(rewind_epochs * self.rewind_end_epoch_ratio)
            rewind_start_epoch = ( max_local_epochs - rewind_epochs - rewind_ending_epochs_count)
           
        elif ( self.rewind_strategy == "halfway" ):
            rewind_start_epoch = max_local_epochs//2
        elif ( self.rewind_strategy == "interval" ):
            rewind_start_epoch = max_local_epochs / rewind_epochs // 2
        
        if ( step == rewind_start_epoch or self.rewind_strategy == "atend" ) and rewind_node_count > 0:
                for teacher in rewind_nodes:
                    if ( teacher != None ):
                        local_loss, rw_loss = self.rewind_train_metrics(teacher)
                        if not self.no_wandb:
                            wandb.log({f"train/model_{self.model.id}/pre_rewind_loss_on_local": local_loss, "round": self.round})
                            wandb.log({f"train/model_{self.model.id}/pre_rewind_loss_on_previous": rw_loss, "round": self.round})
                        self.rewind_train ( rewind_epochs, teacher, device )
                        
                        local_loss, rw_loss = self.rewind_train_metrics(teacher)
                        if not self.no_wandb:
                            wandb.log({f"train/model_{self.model.id}/post_rewind_loss_on_local": local_loss, "round": self.round})
                            wandb.log({f"train/model_{self.model.id}/post_rewind_loss_on_previous": rw_loss, "round": self.round})

    def rewind_train(self, rewind_epochs = 0, rewind_train_node = None, device = 0):

        # rewind_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        if ( rewind_epochs == 0 or rewind_train_node == None):
            return
        
        print ( "\nStep on node %d, %s rewinding to node %d for %d epochs" % (self.id, self.rewind_strategy, rewind_train_node.id, rewind_epochs ) )
        
        dataloader = rewind_train_node.load_train_data()
        start_time = time.time()
        device = self.model.device
        # self.model.to(device)
        starting_lr = self.model.optimizer.param_groups[0]['lr']
        if ( self.rewind_learning_rate_decay ):
            rewind_lr = starting_lr * self.rewind_learning_rate_decay_ratio
            self.model.optimizer.param_groups[0]['lr'] = rewind_lr
            print ( f"Original LR: {starting_lr} new LR: {rewind_lr}")
            print ( f"Rewind loss: ", end='')
        for step in range(rewind_epochs):
            for i, (x, y) in enumerate(dataloader):
                if type(x) == type([]):
                    x[0] = x[0].to(device)
                    self.transform(x[0])
                else:
                    x = x.to(device)
                    self.transform(x)
                y = y.to(device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.model.loss(output, y)

                self.model.optimizer.zero_grad()
                
                loss.backward()
                # end_lr = self.optimizer.param_groups[0]['lr']
                self.model.optimizer.step()
            print ( f" {loss} ", end='')
        if not self.rewind_learning_rate_keep:
            print ( "\nRestoring LR to ", starting_lr)
            self.model.optimizer.param_groups[0]['lr'] = starting_lr
            


        # self.rewind_step += 1
        # self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def set_logits(self, global_logits):
        self.global_logits = copy.deepcopy(global_logits)

    # def train_metrics(self):
    #     trainloader = self.load_train_data()
    #     # self.model = self.load_model('model')
    #     # self.model.to(self.device)
    #     self.model.eval()

    #     train_num = 0
    #     losses = 0
    #     lsf = sigmoid_focal_loss
    #     self.model.to(self.device)
    #     with torch.no_grad():
    #         for x, y in trainloader:
    #             if type(x) == type([]):
    #                 x[0] = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #             y = y.to(self.device)
    #             output = self.model(x)
    #             loss = self.loss(output, y)

    #             # if self.global_logits != None:
    #             #     logit_new = copy.deepcopy(output.detach())
    #             #     for i, yy in enumerate(y):
    #             #         y_c = yy.item()
    #             #         if type(self.global_logits[y_c]) != type([]):
    #             #             logit_new[i, :] = self.global_logits[y_c].data
    #             #     loss += self.loss_mse(logit_new, output) * self.lamda
                    
    #             train_num += y.shape[0]
    #             losses += loss.item() * y.shape[0]

    #     # self.model.cpu()
    #     # self.save_model(self.model, 'model')

    #     return losses, train_num
    
    def rewind_train_metrics(self, rewind_train_node = None):
        self.rewind_step += 1
        losses, train_num = self.train_metrics()
        loss = losses / train_num
       
        print(f"\n** REWIND: rewind loss on node's {self.id} dataset: ", loss)
        if ( rewind_train_node != None ):
            rewind_loader = rewind_train_node.load_train_data()
            rw_losses, rw_train_num = self.train_metrics(rewind_loader)
            rw_loss = rw_losses / rw_train_num
            print(f"** REWIND: rewind loss on rewind {rewind_train_node.id} dataset: ", rw_loss)
        return loss, rw_loss

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
        # self.save_model(self.model, 'mode/tral')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        y_true_tensor = torch.tensor(y_true)
        y_prob_tensor = torch.tensor(y_prob)
        if y_true_tensor.isnan().any():
            y_true = y_true_tensor.nan_to_num().numpy()
            print ( "nan in y_true", y_true)
        if y_prob_tensor.isnan().any():
            y_prob = y_prob_tensor.nan_to_num().numpy()
            print ( "nan in y_prob", y_prob)

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

def unique_node ( nodes ):
    unique = []
    for node in nodes:
        if not node in unique:
            unique.append(node)
    return unique