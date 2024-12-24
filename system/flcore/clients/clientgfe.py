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
from flcore.clients.clientRewind import clientRewind
from sklearn.preprocessing import label_binarize
from sklearn.metrics import balanced_accuracy_score
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torchvision import  transforms
from flcore.trainmodel.downstreamclassification import DownstreamClassification
from flcore.trainmodel.singlelayerclassification import SingleLayerClassification

from tqdm import tqdm

class clientGFE(clientRewind):
    def __init__(self, args, model_id, train_samples, test_samples, dataset = None, is_strong = False, id_by_type=-1, rewind_epochs = 0, rewind_interval = 0, rewind_ratio = 0, pretext_tasks = [], **kwargs):
        super().__init__(args, model_id, train_samples, test_samples, **kwargs)

        self.node_routes = []
     
        self.id_by_type = id_by_type
        self.train_loader = None
        self.no_wandb = args.no_wandb
        self.train_dataloader = None
        self.test_dataloader = None

        if dataset != None:
            self.dataset = dataset
        else:
            self.dataset = args.dataset
        
        self.node_data.dataset = self.dataset
        self.node_data.id = 0
        self.node_data_losses = []

        self.downstream_task = DownstreamClassification(self.model.backbone, num_classes=self.num_classes)
        self.model.downstream_task_set(self.downstream_task)
        self.downstream_tasks = args.downstream_tasks
        self.downstream_loss_operation = args.downstream_loss_operation
        # self.downstream_task = SingleLayerClassification(self.model.inner_model.vit.embed_dim, self.num_classes)
        self.downstream_task.to(self.device)
        self.downstream_task.loss = nn.CrossEntropyLoss()
        self.no_downstream_tasks = args.no_downstream_tasks

        self.model_optimizer = args.model_optimizer


        # self.transform = transforms.Compose(
        #     [
        #     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #      transforms.Resize([224, 224]),
        #     #  transforms.ToTensor()
        #     ])

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
        # self.model.train().to(device)

        # print ( "\n--------\nNode %d: training model %d (%d) on dataset %d " % ( self.id, self.train_model.id, self.model.id, self.node_data.id ) )
        # print ( "Training on model %s loss %s optimizer %s" % ( hex(id(self.model.inner_model)), hex(id (self.model.loss)), hex(id(self.model.optimizer)) ) )

        # if self.train_model_id == self.id:
        #     print ( f"Node {self.id} training model {self.train_model_id} on self dataset")

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        local_epochs = max_local_epochs
        if ( self.rewind_epochs > 0 and rewind_train_node != None ):
            rewind_epochs = self.rewind_epochs
        else:
            rewind_epochs, local_epochs, rewind_nodes_count = self.prepare_rewind(max_local_epochs)

        if local_epochs == 0:
            return

        epoch_start_lr = []
        epoch_end_lr = []
        starting_lr = self.local_learning_rate
        # self.model.optimizer.param_groups[0]['lr'] = self.local_learning_rate
        # print ( "Epoch starting LR ", starting_lr)

        pbarbatch = None
        num_batches = len(trainloader.dataset)//trainloader.batch_size

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        if self.model_optimizer == "AdamW":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        self.optimizer.add_param_group({'params': self.downstream_task.parameters(), 'lr': self.learning_rate}) 
        # Call it AFTER the optimizer is created, otherwise the optimizer will already have use pretext_task parameters

        # self.downstream_task.backbone = self.model.backbone
        # self.model.downstream_task_set( self.downstream_task)
        self.model.pretext_train = True

        for pretext_task_name in self.pretext_tasks:
            self.model.pretext_task_name = pretext_task_name
            print ( f"Node {self.id} training on pretext task {pretext_task_name}")
        
            self.freeze(backbone=False, pretext=False, downstream=True)
            # self.downstream_task.backbone = self.model.inner_model

            self.optimizer.add_param_group({'params': self.model.pretext_task.parameters(), 'lr': self.learning_rate})
            
            for step in range(local_epochs):
                if ( ( self.rewind_strategy == "halfway" or self.rewind_strategy == "interval" or self.rewind_strategy == "atend_pre"  ) and len(self.rewind_previous_node) > 0 ):
                    self.rewind(step, max_local_epochs, rewind_epochs, rewind_nodes_count)

                trainloader = node_trainloader
                if trainloader == None:
                    print ( f"Node {self.id} has no data")
                    return
                losses = 0
                downstream_losses = 0

                print ( "Round %d Epoch %d Optimizer: %s " % ( self.round, step, hex(id(self.model.optimizer))), end='')
                # with tqdm(total=local_epochs, desc=f"Epoch {step+1}/{local_epochs}", unit='epoch') as pbarepoch:

                    # print ( "Samples worked: ", end='') 
                pbarbatch = tqdm(total=num_batches, desc=f"Batch ", unit='batch', leave=False) 
                # pbarbatch = tqdm(total=num_batches, desc=f"Batch {i+1}/{num_batches}", unit='batch', leave=False) 
                for i, (x, y) in enumerate(trainloader):
                    if self.check_batch(x, y) == False:
                        continue

                    if type(x) == type([]):
                        x[0] = x[0].to(device)
                        # x[0] = self.transform(x[0])
                    else:
                        x = x.to(device)
                        # x= self.transform(x)
                    y = y.to(device)
                    
                    output = self.model(x)
                    # output = heads(output)
                    # loss = self.model.loss(output, y).to(device)
                    loss = self.model.loss( output, y )
                    losses += loss.item()


                    

                    # downstream_loss = loss
                    # with torch.no_grad():
                    self.downstream_task.backbone_enabled = True
                    downstream_output = self.downstream_task(x)
                    downstream_loss = self.downstream_task.loss ( downstream_output, y)
                    downstream_losses += downstream_loss.item()
                    self.downstream_task.backbone_enabled = False

                    self.optimizer.zero_grad()

                    summed_loss = loss
                    if self.downstream_loss_operation == "sum":
                        summed_loss = loss + downstream_loss
                    
                    summed_loss.backward()

                    self.optimizer.step()

                    pbarbatch.set_postfix({'Loss': f'{loss.item():.2f}', 'DSLoss': f'{downstream_loss.item():.2f}', 'Epoch': f'{step+1}/{local_epochs}'})
                    pbarbatch.update(1)
                    if ( self.args.limit_samples_number > 0 and i*trainloader.batch_size > self.args.limit_samples_number ):
                        break

                # pbarepoch.update(1)
                # print ( f"loss {losses/i:.2f} downstream loss {downstream_losses/i:2f}")
                self.data_log({f"train/node_{self.id}/pretext_train_loss_{pretext_task_name}": losses/i, "round": self.round})

            pbarbatch.close()
            print()

            print ( "Calculating pretext task accuracy")
            testloader = self.load_test_data()
            for i, (x, y) in enumerate(testloader):
                if type(x) == type([]):
                    x[0] = x[0].to(device)
                    # x[0] = self.transform(x[0])https://www.federbridge.it/Simultanei/Classifica.asp?simdate=18/12/2024&simcode=NSTNFIGB1
                else:
                    x = x.to(device)
                    # x= self.transform(x)
                y = y.to(device)
                output = self.model(x)
                pretext_accuracy = self.model.accuracy(output)
                print ( f"Pretext accuracy {pretext_accuracy}")
                break

        if self.downstream_task != None:
            backbone_grad = False
            if len(self.pretext_tasks) == 0:
                print ( f"No pretext task, not freezing backbone parameters")
                self.freeze(backbone=False, pretext=True, downstream=False)
                backbone_grad = True
            else:
                self.freeze(backbone=True, pretext=True, downstream=False)

            
            # if len(self.pretext_tasks) < len(self.optimizer.param_groups):
            #     self.optimizer.add_param_group({'params': self.downstream_task.parameters(), 'lr': self.learning_rate})



            # if self.no_downstream_tasks != True:
            #     self.model.inner_model.vit.head = self.downstream_task

            
            self.model.pretext_train = False
            # self.freeze(freeze = True)
            # self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
            
            for step in range(local_epochs):
                pbarbatch = tqdm(total=num_batches, desc=f"Batch ", unit='batch', leave=False)  

                for i, (x, y) in enumerate(trainloader):
                    if self.check_batch(x, y) == False:
                        continue

                    if type(x) == type([]):
                        x[0] = x[0].to(device)
                        # x[0] = self.transform(x[0])
                    else:
                        x = x.to(device)
                        # x= self.transform(x)
                    y = y.to(device)
                    if self.train_slow:
                        time.sleep(0.1 * np.abs(np.random.rand()))
                    
                    output = self.downstream_task(x).to(device)

                    self.optimizer.zero_grad()
                    loss = self.downstream_task.loss(output, y).to(device)

                    loss.backward()
                    self.optimizer.step()
                    pbarbatch.set_postfix({'Loss': f'{loss.item():.4f}', 'Epoch': f'{step+1}/{local_epochs}'})
                    pbarbatch.update(1)
                    if ( self.args.limit_samples_number > 0 and i*trainloader.batch_size > self.args.limit_samples_number ):
                        break

        local_loss, num = self.train_metrics()
        print ( f"Downstream task loss {local_loss/num}")
        self.data_log({f"train/node_{self.id}/downstream_train_loss": local_loss/num, "round": self.round})
        
        # self.downstream_optimizer = torch.optim.SGD(self.downstream_net.parameters(), lr=self.learning_rate)
        # for downstream_task in self.downstream_tasks:
        #     self.model.pretext_train = False
        #     self.model.pretext_task = None
        #     self.model.downstream_task = downstream_task
        #     print ( f"Downstream task {downstream_task}")
            

        # loss, num = self.train_metrics()

        # if len(self.rewind_previous_node) > 0:
        #     rewind_node = self.rewind_previous_node[-1]
        #     local_loss, rw_loss = self.rewind_train_metrics(rewind_node)
        #     if not self.no_wandb:
        #         wandb.log({f"train/model_{self.model.id}/atend_loss_on_local": local_loss, "round": self.round})
        #         wandb.log({f"train/model_{self.model.id}/atend_loss_on_previous": rw_loss, "round": self.round})



    def freeze(self, backbone = True, pretext = False, downstream = False):
        backbone_grad = not backbone
        pretext_head_grad = not pretext
        downstream_grad = not downstream

        print ( f"Freezing backbone parameters {backbone_grad} pretext head {pretext_head_grad} downstream {downstream_grad}")   
        for param in self.model.parameters():
            param.requires_grad = backbone_grad

        if self.model.inner_model.pretext_head != None:
            for param in self.model.inner_model.pretext_head.parameters():
                param.requires_grad = pretext_head_grad
        
        for param in self.downstream_task.parameters():
            param.requires_grad = downstream_grad


    def prepare_rewind(self, max_local_epochs, rewind_train_node = None):
        rewind_nodes_count = len(self.rewind_previous_node)

        if rewind_nodes_count == 0:
            return max_local_epochs, max_local_epochs, rewind_nodes_count

        if ( self.rewind_epochs > 0 and rewind_train_node != None ):
            rewind_epochs = self.rewind_epochs
        else:
            rewind_epochs = math.ceil ( max_local_epochs * self.rewind_ratio )
        local_epochs = max_local_epochs - rewind_epochs
        
        return rewind_epochs, local_epochs, rewind_nodes_count
    
    def rewind(self, step, max_local_epochs = 0, rewind_epochs = 0, rewind_node_count = 0, device = 0):

        if self.rewind_donkey:
            rewind_nodes = unique_node ( self.rewind_previous_node[-self.rewind_donkey_count::] )
        else:
            rewind_nodes = [self.rewind_previous_node[-1]]
        rewind_node_count = len(rewind_nodes)

        if rewind_epochs == 0:
            return
        
        noise = self.rewind_noise
        
        rewind_start_epoch = -1
        if ( self.rewind_strategy == "atend_pre" ):
            rewind_ending_epochs_count = math.ceil(rewind_epochs * self.rewind_end_epoch_ratio)
            rewind_start_epoch = ( max_local_epochs - rewind_epochs - rewind_ending_epochs_count)
           
        elif ( self.rewind_strategy == "halfway" ):
            rewind_start_epoch = max_local_epochs//2
        elif ( self.rewind_strategy == "interval" ):
            rewind_start_epoch = max_local_epochs / rewind_epochs // 2
        
        if ( step == rewind_start_epoch or self.rewind_strategy == "atend" ) and rewind_node_count > 0:
                if self.rewind_random:
                    rewind_nodes = [self.rewind_random_clients[np.random.randint(0, rewind_node_count)]]
                for teacher in rewind_nodes:
                    if ( teacher != None ):
                        local_loss, rw_loss = self.rewind_train_metrics(teacher)
                        if not self.no_wandb:
                            wandb.log({f"train/model_{self.model.id}/pre_rewind_loss_on_local": local_loss, "round": self.round})
                            wandb.log({f"train/model_{self.model.id}/pre_rewind_loss_on_previous": rw_loss, "round": self.round})
                        self.rewind_train ( rewind_epochs, teacher, device, noise = noise )
                        
                        local_loss, rw_loss = self.rewind_train_metrics(teacher)
                        if not self.no_wandb:
                            wandb.log({f"train/model_{self.model.id}/post_rewind_loss_on_local": local_loss, "round": self.round})
                            wandb.log({f"train/model_{self.model.id}/post_rewind_loss_on_previous": rw_loss, "round": self.round})

    def rewind_train(self, rewind_epochs = 0, rewind_train_node = None, device = 0, noise = False):

        # rewind_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        if ( rewind_epochs == 0 or rewind_train_node == None):
            return
        
        print ( "\nStep on node %d, %s rewinding to node %d for %d epochs" % (self.id, self.rewind_strategy, rewind_train_node.id, rewind_epochs ), end='' )
        
        dataloader = rewind_train_node.load_train_data()
        start_time = time.time()
        device = self.model.device
        # self.model.to(device)
        starting_lr = self.model.optimizer.param_groups[0]['lr']
        if ( self.rewind_learning_rate_decay ):
            rewind_lr = starting_lr * self.rewind_learning_rate_decay_ratio
            self.model.optimizer.param_groups[0]['lr'] = rewind_lr
            print ( f"(original LR: {starting_lr} new LR: {rewind_lr},", end='' )
            print ( f"rewind loss: ", end='')
        for step in range(rewind_epochs):
            for i, (x, y) in enumerate(dataloader):
                if type(x) == type([]):
                    x[0] = x[0].to(device)
                    self.transform(x[0])
                else:
                    x = x.to(device)
                    self.transform(x)
                if noise == True:
                    y = torch.randint(0, self.num_classes, (y.shape[0],)).to(device) 
                y = y.to(device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                if ( torch.isnan(output).any() ):
                    self.log_once ( f'\nrewind_train: nan in output {self.id}\n' )
                loss = self.model.loss(output, y)

                self.model.optimizer.zero_grad()
                
                loss.backward()
                # end_lr = self.optimizer.param_groups[0]['lr']
                self.model.optimizer.step()
            print ( f" {loss} ", end='')
        if not self.rewind_learning_rate_keep and self.rewind_learning_rate_decay:
            print ( "\nRestoring LR to ", starting_lr)
            self.model.optimizer.param_groups[0]['lr'] = starting_lr
            


        # self.rewind_step += 1
        # self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def train_metrics(self, trainloader=None):
        if ( trainloader == None):
            trainloader = self.load_train_data()
        if trainloader == None:
            print ( "No train data for client ", self.id)
            return 0, 0
        self.model.eval()

        train_num = 0
        losses = 0
        self.model.to(self.device)
        self.model.pretext_train = False


        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                if ( torch.isnan(output).any() ):
                    self.log_once ( "Output NAN")

                loss = self.model.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

            # print ( f"Downstream task loss {losses/train_num}")

        return losses, train_num

    def rewind_train_metrics(self, rewind_train_node = None):
        self.rewind_step += 1
        losses, train_num = self.train_metrics()
        loss = losses / train_num
       
        print(f"\n** REWIND: rewind loss on node's {self.id} dataset: ", loss)
        if ( rewind_train_node != None ):
            rewind_loader = rewind_train_node.load_train_data()
            rw_losses, rw_train_num = self.train_metrics(rewind_loader)
            rw_loss = rw_losses / rw_train_num
            if math.isnan(rw_loss):
                print(f"** REWIND: rewind loss on rewind {rewind_train_node.id} dataset is NAN!!")
            else:
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