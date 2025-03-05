from flcore.trainmodel.downstream import Downstream
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer
from transformers import ViTModel
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import torch.nn.functional as F

import torch
import wandb


class ClassificationHeadConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ClassificationHeadConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=768, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

        self.head = nn.Sequential(
            # Primo layer convoluzionale:
            # - in_channels: 1 (perché aggiungiamo una dimensione canale)
            # - out_channels: 32 (numero di filtri, può essere modificato)
            # - kernel_size: (3, 3) con padding=1 per mantenere la dimensione spaziale
            # self.conv1 = 
            nn.Conv2d(in_channels=768, out_channels=32, kernel_size=(3, 3), padding=1),
        
        # Secondo layer convoluzionale:
        # - in_channels: 32
        # - out_channels: 64 (numero di filtri)
        # - kernel_size: (3, 3) con padding=1
        # self.conv2 =
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
        
        # Pooling globale per aggregare le informazioni spaziali in un singolo vettore per canale
        # self.global_pool = 
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        # Layer fully connected per la classificazione finale.
        # L'input sarà il numero di canali dell'ultimo conv layer (64)
        # self.fc = 
            nn.Linear(64, output_dim),
        
        # Funzione di attivazione ReLU
        # self.relu =
            nn.ReLU()
        )

        

    def forward(self, x):
        x = x[:,1:,:].permute(0,2,1)
        x = x.view(-1, 768, 14, 14)  
        return self.head(x)
    
class ClassificationHeadLinear(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ClassificationHeadLinear, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.head(x)

class DownstreamClassification (Downstream):
    def __init__(self, backbone, num_classes=10, wandb_log=False, classification_tasks_labels=[]):
        super(DownstreamClassification, self).__init__(backbone, wandb_log=wandb_log)

        self.input_dim = 0
        if isinstance(backbone, VisionTransformer):
            self.input_dim = backbone.embed_dim
        elif isinstance(backbone, ViTModel):
            self.input_dim = backbone.config.hidden_size
        
        self.output_dim = num_classes
        self.num_classes = num_classes

        self.classification_task_count = len(classification_tasks_labels)
        self.classification_head = nn.ModuleList()
        self.classification_losses = []
        for task_index, task_labels in enumerate(classification_tasks_labels):
            task_labels_count = max(task_labels)+1
            self.classification_head.add_module(f"classification_head_{task_index}", ClassificationHeadConv(self.input_dim, task_labels_count))
            # self.classification_losses.append(nn.CrossEntropyLoss())

        self.downstream_head = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
        )

        self.classification_loss = nn.CrossEntropyLoss()

        # self.defined_test_metrics = [ "accuracy", "auc"] 
        self.defined_test_metrics = [ "accuracy" ] 
        self.defined_train_metrics = [ "loss" ] 

        self.test_metrics = []
        self.test_metrics_aggregated = []
        for task in classification_tasks_labels:
            self.test_metrics.append({metric: [] for metric in self.defined_test_metrics})
            self.test_metrics_aggregated.append({metric: [] for metric in self.defined_test_metrics})

 
        self.classification_heads_losses = self.init_heads_losses()

        # self.test_metrics = {metric: [] for metric in self.defined_test_metrics}
        # self.test_metrics_aggregated = {metric: [] for metric in self.defined_test_metrics}


    def init_heads_losses(self):
        losses = [torch.tensor(0.0,requires_grad=True) for i in range(self.classification_task_count)]
        return losses
    
    def parameters(self, recurse=True):
        moduleList = nn.ModuleList()
        moduleList.add_module("downstream_head", self.downstream_head)
        moduleList.add_module("classification_head", self.classification_head)
        # for i, head in enumerate(self.classification_head):
        #     moduleList.add_module(f"classification_head_{i}", head)
        
        return moduleList.parameters(recurse)
    
    def parameters_count(self):
        head_params = sum(p.numel() for p in self.downstream_head.parameters())
        print ( f"Head params: {head_params}")

    def define_metrics(self, metrics_path = None):
        if self.wandb_log == False:
            return None

        super().define_metrics( metrics_path=metrics_path)
        path = "/" if metrics_path is None else "/"+metrics_path+"/"
        metrics = []

        for task_index in range(self.classification_task_count):
            for metric in self.defined_test_metrics:
                metrics.append(f"test{path}{metric}_{task_index}")
            
            for metric in self.defined_train_metrics:
                metrics.append(f"train{path}{metric}_{task_index}")
        
            for metric in metrics:
                a = wandb.define_metric(metric, step_metric="round")
    
    def test_metrics_log(self, round = None, prefix="test"):
        if round is None:
            round = self.round

        if self.wandb_log == True:
            for task_index in range(self.classification_task_count):
                wandb.log({f"test/{self.metrics_path}/accuracy_{task_index}": self.test_metrics_aggregated[task_index]['accuracy']})
                # wandb.log({f"test/{self.metrics_path}/auc_{task_index}": self.test_metrics_aggregated[task_index]['auc']})

    def train_metrics ( self, dataloader ):
        train_num = 0
        losses = 0
        heads_losses = self.init_heads_losses()
        with torch.no_grad():
            for x, y in dataloader:
                if type(y) == dict:
                    if 'labels' in y:
                        y = y['labels']

                y = y.to(x.device)
                output = self(x)

                if output == None:
                    continue

                loss = self.downstream_loss(output, y, losses = heads_losses)
                train_num += 1
                losses += loss.item()

            losses /= train_num
            heads_losses = [loss/train_num for loss in heads_losses]

            # print ( f"Downstream task loss {losses/train_num}")
            self.train_metrics_log( loss=losses, round = self.round, heads_losses=heads_losses )


    def train_metrics_log(self, loss = None, round = 0, heads_losses = None, prefix="train"):
        if self.wandb_log == True:
            if loss is not None:
                wandb.log({f"train/{self.metrics_path}/loss": loss, "round": round})
            if heads_losses is not None:
                for task_index in range(self.classification_task_count):
                    wandb.log({f"train/{self.metrics_path}/loss_{task_index}": heads_losses[task_index], "round": round})

    def test_metrics_calculate(self, logits, labels, round = None):

        if round is None:
            round = self.round

        batch_size = labels.shape[0]
        accuracy, num, auc, y_true, y_prob = self.test_metrics_accuracy(logits, labels)

        if self.test_running_round != self.round:
            for task_index in range(self.classification_task_count):
                self.test_metrics[task_index]['accuracy'] = []
                self.test_metrics[task_index]['auc'] = []
            self.test_running_round = self.round

        for task_index in range(self.classification_task_count):
            self.test_metrics[task_index]['accuracy'].append(accuracy[task_index]/num[task_index])
            self.test_metrics[task_index]['auc'].append(auc[task_index])
        # self.test_metrics['accuracy'].append(accuracy/batch_size)
        # self.test_metrics['auc'].append(auc/batch_size)

        self.test_metrics_aggregrate(self.test_metrics,batch_size=batch_size)
        
        # self.test_metrics_aggregated['accuracy'] = torch.mean(torch.stack(self.test_metrics['accuracy']))
        # self.test_metrics_aggregated['auc'] = torch.mean(torch.stack(self.test_metrics['auc']))
        
        # print ( f"Accuracy: {self.test_metrics_aggregated['accuracy']}, AUC: {self.test_metrics_aggregated['auc']}")

        return self.test_metrics
    
    def test_metrics_aggregrate (self, metrics, batch_size = 1):
        for metric in self.defined_test_metrics:
            for task_index, task_metric in enumerate(metrics):
                metric_tensor = torch.tensor(task_metric[metric],dtype=torch.float32)
                if len(metric_tensor) > 0:
                    self.test_metrics_aggregated[task_index][metric] = torch.mean(metric_tensor)
                else:
                    self.test_metrics_aggregated[task_index][metric] = metric_tensor
        # metric_tensor = torch.tensor(metrics[metric],dtype=torch.float32)
        # if len(metric_tensor) > 0:
        #     self.test_metrics_aggregated[metric] = torch.mean(metric_tensor)
        # else:
        #     self.test_metrics_aggregated[metric] = metric_tensor

        return self.test_metrics_aggregated

    
    def test_metrics_accuracy(self, logits, labels):
        super().test_metrics_accuracy(logits, labels) 
       
        test_acc = []
        test_num = []
        y_prob = []
        y_true = []
        auc = []

        if isinstance(logits, list):
            predictions = []
            for task_index,task_logits in enumerate(logits):
                task_prob = []
                task_y_prob = []
                task_y_true = []
                task_test_acc = 0
                task_test_num = 0
                if len(labels.shape) == 1:
                    task_labels = labels
                else: 
                    task_labels = labels[:, task_index]
                    
                task_predictions = torch.argmax(task_logits, dim=1)
                predictions.append(task_predictions)
                task_test_acc += (torch.sum(task_predictions == task_labels)).item()
                task_test_num += labels.shape[0]

                if torch.isnan(task_logits).any().item():
                    if not self.no_wandb:
                       wandb.log({f'warning/{self.id}': torch.isnan(logits)})
                  # print(f'warning for client {self.id} in round {self.round}:', torch.isnan(output))
                    self.log_once(f'warning for client {self.id} in round {self.round}: output contains nan"')

                task_prob = F.softmax(task_logits, dim=1) 
        # y_prob.append(prob.detach().cpu().numpy()) 
                task_y_prob.append(task_logits.detach().cpu().numpy()) 
        # nc = self.num_classes
        # if self.num_classes == 2:
        #     nc += 1
        # lb = label_binarize(labels.detach().cpu().numpy(), classes=np.arange(nc))
        # if self.num_classes == 2:
        #     lb = lb[:, :2]
                task_y_true.append(task_labels.detach().cpu().numpy())

                # if self.downstream_task_name == 'segmentation':
                    # groundtruth_mask = y if y.shape[1] == 1 else y.permute(0, 3, 1, 2)
                    # pred_mask = x
                    # b, c, h, w = pred_mask.shape
                    # if groundtruth_mask.shape[2] != h or groundtruth_mask.shape[3] != w:
                    #     groundtruth_mask = F.interpolate(groundtruth_mask, size=(h, w), mode='nearest')

                    # intersect = np.sum(pred_mask*groundtruth_mask)
                    # total_pixel_pred = np.sum(pred_mask)
                    # precision = np.mean(intersect/total_pixel_pred)
                    # return round(precision, 3)
                    # predictions = torch.argmax(output, dim=1)

                if len(task_y_prob) > 0:
                    task_y_prob = np.concatenate(task_y_prob, axis=0)
                    task_y_true = np.concatenate(task_y_true, axis=0)
                    task_prob = task_prob.detach().cpu().numpy()
                
                # auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
                task_auc = 0
                
                test_acc.append(task_test_acc)
                test_num.append(task_test_num)
                y_prob.append(task_y_prob)
                y_true.append(task_y_true)
                auc.append(task_auc)
        
        return test_acc, test_num, auc, y_true, y_prob

            
    def downstream_loss(self, logits, labels, samples = None, losses = None):

        if isinstance(labels, dict):
            labels = labels['labels']
        labels = labels.long()

        # if self.metrics_last_round != self.round:
        #     self.init_heads_losses()
        #     self.metrics_last_round = self.round
        #     self.metrics_round_epochs_count = 0

        # self.metrics_round_epochs_count += 1

        loss = torch.tensor(0.0, requires_grad=True, device=labels.device)
        for task_id, task_logits in enumerate(logits):
            if len(labels.shape) == 1:
                task_labels = labels
            else:
                task_labels = labels[:, task_id].long()
            task_loss = self.classification_loss(task_logits, task_labels)
            if losses is not None:  
                losses[task_id] = losses[task_id] + task_loss
            loss = loss + task_loss
        return loss
        return self.classification_loss(logits, labels)

    # def parameters(self, recurse=True):
    #     moduleList = nn.ModuleList()
    #     moduleList.add_module("downstream_head", self.downstream_head)
    #     return moduleList.parameters(recurse)

    def forward(self, x):
        x = super().forward(x)
        # x = self.downstream_head(x)
        output = []
        for i, head in enumerate(self.classification_head):
            head.to(x.device)
            output.append(head(x))
        return output
    
    def backbone_forward(self, x):
        return super().backbone_forward(x)
