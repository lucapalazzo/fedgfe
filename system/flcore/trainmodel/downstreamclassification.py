from flcore.trainmodel.downstream import Downstream
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer
from transformers import ViTModel
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import torch.nn.functional as F

from utils.node_metric import NodeMetric

import torch
import wandb


class ClassificationHeadConv(nn.Module):
    def __init__(self, input_dim, output_dim, cls_token_only=False):
        super(ClassificationHeadConv, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=768, out_channels=32, kernel_size=(3, 3), padding=1)
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        # self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(64, output_dim)
        # self.relu = nn.ReLU()
        self.cls_token_only = cls_token_only
        hidden_dim1 = 128
        hidden_dim2 = 64

        self.head = nn.Sequential(
            # Primo layer convoluzionale:
            # - in_channels: 1 (perché aggiungiamo una dimensione canale)
            # - out_channels: 32 (numero di filtri, può essere modificato)
            # - kernel_size: (3, 3) con padding=1 per mantenere la dimensione spaziale
            # self.conv1 = 
            nn.Conv2d(in_channels=768, out_channels=hidden_dim1, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(hidden_dim1),
            nn.ReLU(),
        
        # Secondo layer convoluzionale:
        # - in_channels: 32
        # - out_channels: 64 (numero di filtri)
        # - kernel_size: (3, 3) con padding=1
        # self.conv2 =
            # nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), padding=1),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
        
        # Pooling globale per aggregare le informazioni spaziali in un singolo vettore per canale
        # self.global_pool = 
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
        # Layer fully connected per la classificazione finale.
        # L'input sarà il numero di canali dell'ultimo conv layer (64)
        # self.fc = 
            nn.Linear(hidden_dim1, output_dim),
        
        # Funzione di attivazione ReLU
        # self.relu =
            # nn.ReLU()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize convolutional and linear layer weights"""
        for module in self.head:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.1)



    def forward(self, x):
        return self.head(x)
    
class ClassificationHeadLinear(nn.Module):
   
    def __init__(self, input_dim, output_dim, cls_token_only=False):
        super(ClassificationHeadLinear, self).__init__()
        self.cls_token_only = cls_token_only
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim1 = 256
        self.hidden_dim2 = 64
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_dim1),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_dim1, output_dim)
            # nn.Linear(input_dim, self.hidden_dim1),
            # nn.ReLU(),
            # nn.Linear(self.hidden_dim1, self.hidden_dim2),
            # nn.ReLU(),
            # nn.Linear(self.hidden_dim2, output_dim)
        )
        self._initialize_weights()

    def to(self, device):
        self.head.to(device)
        return self
    
    def _initialize_weights(self):
        """Initialize linear layer weights using Xavier/Glorot initialization"""
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.1)

    def forward(self, x):
        if self.cls_token_only == False:
            x = x.mean(dim=(2,3))
        return self.head(x)

class DownstreamClassification (Downstream):
    def __init__(self, backbone, num_classes=10, wandb_log=False, classification_tasks_labels=[], device=None, classification_labels_used = None, cls_token_only=False):
        super(DownstreamClassification, self).__init__(backbone, wandb_log=wandb_log, device=device)

        self.task_name = "classification"
        self.input_dim = 0
        if isinstance(backbone, VisionTransformer):
            self.input_dim = backbone.embed_dim
        elif isinstance(backbone, ViTModel):
            self.input_dim = backbone.config.hidden_size
        
        self.output_dim = num_classes
        self.num_classes = num_classes
        self.cls_token_only = cls_token_only

        self.use_conv_head = False
        if self.use_conv_head:
            self.classification_head_to_use = ClassificationHeadConv
        else:
            self.classification_head_to_use = ClassificationHeadLinear

        self.classification_task_count = len(classification_tasks_labels)
        self.classification_head = nn.ModuleList()
        self.classification_losses = []
        if classification_labels_used is None:
            self.classification_labels_used = range(self.classification_task_count)
        else:
            self.classification_labels_used = classification_labels_used
            self.classification_task_count = len(self.classification_labels_used)
        

        for task_index, task_labels in enumerate(self.classification_labels_used):
            task_labels_id = self.classification_labels_used[task_index]
            task_labels_count = classification_tasks_labels[task_labels_id]['num_classes']
            self.classification_head.add_module(f"classification_head_{task_index}", self.classification_head_to_use(self.input_dim, task_labels_count, cls_token_only=self.cls_token_only))

        self.downstream_head = self.classification_head

        self.classification_loss = nn.CrossEntropyLoss()

        # self.defined_test_metrics = [ "accuracy", "auc"] 
        # self.defined_test_metrics = { "accuracy": None, "auc": None }
        self.defined_test_metrics = { "accuracy": None }
        self.defined_train_metrics = { "loss": None }

        self.init_metrics()
        # self.test_metrics_aggregated = []
        # for task_index in range(self.classification_task_count):
        #     self.task_test_metrics[task_index] = {metric: [] for metric in self.defined_test_metrics}
        #     self.task_test_metrics_aggregated.append({metric: [] for metric in self.defined_test_metrics})

 
        self.classification_heads_losses = self.init_heads_losses()

        # self.test_metrics = {metric: [] for metric in self.defined_test_metrics}
        # self.test_metrics_aggregated = {metric: [] for metric in self.defined_test_metric

    def train(self, mode: bool = True) -> None:
        super(DownstreamClassification, self).train(mode)
        self.classification_head.train(mode)
    
    def eval(self, mode: bool = True) -> None:
        super(DownstreamClassification, self).eval()
        self.classification_head.eval()

    def define_task_metrics(self):
        task_metrics = {metric: torch.Tensor for metric in self.defined_test_metrics}
        return task_metrics

    def init_metrics(self):
        self.task_test_metrics = {}
        self.task_test_metrics_aggregated = {}
        for task_index in range(self.classification_task_count):
            self.task_test_metrics[task_index] = self.define_task_metrics()
            self.task_test_metrics_aggregated[task_index] = self.define_task_metrics()
        return self.task_test_metrics

    def init_heads_losses(self):
        losses = {}
        for task_index in range(self.classification_task_count):
            losses[task_index] = torch.tensor(0.0,requires_grad=True)
        return losses

    def named_parameters(self, prefix = '', recurse = True, remove_duplicate = True):
        # moduleList = nn.ModuleList()
        # moduleList.add_module("classification_head", self.classification_head)
        # moduleList.add_module("backbone", self.backbone)
        return self.classification_head.named_parameters(prefix, recurse, remove_duplicate)

    def parameters(self, recurse=True):
        # moduleList = nn.ModuleList()
        # moduleList.add_module("downstream_head", self.downstream_head)
        # moduleList.add_module("classification_head", self.classification_head)
        # for i, head in enumerate(self.classification_head):
        #     moduleList.add_module(f"classification_head_{i}", head)
        return self.classification_head.parameters(recurse)
    
    def parameters_count(self):
        head_params = sum(p.numel() for p in self.downstream_head.parameters())
        print ( f"Classification Head params: {head_params}")

    def define_metrics(self, metrics_path = None):
        if self.wandb_log == False:
            return None

        super().define_metrics( metrics_path=metrics_path)
        path = "/" if metrics_path is None else "/"+metrics_path+"/"
        metrics = []

        for metric in self.defined_test_metrics:
            defined_metric = f"test{path}{self.task_name}_{metric}"
            self.defined_test_metrics[metric] = defined_metric
            for task_id in range(self.classification_task_count):
                task_defined_metric = f"{defined_metric}_{task_id}"
                metrics.append(task_defined_metric)
            
        for metric in self.defined_train_metrics:
            defined_metric = f"train{path}{self.task_name}_{metric}"
            self.defined_train_metrics[metric] = defined_metric
            for task_id in range(self.classification_task_count):
                task_defined_metric = f"{defined_metric}_{task_id}"
                metrics.append(task_defined_metric)

        loss_metric = f"train{path}{self.task_name}_loss"
        metrics.append(loss_metric)
        
        for metric in metrics:
            a = wandb.define_metric(metric, step_metric="round")
    
    def test_metrics_log(self, metrics = None, round = None):
        if metrics is None:
            return

        if round is None or round < 0:
            round = self.round

        if self.wandb_log == True:
            for task_id in range(metrics.task_count):
                for metric_type in metrics.metrics:
                    if metric_type in self.defined_test_metrics:
                        metric_path = f"{self.defined_test_metrics[metric_type]}_{task_id}"
                        metric_value = metrics[task_id][metric_type]/metrics[task_id]['steps'] if metrics[task_id]['steps'] > 0 else 0
                        wandb.log({metric_path: metric_value, "round": round})

    def train_metrics ( self, dataloader, metrics = None):
        train_num = 0
        total_loss = 0

        metrics = NodeMetric(phase=NodeMetric.Phase.TRAIN)
        metrics.define_metrics(self.defined_train_metrics, task_count=self.classification_task_count)
        metrics.task_name = self.task_name
        metrics.task_type = NodeMetric.TaskType.CLASSIFICATION

        tasks_losses = [torch.tensor(0.0,requires_grad=True) for i in range(self.classification_task_count)]
        step = 0
        samples = 0
        with torch.no_grad():
            for x, y in dataloader:
                batch_losses = [torch.tensor(0.0,requires_grad=True) for i in range(self.classification_task_count)]
                step += 1
                samples += x.shape[0]
                if type(y) == dict:
                    if 'labels' in y:
                        y = y['labels']
                # if len(y.shape) > 1 and y.shape[1] > 1:
                #     y = y [:,task_labels_id]

                x = x.to(self.device)
                y = y.to(self.device)
                output = self(x)

                if output == None:
                    continue

                loss = self.downstream_loss(output, y, losses = batch_losses)
                train_num += 1
                total_loss += loss.item()

                for task_index in range(self.classification_task_count):
                    tasks_losses[task_index] += batch_losses[task_index].item()


            total_loss /= train_num
            
            for task_index in range(self.classification_task_count):
                tasks_losses[task_index] /= train_num

            for task_index in range(self.classification_task_count):
                metrics[task_index]['steps'] = train_num
                metrics[task_index]['samples'] = samples
                metrics[task_index]['loss'] = tasks_losses[task_index]

            # print ( f"Downstream task loss {losses/train_num}")
            self.train_metrics_log( metrics=metrics, round = self.round )
        return metrics

    def train_metrics_log(self, metrics = None, round = 0):
        if metrics is None:
            return

        if round is None or round < 0:
            round = self.round

        if self.wandb_log == True:
            for task_id in range(metrics.task_count):
                for metric_type in metrics._defined_metrics:
                    if metric_type in self.defined_train_metrics:
                        metric_path = f"{self.defined_train_metrics[metric_type]}_{task_id}"
                        metrics_value = metrics[task_id][metric_type]/metrics[task_id]['steps'] if metrics[task_id]['steps'] > 0 else 0
                        wandb.log({metric_path: metrics_value, "round": round})

    def test_metrics(self, logits, labels = None, samples = None, round = None, metrics = None, task = 0):
        batch_size = labels.shape[0]

        task_metrics = NodeMetric(phase=NodeMetric.Phase.TEST, task_count=self.classification_task_count)
        task_metrics.define_metrics( self.defined_test_metrics, task_count=self.classification_task_count)
        task_metrics.task_name = self.task_name
        task_metrics.task_type = NodeMetric.TaskType.CLASSIFICATION

        accuracy, num, auc, y_true, y_prob = self.test_metrics_accuracy(logits, labels)
        task_metrics[task]['samples'] = num
        task_metrics[task]['accuracy'] = accuracy/num
        task_metrics[task]['steps'] = 1

        return task_metrics
    
    def test_metrics_accuracy(self, logits, labels):
       
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
                task_y_prob.append(task_prob.detach().cpu().numpy()) 

                task_y_true.append(task_labels.detach().cpu().numpy())

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
        else:
            task_prob = []
            task_y_prob = []
            task_y_true = []
            task_test_acc = 0
            task_test_num = 0

            if len(labels.shape) == 1:
                task_labels = labels
            else: 
                task_labels = labels[:, task_index]

            task_logits = logits    
            task_predictions = torch.argmax(task_logits, dim=1)
            predictions = task_predictions
            task_test_acc += (torch.sum(task_predictions == task_labels)).item()
            task_test_num += labels.shape[0]

            if torch.isnan(task_logits).any().item():
                if not self.no_wandb:
                   wandb.log({f'warning/{self.id}': torch.isnan(logits)})
              # print(f'warning for client {self.id} in round {self.round}:', torch.isnan(output))
                self.log_once(f'warning for client {self.id} in round {self.round}: output contains nan"')

            task_prob = F.softmax(task_logits, dim=1).detach().cpu().numpy()
            task_true = task_labels.detach().cpu().numpy() 
            task_y_prob.append(task_prob) 
            task_y_true.append(task_true)

            if len(task_y_prob) > 0:
                task_y_prob = np.concatenate(task_y_prob, axis=0)
                task_y_true = np.concatenate(task_y_true, axis=0)
            
            # auc = metrics.roc_auc_score(task_true, task_prob, average='micro')
            task_auc = 0
            
            test_acc.append(task_test_acc)
            test_num.append(task_test_num)
            y_prob.append(task_y_prob)
            y_true.append(task_y_true)
            auc.append(task_auc)
        
        return task_test_acc, task_test_num, auc, task_y_true, task_y_prob

            
    def downstream_loss(self, logits, labels, samples = None, losses = None):

        if isinstance(labels, dict):
            labels = labels['labels']
        labels = labels.long()
        tasks_loss = {}

        loss = None

        for task_id, task_logits in enumerate(logits):
            classification_label_id = self.classification_labels_used[task_id]
            if len(labels.shape) == 1:
                task_labels = labels
            else:
                task_labels = labels[:, classification_label_id].long()
            task_loss = self.classification_loss(task_logits, task_labels)
            tasks_loss[task_id] = task_loss

        if losses is not None:
            for task_index in range(self.classification_task_count):
                losses[task_index] = tasks_loss[task_index]

        loss = sum(tasks_loss.values())
        # loss /= self.classification_task_count
        return loss

    def loss(self, logits, labels, samples=None):
        return self.downstream_loss(logits, labels, samples=samples)

    def forward(self, x):
        x = super().forward(x)
        # x = self.downstream_head(x)

        if self.cls_token_only == False:
            patch_size = int(np.sqrt(x.shape[1]))
            x = x.permute(0,2,1)
            x = x.view(-1, 768, patch_size, patch_size)
        
        output = []
        for i, head in enumerate(self.classification_head):
            # Only move to device if not already on correct device
            if next(head.parameters()).device != x.device:
                head.to(x.device)
            output.append(head(x))
        return output