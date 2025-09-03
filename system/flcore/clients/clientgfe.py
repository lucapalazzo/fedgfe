from collections import defaultdict
import copy
import os
import sys
import sklearn
import torch
import torch.nn as nn
import numpy as np
import time

import wandb

from flcore.clients.clientRewind import clientRewind
from modelutils.optimizer_manager import OptimizerManager
from modelutils.pretext_trainer import PretextTrainer
from modelutils.downstream_trainer import DownstreamTrainer
from sklearn.preprocessing import label_binarize
from sklearn.metrics import balanced_accuracy_score
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torchvision import  transforms
from flcore.trainmodel.downstreamclassification import DownstreamClassification
from flcore.trainmodel.downstreamfivelayerclassification import DownstreamFiveLayerClassification
from flcore.trainmodel.downstreamsegmentation import DownstreamSegmentation

from flcore.trainmodel.imagerotation import ImageRotation
from flcore.trainmodel.patchordering import PatchOrdering
from flcore.trainmodel.patchmasking import PatchMasking
from utils.check_parameters import check_optimizer_params, print_model_gradients_status
from utils.node_metric import NodeMetric

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR


from modelutils.model_stats import count_changed_weights

import torch.nn.functional as F


from tqdm import tqdm

from torchviz import make_dot

from collections import Counter


class clientGFE(clientRewind):
    def __init__(self, args, model_id, train_samples, test_samples, dataset = None, is_strong = False, id_by_type=-1, rewind_epochs = 0, rewind_interval = 0, rewind_ratio = 0, pretext_tasks = [], model=None, patch_count = -1, img_size = 224, patch_size = -1, **kwargs):
        super().__init__(args, model_id, train_samples, test_samples, model=model, **kwargs)

        self.node_routes = []
     
        self.id_by_type = id_by_type
        self.train_loader = None
        self.no_wandb = args.no_wandb
        self.train_dataloader = None
        self.test_dataloader = None
        self.ssl_round = 0
        self.global_rounds = args.global_rounds

        if 'data_log' in kwargs:
            self.data_log = kwargs['data_log']
        else:
            self.data_log = None

        self.cls_token_only = args.cls_token_only

        self.optimizer_weight_decay = args.model_optimizer_weight_decay
        self.optimizer_momentum = args.model_optimizer_momentum

        self.img_size = img_size
        if patch_count > 0:
            self.patch_count = patch_count
            self.patch_size = self.img_size // (patch_count ** 0.5)
        elif patch_size > 0:
            self.patch_size = patch_size
            self.patch_count = (self.img_size // patch_size) ** 2


        if dataset != None:
            self.dataset = dataset
        else:
            self.dataset = args.dataset
        
        self.node_data.dataset = self.dataset
        self.node_data.dataset_id = 0
        self.node_data_losses = []

        self.downstream_task_name = 'none'
        self.downstream_loss_operation = args.downstream_loss_operation
        self.no_downstream_tasks = args.no_downstream_tasks

        self.train_optimizer = None
        self.finetuning_optimizer = None

        self.model_optimizer = args.model_optimizer
        self.segmentation_masks_count = self.node_data.segmentation_mask_count()
        self.classification_labels_count = self.node_data.classification_labels_count()
        self.classification_labels_used = [ i for i in range(self.classification_labels_count) ]
        # self.classification_labels_used = [2]
        
        self.train_time_cost['num_sslrounds'] = 0
        self.defined_test_metrics = {}
        
        # Initialize optimizer manager
        self.optimizer_manager = OptimizerManager(
            optimizer_type=self.model_optimizer,
            learning_rate=self.learning_rate,
            weight_decay=self.optimizer_weight_decay,
            momentum=self.optimizer_momentum
        )
        
        # Initialize pretext trainer
        self.pretext_trainer = PretextTrainer(
            client_id=self.id,
            args=args,
            data_log_callback=self.data_log
        )
        
        # Initialize downstream trainer
        self.downstream_trainer = DownstreamTrainer(
            client_id=self.id,
            args=args
        )

    @property
    def downstream_task_name(self):
        return self._downstream_task_name
    
    @downstream_task_name.setter
    def downstream_task_name(self, value):
        self._downstream_task_name = value

        if value == "none":
            self.downstream_task = None
        elif value == "classification":
            dataset_labels_count = self.node_data.classification_labels_count()
            if self.classification_labels_count > dataset_labels_count:
                self.classification_labels_count = dataset_labels_count
            stats = self.node_data.train_stats_get()
           
            classification_labels = stats[0]
            self.classification_labels_count = len(classification_labels)
            # classification_labels = [stats[0][i] for i in range(self.classification_labels_count)]
            
            self.downstream_task = DownstreamClassification(self.model.backbone, num_classes=self.num_classes,wandb_log=(not self.no_wandb), classification_tasks_labels=classification_labels, classification_labels_used = self.classification_labels_used, cls_token_only=self.cls_token_only).to(self.device)
        elif value == "segmentation":
            masks = self.node_data.segmentation_mask_count()
            self.downstream_task = DownstreamSegmentation(self.model.backbone, num_classes=masks, patch_size=self.patch_size,wandb_log=(not self.no_wandb))
            self.downstream_task.segmentation_mask_threshold = self.args.segmentation_mask_threshold
        elif value == "5lclassification":
            self.downstream_task = DownstreamFiveLayerClassification(self.model.backbone, num_classes=self.num_classes,wandb_log=(not self.no_wandb))

        if self.downstream_task != None:
            self.downstream_task.id = self.id
            self.model.downstream_task_set(self.downstream_task)
    
    def define_metrics(self):
        defined_metrics = []
        if self.downstream_task != None:
            for metric in self.downstream_task.defined_test_metrics:
                defined_test_metric = f"{self.metrics_path}/{metric}"
                self.defined_test_metrics[metric] = defined_test_metric
                print ( f"Node {self.id} defined metric {defined_test_metric}")
            metrics = self.downstream_task.define_metrics(self.metrics_path)
            # defined_metrics += metrics
        

        for pretext_task in self.pretext_tasks:
            module = None
            if ( pretext_task == ImageRotation.pretext_task_name):
                module = ImageRotation
            elif ( pretext_task == PatchOrdering.pretext_task_name):
                module = PatchOrdering
            elif ( pretext_task == PatchMasking.pretext_task_name):
                module = PatchMasking

            if module != None:
                metrics = module.define_metrics(self.metrics_path)
                defined_metrics += metrics
        return defined_metrics

    def log_metrics(self, metrics, round = None):
        if metrics == None:
            return
        task_name = self.model.task_name

        for metric_name in metrics._defined_metrics:
            if metrics.phase == NodeMetric.Phase.TRAIN:
                metric_path = f"train/{self.metrics_path}/{task_name}_{metric_name}"
            elif metrics.phase == NodeMetric.Phase.TEST:
                metric_path = f"test/{self.metrics_path}/{task_name}_{metric_name}"
            metric_value = metrics[metric_name]['mean']
            if round == None:
                round = self.round

            self.data_log({metric_path: metric_value, "round": round})
           

    def get_label(self, y):
        if type(y) == dict:
            if self.downstream_task_name == "segmentation":
                if 'semantic_masks' in y:
                    y = y['semantic_masks'].to(self.device)
                else:
                    y = y['masks'].to(self.device)
            elif self.downstream_task_name == "classification":
                y = y['labels'].long().to(self.device)
        return y
    
    # def train_downstream_loss(self, output, y):
    #     self.downstream_task.backbone_enabled = True
    #     downstream_output = self.downstream_task(x)
    #     if downstream_output != None:
    #         downstream_loss = self.downstream_task.loss ( downstream_output, y)
    #         downstream_losses += downstream_loss.item()
           

    #     self.downstream_task.backbone_enabled = False
    #     return downstream_loss
    

    def train(self, client_device = None, rewind_train_node = None, training_task = "both"):
        backbone_id = hex(id(self.model.backbone))
        downstream_backbone_id = hex(id(self.downstream_task.backbone))
        print ( f"Node {self.id} Task {self.downstream_task_name} backbone id {backbone_id} downstream backbone id {downstream_backbone_id}")

        print ( f"*** Node {self.id} memory before training {torch.cuda.memory_allocated(self.device)//(1024**2)} MB")
        node_trainloader = self.load_train_data()
        node_testloader = self.load_test_data()

        trainloader = node_trainloader
        start_time = time.time()
        device = self.device
        if (client_device != None):
            device = client_device
        
        # Move model and data to GPU before training
        self._move_to_gpu(self.device)
        
        if self.loss_weights != None and self.round == 0:
            print ( f"Node {self.id} setting loss weights to {self.loss_weights}")

        self.rewind_previous_model_id.append(self.next_train_model_id)
        max_local_epochs = self.local_epochs
        # if self.train_slow:
        #     max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        local_epochs = max_local_epochs
        if ( self.rewind_epochs > 0 and rewind_train_node != None ):
            rewind_epochs = self.rewind_epochs
        else:
            rewind_epochs, local_epochs, rewind_nodes_count = self.prepare_rewind(max_local_epochs)

        if local_epochs == 0:
            return

        pbarbatch = None
        num_batches = len(trainloader.dataset)//trainloader.batch_size

        # if self.round == 1:
        if self.round == 1:
            print (f"Creating optimizer for node {self.id} with model optimizer {self.model_optimizer} and learning rate {self.learning_rate}")
            self.setup_optimizer()
            self.optimizer = self.train_optimizer

            print ( "Creating learning rate scheduler for node %d" % self.id)
            self.scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=self.global_rounds)

            if self.learning_rate_schedule:
                self.setup_learning_rate_scheduler(self.global_rounds)

            
        
        # else:
        #     saved_optimizer_state = self.optimizer.state_dict()
        #     if self.model_optimizer.lower() == "adamw":
        #         self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        #     elif self.model_optimizer.lower() == "sgd":
        #         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        #     if self.downstream_task != None:
        #         self.optimizer.add_param_group({'params': self.downstream_task.parameters(), 'lr': self.learning_rate})
        #     # self.optimizer.load_state_dict(saved_optimizer_state)


        if self.round == 1:
            if self.no_wandb == False:
                wandb.watch(self.model.backbone, log='all', log_freq=100, criterion=None, log_graph=False, idx=self.id)
                # wandb.watch(self.downstream_task.vitseg.transformer.encoder, log='all', log_freq=100, criterion=None, log_graph=False, idx=self.id+10)  
                # wandb.watch(self.model.backbone, log='all', log_freq=100, criterion=None, log_graph=False, idx=self.id)

            self.optimizer_manager.print_optimizer_info(self.id)


        self.model.round = self.round
        self.model.train()
        self.model.pretext_train = True
        
        # Calcola la media dei pesi per diversi layer del backbone
        layer_means = {}
        # for i in range(min(4, len(self.model.backbone.encoder.layer))):
        #     layer_means[f"layer_{i}_query"] = self.model.backbone.encoder.layer[i].attention.attention.query.weight.mean().item()
        #     layer_means[f"layer_{i}_key"] = self.model.backbone.encoder.layer[i].attention.attention.key.weight.mean().item()
        #     layer_means[f"layer_{i}_value"] = self.model.backbone.encoder.layer[i].attention.attention.value.weight.mean().item()
        #     layer_means[f"layer_{i}_output"] = self.model.backbone.encoder.layer[i].attention.output.dense.weight.mean().item()
        #     layer_means[f"layer_{i}_mlp"] = self.model.backbone.encoder.layer[i].intermediate.dense.weight.mean().item()
        
        # all_means = sum(layer_means.values()) / len(layer_means)
        # print(f"+++ Backbone mean weight across all sampled layers: {all_means:.6f}")
        # print(f"+++ Layer-wise means: {layer_means}")
        
        if ( training_task == "both" or training_task == "pretext" )and len(self.pretext_tasks) > 0:
            print ( f"Node {self.id} memory before pretext training {torch.cuda.memory_allocated(device)//1024**2} MB")
            self.train_pretext(local_epochs, trainloader, client_device = device, training_task = training_task)
            print ( f"Node {self.id} memory after pretext training {torch.cuda.memory_allocated(device)//1024**2} MB")
            # self.test_pretext(trainloader, metrics=pretext_testmetric_on_train)
            # self.test_pretext(node_testloader, metric = pretext_testmetric)

        self.model.pretext_train = False 
        if self.downstream_task != None and ( training_task == "both" or training_task == "downstream"):
            print ( f"Node {self.id} memory before downstream training {torch.cuda.memory_allocated(device)//1024**2} MB")
            self.train_downstream(local_epochs, trainloader, node_device = device, training_task = training_task)
            print ( f"Node {self.id} memory after downstream training {torch.cuda.memory_allocated(device)//1024**2} MB")


        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        
        # Move model and downstream task to CPU after training to save GPU memory
        self._move_to_cpu()
        print( f"*** Node {self.id} memory after training and moving to CPU {torch.cuda.memory_allocated(device)//1024**2} MB")
        
    def train_downstream(self, epochs, dataloader, node_device=None, training_task="both"):
        """Train downstream task using the DownstreamTrainer."""
        device = node_device if node_device is not None else self.device
        
        self.downstream_trainer.train_downstream(
            model=self.model,
            downstream_task=self.downstream_task,
            epochs=epochs,
            dataloader=dataloader,
            optimizer=self.optimizer,
            pretext_tasks=self.pretext_tasks,
            device=device,
            training_task=training_task,
            model_freeze_callback=self.model_freeze,
            get_label_callback=self.get_label,
            check_batch_callback=self.check_batch
        )

    def train_pretext(self, epochs, dataloader, client_device=None, training_task="both"):
        """Train pretext tasks using the PretextTrainer."""
        self.train_time_cost['num_sslrounds'] += 1
        
        device = client_device if client_device is not None else self.device
        
        # self.pretext_trainer.train_pretext(
        #     model=self.model,
        #     pretext_tasks=self.pretext_tasks,
        #     epochs=epochs,
        #     dataloader=dataloader,
        #     optimizer=self.optimizer,
        #     optimizer_manager=self.optimizer_manager,
        #     downstream_task=self.downstream_task,
        #     downstream_loss_operation=self.downstream_loss_operation,
        #     device=device,
        #     ssl_round=self.ssl_round,
        #     model_freeze_callback=self.model_freeze,
        #     get_label_callback=self.get_label,
        #     copy_model_params_callback=self.copy_model_parameters,
        #     count_updated_params_callback=self.count_updated_params
        # )

        self.pretext_trainer.train_pretext(
            model=self.model,
            pretext_tasks=self.pretext_tasks,
            epochs=epochs,
            dataloader=dataloader,
            optimizer=self.optimizer,
            optimizer_manager=self.optimizer_manager,
            downstream_task=self.downstream_task,
            downstream_loss_operation=self.downstream_loss_operation,
            device=device,
            ssl_round=self.ssl_round,
            model_freeze_callback=self.model_freeze,
            get_label_callback=self.get_label,
            copy_model_params_callback=None,
            count_updated_params_callback=None
        )
        
        # Test pretext on separate test data
        test_dataloader = self.load_test_data()
        pretext_testmetric = NodeMetric(phase=NodeMetric.Phase.TEST)
        pretext_testmetric.define_metrics(self.model.defined_test_metrics)
        pretext_testmetric_on_train = NodeMetric(phase=NodeMetric.Phase.TRAIN)
        pretext_testmetric_on_train.define_metrics(self.model.defined_test_metrics)
        
        self.test_pretext(test_dataloader, metrics=pretext_testmetric)
        self.test_pretext(dataloader, metrics=pretext_testmetric_on_train)
        self.log_metrics(pretext_testmetric, round=self.round)
        
        for pretext_task_name in self.pretext_tasks:
            print(f"Node {self.id} pretext task {pretext_task_name} metrics on train {pretext_testmetric_on_train}")
            print(f"Node {self.id} pretext task {pretext_task_name} metrics on test {pretext_testmetric}")

    def test_pretext(self, testloader=None, metrics=None):
        """Test pretext tasks using the PretextTrainer."""
        if testloader is None:
            testloader = self.load_test_data()
        
        test_metrics = self.pretext_trainer.test_pretext(
            model=self.model,
            testloader=testloader,
            get_label_callback=self.get_label,
            device=self.device
        )
        
        if metrics is not None:
            metrics += test_metrics
        
        return test_metrics
    def setup_optimizer(self):
        """Setup optimizer using the OptimizerManager."""
        self.train_optimizer = self.optimizer_manager.setup_optimizer(
            model=self.model,
            downstream_task=self.downstream_task,
            pretext_tasks=self.pretext_tasks,
            client_id=self.id
        )
        self.finetuning_optimizer = self.optimizer_manager.finetuning_optimizer

    def setup_learning_rate_scheduler(self, rounds):
        """Setup learning rate scheduler using the OptimizerManager."""
        self.scheduler = self.optimizer_manager.setup_learning_rate_scheduler(
            optimizer=self.optimizer,
            rounds=rounds,
            use_scheduler=self.learning_rate_schedule
        )
    def model_freeze(self, backbone = False, pretext = False, downstream = False):
        backbone_grad = not backbone
        pretext_head_grad = not pretext
        downstream_grad = not downstream

        # print ( f"Gradients: backbone {backbone_grad} pretext head {pretext_head_grad} downstream {downstream_grad}")   
        for param in self.model.backbone.parameters():
            param.requires_grad = backbone_grad

        if self.model.inner_model.pretext_head != None:
            for param in self.model.inner_model.pretext_head.parameters():
                param.requires_grad = pretext_head_grad
        
        if self.downstream_task != None:
            for param in self.downstream_task.parameters():
                param.requires_grad = downstream_grad

    def train_metrics(self, trainloader=None, metrics = None):


        
        if self.downstream_task == None:
            return 0, 0
        
        if ( trainloader == None):
            trainloader = self.load_train_data()
        if trainloader == None:
            print ( "No train data for client ", self.id)
            return 0, 0
        
        self.model.to(self.device)
        self.model.eval()

        task_losses = self.downstream_task.init_heads_losses()

        self.downstream_task.train_metrics( trainloader, metrics = metrics)

        train_num = 0
        task_losses = 0
        self.model.to(self.device)
        self.model.pretext_train = False


        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                if type(y) == dict:
                    if self.downstream_task_name == "segmentation":
                        if 'semantic_masks' in y:
                            y = y['semantic_masks'].to(self.device)
                        else:
                            y = y['masks'].to(self.device)
                    elif self.downstream_task_name == "classification":
                        y = y['labels'].to(self.device)

                y = y.to(self.device)
                output = self.model(x)

                if output == None:
                    continue
                # if ( torch.isnan(output).any() ):
                #     self.log_once ( "Output NAN")

                loss = self.model.loss(output, y)
                train_num += y.shape[0]
                task_losses += loss.item()

            # print ( f"Downstream task loss {losses/train_num}")
            # self.downstream_task.train_metrics_log( round = self.round )
        # Move to CPU and cleanup GPU memory
        self._move_to_cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return task_losses
        return task_losses, train_num

    def test_metrics_other(self, test_client = None):
        if ( test_client == None and test_client.id != self.id):
            return
        
        testloaderfull = test_client.load_test_data()
       
        self.model.eval()

        # test_acc = 0
        # test_num = 0
        # y_prob = []
        # y_true = []
        
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
        
        # Cleanup debug tensors
        del y_true_tensor, y_prob_tensor

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc, y_true, y_prob
    
    def set_parameters(self, global_model):
        updated_parameters = 0
        updated_parameters_problems = 0
        not_updated_parameters = 0
        node_param_id_to_name = None
        global_model_zero_parameters = 0
        node_model_zero_parameters = 0

        self._move_to_gpu(self.device)

        for global_param, node_param in zip(global_model.parameters(), self.model.backbone.parameters()):
            zero_parameter = False
            if torch.all(node_param == 0 ).item():
                node_model_zero_parameters += 1
            if torch.all(global_param == 0 ).item():
                global_model_zero_parameters += 1

            if ( torch.equal(node_param, global_param) == False):
                updated_parameters += 1
                # self.log_once ( "pre parameters not equal")
            # print ( "old_param.data", old_param.data, "new_param.data", new_param.data)
                node_param.data.copy_(global_param)
                if ( torch.equal(node_param.data, global_param.data) == False):
                    updated_parameters_problems += 1
                    if ( node_param_id_to_name == None):    
                        node_param_id_to_name = {}
                        for name, param in self.model.named_parameters():
                            node_param_id_to_name[id(param)] = name
                    print ( f"Node {self.id} parameter {node_param_id_to_name[id(global_param)]} not equal after update")
            else:
                if ( node_param_id_to_name == None):
                    node_param_id_to_name = {}
                    for name, param in self.model.named_parameters():
                        node_param_id_to_name[id(param)] = name
                if torch.all(node_param == 0).item():
                    zero_parameter = True    
                print ( f"Node {self.id} parameter {node_param_id_to_name[id(node_param)]} is zero {zero_parameter} not updated")
                not_updated_parameters += 1
                # self.log_once  ( "parameters not updated")
            # print ( "old_param.data", old_param.data, "new_param.data", new_param.data)
        print ( f"Node {self.id} Updated parameters {updated_parameters} not updated {not_updated_parameters} problems {updated_parameters_problems} zero values {global_model_zero_parameters} {node_model_zero_parameters}")
        if ( updated_parameters == 0 ):
            print ( "*** No parameters updated ***")

        self._move_to_cpu()

    def test_metrics_data_classification(self, dataloader, model, metrics = None):

        # classification_tasks = len(self.node_data.classification_labels_count)
        classification_tasks = len(self.classification_labels_used)

        node_metrics = NodeMetric(phase=NodeMetric.Phase.TEST)
        node_metrics.define_metrics(self.model.defined_test_metrics, task_count=classification_tasks)

        if metrics != None:
            metrics.define_metrics(self.downstream_task.defined_test_metrics, self.downstream_task.classification_task_count)

        for classification_task in range(classification_tasks):
            test_acc = 0
            test_num = 0
            y_pred = []
            y_prob = []
            y_true = []
            a = []   

            model.to(self.device)
            model.eval()
            self.node_data.to(self.device)
            steps = 0
            samples = 0

            test_labels_index = self.classification_labels_used[classification_task]

            batch_metrics = NodeMetric(phase=NodeMetric.Phase.TEST)
            batch_metrics.define_metrics(self.model.defined_test_metrics, task_count=classification_tasks)

            with torch.no_grad():
                for x, y in dataloader:
                    steps += 1
                    batch_metrics[classification_task].steps = steps

                    samples += x.shape[0]
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    if type(y) == dict:
                        y = y['labels'].to(self.device)
                    
                    if len(y.shape) > 1:
                        if y.shape[1] > 1:
                            y = y[:,test_labels_index]

                    y = y.to(self.device)
                    output = model(x)

                    if isinstance(output, list):
                        output = output[classification_task]

                    test_metrics = self.model.test_metrics(output, y, metrics = batch_metrics, task=classification_task)
                    node_metrics += batch_metrics

        aggregated_metrics = {}
        metrics.steps = steps
        for classification_task in range(classification_tasks):
            for key, value in node_metrics[classification_task].items():
                if metrics != None:
                    if key in metrics.defined_metrics:
                        metrics[classification_task][key] = value/samples

                if key not in aggregated_metrics:
                    aggregated_metrics[key] = []
                aggregated_metrics[key].append(value)
        if metrics != None:
            metrics['samples'] = samples

        aggregated_metrics = {key: np.mean(value) for key, value in aggregated_metrics.items()}
        node_metrics['aggregated'] = aggregated_metrics

        return node_metrics
    
    def test_metrics_data_segmentation(self, dataloader, model, metrics = None):

        node_metrics = {}

        if metrics != None:
            metrics.define_metrics(self.downstream_task.defined_test_metrics)
        task_index = 0

        samples = 0
        steps = 0
        with torch.no_grad():
            for x, y in dataloader:
                samples += x.shape[0]
                steps += 1
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                if type(y) == dict:
                    if 'semantic_masks' in y:
                        y = y['semantic_masks'].to(self.device)
                        test_num = y.shape[1]
                    else:
                        y = y['masks'].to(self.device)

                y = y.to(self.device)
                output = model(x)

                batch_metrics = self.model.test_metrics(output, y)
                if node_metrics == {}:
                    node_metrics = batch_metrics
                else:
                    # node_metrics = dict(Counter(node_metrics) + Counter(batch_metrics))
                    node_metrics = {k: node_metrics.get(k, 0) + batch_metrics.get(k, 0) for k in (set(node_metrics) | set(batch_metrics))}
            aggregated_metrics = {}
            aggregated_value = []
            metrics['samples'] = samples
            metrics['steps'] = steps

            for key, value in node_metrics.items():
                if metrics != None:
                    if key in metrics.defined_metrics:
                        metrics[task_index][key] = value.detach().cpu().numpy()/steps/samples
                if key != 'samples':
                    aggregated_value.append(value.detach().cpu().numpy())
            

        aggregated_metrics = {'samples': len(aggregated_value), 'mean': np.mean(aggregated_value), 'std': np.std(aggregated_value)}
        node_metrics['aggregated'] = aggregated_metrics
                   
                

        return node_metrics
                    

    def test_metrics_data(self, dataloader, test_model = None, metrics = None):

        if dataloader == None:
            return 0, 0, 0, [], []
        test_acc = 0
        test_num = 0
        y_pred = []
        y_prob = []
        y_true = []
        a = []   
        model = self.model

        if ( test_model != None):
            model = test_model

        # Move to GPU for testing
        model.to(self.device)
        model.eval()
        self.node_data.to(self.device)
        self.model.downstream_task.metrics_reset()
        classification_task = 0

        if self.downstream_task_name == 'classification':
            test_metrics = self.test_metrics_data_classification(dataloader, model, metrics = metrics)
        if self.downstream_task_name == 'segmentation':
            test_metrics = self.test_metrics_data_segmentation(dataloader, model, metrics = metrics)
         
        return test_metrics
    
    def _move_to_device(self, device):
        """Move model, downstream task, and data to GPU before training."""
        
        # Move model to GPU
        self.model.to(device)
        
        # Move downstream task to GPU if it exists
        if hasattr(self, 'downstream_task') and self.downstream_task is not None:
            self.downstream_task.to(device)
        
        # Move node data to GPU
        if hasattr(self, 'node_data') and self.node_data is not None:
            self.node_data.to(device)

        if isinstance( self.optimizer , torch.optim.AdamW):
            move_optimizer_state(self.optimizer, device)

        

    def _move_to_gpu(self, device):
        """Move model, downstream task, and data to GPU before training."""
        print(f"Node {self.id} moving to GPU: {device}")
        self._move_to_device(device) 
    
    def _move_to_cpu(self):
        """Move model, downstream task, and data to CPU after training to save GPU memory."""
        print(f"Node {self.id} moving to CPU for memory optimization")

        self._move_to_device('cpu')

    
@torch.no_grad()
def move_optimizer_state(optimizer,device):
    for pg in optimizer.param_groups:
        for p in pg["params"]:
            st = optimizer.state.get(p, None)
            if not st:
                continue
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    st[k] = v.to(device)  # rimpiazza il tensor sullo state
    # opzionale: libera cache allocator GPU
    torch.cuda.empty_cache()    

@torch.no_grad()
def offload_optimizer_state_to_cpu(optimizer):
    for pg in optimizer.param_groups:
        for p in pg["params"]:
            st = optimizer.state.get(p, None)
            if not st:
                continue
            for k, v in list(st.items()):
                if torch.is_tensor(v) and v.is_cuda:
                    st[k] = v.detach().to("cpu")  # rimpiazza il tensor sullo state
    # opzionale: libera cache allocator GPU
    torch.cuda.empty_cache()

@torch.no_grad()
def reload_optimizer_state_to_device(optimizer, device=None):
    for pg in optimizer.param_groups:
        for p in pg["params"]:
            target = p.device if device is None else torch.device(device)
            st = optimizer.state.get(p, None)
            if not st:
                continue
            for k, v in list(st.items()):
                if torch.is_tensor(v) and v.device != target:
                    st[k] = v.to(target)  # riallinea lo state al device dei pesi

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

    