import copy
import os
import wandb
from flcore.clients.clientgfe import clientGFE
from flcore.servers.serverrewind import FedRewind
from threading import Thread
import time
import numpy as np
from collections import defaultdict
import random
import itertools
from utils.data_utils import read_client_data
import concurrent.futures
import torch.futures as futures
import pandas as pd
from sklearn.metrics import confusion_matrix

import time
from itertools import cycle

import torch
from torchvision import transforms

from flcore.routing.scoredrouting import ScoredRouting
from flcore.routing.randomrouting import RandomRouting
from flcore.routing.staticrouting import StaticRouting

from transformers import ViTMAEConfig, ViTMAEModel, ViTConfig, ViTModel
from timm.models.vision_transformer import VisionTransformer

from flcore.trainmodel.vitfc import VITFC

from torchinfo import summary

from utils.node_metric import NodeMetric


class FedGFE(FedRewind):
    def __init__(self, args, times, pretext_tasks=None):
        super().__init__(args, times, create_clients=False)

        self.global_model = self.create_global_model(args.nodes_backbone_model)

        self.nodes_backbone_model = self.global_model
        self.model_backbone_load_checkpoint = args.model_backbone_load_checkpoint
        self.model_backbone_save_checkpoint = args.model_backbone_save_checkpoint

        self.ssl_rounds = args.ssl_rounds if args.ssl_rounds > 0 else args.global_rounds

        self.ssl_round = 0
        self.downstream_round = 0
        self.federation_grid_metrics = args.federation_grid_metrics

        self.model_aggregation_random = args.model_aggregation_random

        if pretext_tasks is not None:
            self.pretext_tasks = pretext_tasks
        else:
            self.pretext_tasks = list(filter(len,self.args.nodes_pretext_tasks.split(",")))

        self.nodes_datasets = self.args.dataset.split(",")
        self.nodes_downstream_tasks = self.args.nodes_downstream_tasks.split(",")
        self.nodes_training_sequence = self.args.nodes_training_sequence
        # if self.nodes_training_sequence == "sslfirst":
        #     self.global_rounds -= self.ssl_rounds if self.ssl_rounds < self.global_rounds else -20
        
        self.model_aggregation = self.args.model_aggregation
        self.model_backbone_save_checkpoint = args.model_backbone_save_checkpoint
        self.model_backbone_load_checkpoint = args.model_backbone_load_checkpoint
        self.model_backbone_checkpoint = args.model_backbone_checkpoint

        if self.model_backbone_load_checkpoint:
            self.load_checkpoint()
            if self.global_model != None:
                self.nodes_backbone_model = self.global_model

        self.clients = []

        self.model_aggregation_weighted = args.model_aggregation_weighted
        self.vit_config = None

        self.img_size = args.dataset_image_size
        if args.patch_size > 0:
            self.patch_size = args.patch_size
            self.patch_count = (self.img_size // self.patch_size) ** 2

        self.nodes_datasets_status = {}


        if self.routing_static:
            self.routing = StaticRouting(clients_count=self.num_clients, random=self.routing_random) 
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientGFE)

        for client in self.clients:
            print ( "\n*** Client %d dataset %s id %d" % (client.id, client.dataset, client.node_data.split_id) )
            client.node_data.stats_dump()


        if self.no_wandb == False:
            self.define_metrics()

        for client in self.clients:
            client.federation_clients = self.clients

        # routes = self.get_routes()
        # self.distribute_routes(routes)
        
        # print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.num_classes = args.num_classes
        self.statistics_dataframe = None
        #self.global_logits = [None for _ in range(args.num_classes)]

    def create_global_model(self, global_model_string):
        model = None
        if global_model_string == "hf_vit":
            self.vit_config = ViTConfig()
            self.vit_config.loss_type = "cross_entropy"
            self.vit_config.patch_size = self.args.patch_size
            self.vit_config.output_attentions = True
            self.vit_config.output_hidden_states = False
            self.vit_config.num_hidden_layers = self.args.num_hidden_layers
            if self.args.model_pretrain:
                model = ViTModel.from_pretrained(
                    "google/vit-base-patch16-224-in21k",
                    config=self.vit_config,
                    ignore_mismatched_sizes=True
                )
            else:
                model = ViTModel(self.vit_config, add_pooling_layer=False)
        elif global_model_string == "timm_vit":
            model = VisionTransformer(
                in_chans=3,
                num_classes=self.args.num_classes,
                embed_dim=self.args.embedding_size,       # d_model = 16
                depth=12,            # Number of transformer layers = 2
                num_heads=12,        # Number of attention heads
                mlp_ratio=4.0,      # MLP hidden dimension ratio
                patch_size=self.args.patch_size,         # Patch size
                class_token=True,  # Prepend class token to input
                global_pool='',  # Global pool type (one of 'cls', 'mean', 'attn')
            ) 
        return model

    def create_nodes_model ( self, model_string, global_model = None ):
        if global_model != None:
            return copy.deepcopy(global_model)

        model = None
        if model_string == "hf_vit":
            self.vit_config = ViTConfig()
            self.vit_config.loss_type = "cross_entropy"
            self.vit_config.patch_size = self.args.patch_size
            self.vit_config.output_attentions = True
            self.vit_config.output_hidden_states = True
            self.vit_config.num_hidden_layers = self.args.num_hidden_layers
            model = ViTModel(self.vit_config)
        elif model_string == "timm_vit":
            model = VisionTransformer(
        # img_size=img_size,
        # patch_size=patch_size,
        in_chans=3,
        num_classes=self.args.num_classes,
        embed_dim=self.args.embedding_size,       # d_model = 16
        depth=12,            # Number of transformer layers = 2
        num_heads=12,        # Number of attention heads
        mlp_ratio=4.0,      # MLP hidden dimension ratio
        patch_size=self.args.patch_size,         # Patch size
        class_token=True,  # Prepend class token to input
        global_pool='',  # Global pool type (one of 'cls', 'mean', 'attn')
        # qkv_bias=True,
        # representation_size=None,
        # distilled=False,
        # drop_rate=0.0,
        # attn_drop_rate=0.0,
        # drop_path_rate=0.0,
        # pretrained=args.model_pretrain,
    ) 
        return model

    def statistics_init(self):
        # init pandas dataframe for nodes and model statistics per round
        self.statistics_dataframe = pd.DataFrame(columns=['round', 'node', 'model', 'train_loss', 'train_acc', 'test_acc', 'test_auc', 'rewind_loss', 'rewind_acc', 'rewind_auc'])

    def statistics_update(self, round, node, model, train_loss, train_acc, test_acc, test_auc, rewind_loss, rewind_acc, rewind_auc):
        # update pandas dataframe with statistics
        self.statistics_dataframe = self.statistics_dataframe.append({'round': round, 'node': node, 'model': model, 'train_loss': train_loss, 'train_acc': train_acc, 'test_acc': test_acc, 'test_auc': test_auc, 'rewind_loss': rewind_loss, 'rewind_acc': rewind_acc, 'rewind_auc': rewind_auc}, ignore_index=True)
        
    def train_thread(self, client, device=-1, future = None, previous_node = None, training_task = "both"):

        if (device != -1):
            client.device = device
        thread = Thread(target=self.client_thread, args=(client, device, future, previous_node, training_task))
        thread.start()
        
        return thread

    def client_thread(self, client, device=-1, future = None, previous_node = None, training_task = "both"):

        if (device != -1):
            client.device = device
        target=client.train( rewind_train_node = previous_node, training_task = training_task )
        if future != None:
            future.set_result(-1)
  
    def train_clients(self, round, training_task = "both"):
        #self.selected_clients = self.select_clients()
        self.selected_clients = self.clients

        
        client_count = len(self.clients)
        client_index = 0

        while client_index < client_count:
            client = self.clients[client_index]
            client.device = self.device
            client.round = round
            client.ssl_round = self.ssl_round
            client.thread = None
            client.federation_size = len(self.clients)
            client.train( training_task = training_task )

            if training_task == "both" or training_task == "downstream":
                self.client_round_ending_hook( client )
            client_index += 1

    def train(self):
        training_task = "both" 
        if self.nodes_training_sequence == "sslfirst":
            training_task = "pretext"

        self.global_model.to(self.device)
        if self.model_aggregation == "fedavg":
            self.send_models()

        for i in range(1,self.global_rounds+1):
            self.round = i
            if self.round == int(self.global_rounds * 0.8):
                for client in self.clients:
                    client.optimizer = client.finetuning_optimizer 

            if self.round > self.ssl_rounds and self.nodes_training_sequence == "sslfirst":
                training_task = "downstream"

            if training_task == "both":
                if self.pretext_tasks != None and len(self.pretext_tasks) > 0:
                    self.ssl_round += 1
                self.data_log({"ssl_round": self.ssl_round})
                self.downstream_round += 1
                self.data_log({"downstream_round": self.downstream_round})
            elif training_task == "downstream":
                self.downstream_round += 1
                self.data_log({"downstream_round": self.downstream_round})
            elif training_task == "pretext":
                if self.pretext_tasks != None and len(self.pretext_tasks) > 0:
                    self.ssl_round += 1
                self.data_log({"ssl_round": self.ssl_round})

            if self.no_wandb == False:
                self.data_log({"round": self.round})

            s_t = time.time()
            self.selected_clients = self.clients
            # importante commentare questa riga per avere i client sempre ordinati
            #self.selected_clients = self.select_clients()


            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            if self.model_aggregation == "fedavg":
                self.send_models() 
            
            self.train_clients( i, training_task = training_task )
         
            print(self.uploaded_ids)
            # self.send_logits()

            self.Budget.append(time.time() - s_t)
            print('-'*50 + "Round time: ", self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

            if self.model_aggregation == "fedavg":
                self.receive_models()
                self.aggregate_parameters()
                # self.send_models()

            if self.model_backbone_save_checkpoint:
                self.save_checkpoint()

        # if self.nodes_training_sequence == "sslfirst":
        #     for round in range(1,self.global_rounds+1):
        #         # self.train( round, training_task = "downstream" )
        #         self.train( round )






        # print("Node routes\n")
        # for node in self.clients:
        #     print ( "Node %d -> %s <- %s" % (node.id, node.node_routes, node.rewind_previous_node) )
        # for client in self.clients:

        #     client.save_model()
        # print("\nBest accuracy.")
        
        # for test_client in self.clients:
        #     if not self.no_wandb:
        #         wandb.define_metric(f"node_acc_{test_client.id}", step_metric="node")
        #     for dest_client in self.clients:
        #         if ( test_client.id != dest_client.id):
        #             acc, test_num, auc, y_true, y_prob = test_client.test_metrics_other(dest_client)
        #             round_acc = acc/test_num
        #             self.data_log({f"node_acc_{test_client.id}": round_acc, "node": dest_client.id})
        #             print("Accuracy of nodes %d model on node %d: %02f" % (test_client.id, dest_client.id, round_acc ))
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        # print(max(self.rs_test_acc))
        # self.data_log({"best_acc": max(self.rs_test_acc)})
        # wandb.log({"best_acc": max(self.rs_test_acc)})
        # print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.evaluate()
        wandb.finish()

    def save_checkpoint(self):
        if self.save_checkpoint_enable != True and self.model_backbone_save_checkpoint != True:
            return
        if self.save_folder_name == None:
            self.save_folder_name = os.path.join(self.uuid)
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        
        filename = self.model_backbone_checkpoint
        torch.save(self.global_model, os.path.join(self.save_folder_name, filename))

    def load_checkpoint(self):
        if self.model_backbone_load_checkpoint != True:
            return
        
        filename = self.model_backbone_checkpoint
        if os.path.exists(filename):
            self.global_model = torch.load(filename)
        else:
            print(f"Checkpoint {filename} not found")



    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        # active_clients = random.sample(
        #     self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        active_clients = self.selected_clients

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threshold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model.backbone)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
    
    def send_models(self):
        assert (len(self.clients) > 0)


        self.global_model.to(self.device)
        for client in self.clients:
            start_time = time.time()
            global_model_id = hex(id(self.nodes_backbone_model))
            node_model_id = hex(id(client.model))
            node_model_backbone_id = hex(id(client.model.backbone))
            print ( f"Global model {global_model_id} node {client.id} {node_model_id} {node_model_backbone_id}" ) 
            client._move_to_gpu(self.device)
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
            client._move_to_cpu()

        self.global_model.to("cpu")
        
    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)
        # self.global_model = copy.deepcopy(self.uploaded_models[0])

        self.global_model.to(self.device)
        for p in self.global_model.parameters():
            p.data = torch.zeros_like(p)

        if self.model_aggregation_random:
            for param in self.global_model.parameters():
                param.data = torch.randn_like(param).to(param.device)
            return

        selected_clients_num = len(self.selected_clients)
        if self.model_aggregation_weighted: 
            weights = self.uploaded_weights
        else:
            weights = [1/selected_clients_num for _ in range(selected_clients_num)]
        client_models_num = len(self.uploaded_models)

        for client_model_id in range(client_models_num):
            # controlla se 
            for check_model_id in range(client_models_num):
                if client_model_id != check_model_id:
                    # check if the models' parameters are the same
                    equals = 0
                    not_equals = 0
                    for p1, p2 in zip(self.uploaded_models[client_model_id].parameters(), self.uploaded_models[check_model_id].parameters()):
                        if torch.equal(p1.data, p2.data):
                            equals += 1
                        else:
                            not_equals += 1
                    print ( f"Comparing client {client_model_id} with {check_model_id} equals {equals} not equals {not_equals}" )

        for w, client_model in zip(weights, self.uploaded_models):
           self.add_parameters(w, client_model)

        self.global_model.to("cpu")

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def client_round_ending_hook(self, client):

        round_loss, previous_loss = self.round_train_metrics( client )
        round_accuracy = self.round_test_metrics( client )

        print ( "\nNode %d orig loss %02f accuracy %02f" % ( client.id, round_loss, round_accuracy ) )

    def create_clients(self, clientObj, node_id, dataset, client_dataset_split_id):
        file_prefix = dataset 
        # if i != 2 and i != 3:
        # if i != 2:
            # continue 
        # train_data = read_client_data(self.dataset, i, is_train=True, prefix=file_prefix)
        # test_data = read_client_data(self.dataset, i, is_train=False, prefix=file_prefix)
        train_data = None
        test_data = None
        train_data_len = -1
        test_data_len = -1

        node_backbone = copy.deepcopy(self.nodes_backbone_model)
        node_model = VITFC(self.args, node_backbone, self.args.num_classes,patch_size=self.args.patch_size,debug_images=self.args.debug_pretext_images)
        node_model.id = node_id

        client = clientObj(self.args, 
                        node_id, 
                        dataset=dataset,
                        dataset_split_id = client_dataset_split_id,
                        train_samples=train_data_len, 
                        test_samples=test_data_len,
                        train_slow=-1, 
                        send_slow=-1,
                        rewind_epochs=self.rewind_epochs,
                        rewind_interval=self.rewind_interval,
                        rewind_ratio=self.rewind_ratio,
                        train_data=train_data,
                        test_data=test_data,
                        model=node_model,
                        dataset_limit=self.dataset_limit,
                        img_size=self.img_size,
                        patch_size=self.patch_size,
                        patch_count=self.patch_count
                        )
        # client.node_data.split_id = client_dataset_split_id
        return client
   
    def set_clients(self, clientObj):
        """Create clients based on args.nodes_tasks configuration if available, otherwise use legacy method."""

        # Check if we have nodes_tasks configuration
        if hasattr(self.args, 'nodes_tasks') and self.args.nodes_tasks is not None:
            print("Using nodes_tasks configuration for client creation")
            self._set_clients_from_nodes_tasks(clientObj)
        else:
            print("Using legacy dataset configuration for client creation")
            self._set_clients_legacy(clientObj)

    def _set_clients_from_nodes_tasks(self, clientObj):
        """Create clients using args.nodes_tasks configuration."""
        gpus = cycle(self.gpus)
        self.nodes_datasets_status = {}

        # Sort node IDs to ensure consistent ordering
        sorted_node_ids = sorted(self.args.nodes_tasks.keys(), key=lambda x: int(x))

        print(f"Creating {len(sorted_node_ids)} clients from nodes_tasks configuration:")

        for node_id in sorted_node_ids:
            node_config = self.args.nodes_tasks[node_id]
            client_id = int(node_id)

            # Extract configuration for this node
            task_type = node_config.get('task_type', 'classification')
            pretext_tasks = node_config.get('pretext_tasks', [])
            dataset = node_config.get('dataset', self.args.dataset)
            dataset_split = int(node_config.get('dataset_split', client_id))

            print(f"  Node {client_id}: {task_type} on {dataset}:{dataset_split} with pretext_tasks {pretext_tasks}")

            # Create client
            client = self.create_clients(clientObj, client_id, dataset, dataset_split)

            # Configure client with node-specific settings
            client.available_clients = np.arange(self.num_clients)
            client.pretext_tasks = pretext_tasks  # Node-specific pretext tasks
            client.downstream_task_name = task_type
            client.downstream_task.parameters_count()

            # Set metrics path
            metric_dataset_path = dataset.lower()
            client.metrics_path = f"node_{client.id}_{metric_dataset_path}_{dataset_split}"

            # Set transform
            client.transform = transforms.Compose([
                transforms.Resize([224, 224]),
            ])

            # Wandb logging
            if self.no_wandb == False:
                client.node_data.stats_wandb_log()

            client.data_log = self.data_log
            self.clients.append(client)

            # Update dataset status tracking
            if dataset not in self.nodes_datasets_status:
                self.nodes_datasets_status[dataset] = {
                    'created_nodes': [],
                    'used_splits': [],
                    'available_splits': []
                }

            self.nodes_datasets_status[dataset]['created_nodes'].append(client_id)
            self.nodes_datasets_status[dataset]['used_splits'].append(dataset_split)

        print(f"Successfully created {len(self.clients)} clients from nodes_tasks configuration")

    def _set_clients_legacy(self, clientObj):
        """Legacy client creation method using nodes_datasets format."""
        # n_strong = 0
        # n_weak = 0
        gpus = cycle(self.gpus)

        datasets = cycle(self.nodes_datasets)
        dataset_splits = {}
        for dataset_split_id, dataset in enumerate(self.nodes_datasets):
            splitted = dataset.split(":")
            selected_splits = []

            dataset_splits_count = 1
            dataset_client_count = 1

            if len(splitted) == 1:
                dataset_client_count = 1
                selected_splits = [0]
            if len(splitted) >= 2:
                dataset_client_count = int(splitted[1])
                selected_splits = [ s for s in range(dataset_client_count)]
                dataset_splits_count = int(splitted[1])
            if len(splitted) >= 3:
                selected_splits = splitted[2].split(";")
                selected_splits = [int(x) for x in selected_splits]
                dataset_client_count = len(selected_splits)
                dataset_splits_count = int(splitted[1])

            available_splits = [ s for s in range(dataset_splits_count) if s not in selected_splits ]

            dataset_name = splitted[0]
            dataset_splits[dataset_split_id] = {}
            dataset_splits[dataset_split_id]['dataset_name'] = dataset_name
            dataset_splits[dataset_split_id]['client_count'] = dataset_client_count
            dataset_splits[dataset_split_id]['selected_splits'] = selected_splits
            dataset_splits[dataset_split_id]['available_splits'] = available_splits

            if selected_splits == None:
                selected_splits = range(dataset_client_count)

        # downstream_tasks_names = self.nodes_downstream_tasks.split(":")

        # downstream_tasks_names = cycle(self.nodes_downstream_tasks)
        downstream_tasks_names = self.nodes_downstream_tasks

        for dataset_name_id, split_id in enumerate(dataset_splits.keys()):
            dataset_name = dataset_splits[split_id]['dataset_name']
            self.nodes_datasets_status[dataset_name] = {}
            self.nodes_datasets_status[dataset_name]['created_nodes'] = []
            self.nodes_datasets_status[dataset_name]['used_splits'] = []
            self.nodes_datasets_status[dataset_name]['available_splits'] = dataset_splits[split_id]['available_splits']
            dataset_name_clients_count = dataset_splits[dataset_name_id]['client_count']
            for node_downstream_tasks in downstream_tasks_names:
                downstream_tasks_names = node_downstream_tasks.split(",")
                client_dataset_id = 0
                for i in range(dataset_name_clients_count):
                    for downstream_task_name in downstream_tasks_names:
                        client_id = len(self.clients)
                        dataset_split_id = dataset_splits[split_id]['selected_splits'][i]
                        client = self.create_clients(clientObj, client_id, dataset_name, dataset_split_id)

                        # client.prefix=file_psefix
                        # client.device = "cuda:"+str(next(gpus))
                        client.available_clients = np.arange(self.num_clients)
                        client.pretext_tasks = self.pretext_tasks
                        client.downstream_task_name = downstream_task_name
                        client.downstream_task.parameters_count()
                        metric_dataset_oath = dataset_name.lower()
                        client.metrics_path = f"node_{client.id}_{metric_dataset_oath}_{client_dataset_id}"
                        # client.routing = RandomRouting(self.num_clients, id = i)
                        client.transform = transforms.Compose(
                        [
                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        transforms.Resize([224, 224]),
                        #  transforms.ToTensor()
                        ])

                        if self.no_wandb == False:
                            client.node_data.stats_wandb_log()

                        client.data_log = self.data_log
                        self.clients.append(client)
                        self.nodes_datasets_status[dataset_name]['created_nodes'].append(client_id)
                        self.nodes_datasets_status[dataset_name]['used_splits'].append(dataset_split_id)

                client_dataset_id += 1
        
        
        # for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):

        #     dataset = next(datasets)
        #     downstream_task_name = next(downstream_tasks_names)
        #     file_prefix = ""
        #     # if i != 2 and i != 3:
        #     # if i != 2:
        #         # continue 
        #     # train_data = read_client_data(self.dataset, i, is_train=True, prefix=file_prefix)
        #     # test_data = read_client_data(self.dataset, i, is_train=False, prefix=file_prefix)
        #     train_data = None
        #     test_data = None
        #     train_data_len = -1
        #     test_data_len = -1

        #     node_inner_model = copy.deepcopy(self.nodes_backbone_model)
        #     node_model = VITFC(node_inner_model, self.args.num_classes,patch_size=self.args.patch_size,debug_images=self.args.debug_pretext_images).to(self.args.device)

        #     client = clientObj(self.args, 
        #                     model_id=i, 
        #                     dataset=dataset,
        #                     train_samples=train_data_len, 
        #                     test_samples=test_data_len, 
        #                     train_slow=train_slow, 
        #                     send_slow=send_slow,
        #                     rewind_epochs=self.rewind_epochs,
        #                     rewind_interval=self.rewind_interval,
        #                     rewind_ratio=self.rewind_ratio,
        #                     train_data=train_data,
        #                     test_data=test_data,
        #                     model=node_model,
        #                     dataset_limit=self.dataset_limit,
        #                     img_size=self.img_size,
        #                     patch_size=self.patch_size,
        #                     patch_count=self.patch_count
        #                     )
        #     client.prefix=file_prefix
        #     client.device = "cuda:"+str(next(gpus))
        #     client.available_clients = np.arange(self.num_clients)
        #     client.pretext_tasks = self.pretext_tasks
        #     client.downstream_task_name = downstream_task_name
        #     client.downstream_task.parameters_count()


        #     client.metrics_path = f"node_{client.id}"
        #     # client.routing = RandomRouting(self.num_clients, id = i)
        #     client.transform = transforms.Compose(
        #     [
        #     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        #      transforms.Resize([224, 224]),
        #     #  transforms.ToTensor()
        #     ])

        #     # summary(client.model, input_size=(1, 3, 224, 224))
        #     # if self.args.routing_scored:
        #     #     client.routing = ScoredRouting(self.num_clients, id = i, average=self.routing_scored_average)
        #     # if self.args.routing_static:
        #     #     client.routing = self.routing
        #     #     client.routing.create_routes(self.clients)
        #     # else:
        #     #     client.routing = RandomRouting(self.num_clients, id = i)
            
        #     if self.no_wandb == False:
        #         client.node_data.stats_wandb_log()

        #     client.data_log = self.data_log
            
        #     # if is_strong:
        #     #     n_strong += 1
        #     # else:
        #     #     n_weak += 1
        #     self.clients.append(client)
        # if self.rewind_random:
        #     for client in self.clients:
        #         client.rewind_random_clients = self.clients
        #         client.rewind_random = True

    def define_metrics(self):
        wandb.define_metric(f"round")
        wandb.define_metric(f"ssl_round")
        wandb.define_metric(f"downstream_round")
  
        for client in self.clients:
            client.define_metrics()
            wandb.define_metric(f"test/node_{client.id}/acc", step_metric="round")
            wandb.define_metric(f"test/node_{client.id}/bal", step_metric="round")
            wandb.define_metric(f"train/node_{client.id}/round_train_loss", step_metric="round")
            wandb.define_metric(f"train/node_{client.id}/downstream_train_loss", step_metric="round")
            wandb.define_metric(f"test/node_{client.id}/downstream_test_acc", step_metric="round")
            wandb.define_metric(f"test/node_{client.id}/test_std", step_metric="round")
            wandb.define_metric(f"test/node_{client.id}/test_std_on_train", step_metric="round")
            wandb.define_metric(f"train/node_{client.id}/round_train_loss_{client.id}", step_metric="round")
            # for pretext_task in self.pretext_tasks:
            #     wandb.define_metric(f"train/node_{client.id}/pretext_train_loss_{pretext_task}", step_metric="round")
            #     wandb.define_metric(f"train/node_{client.id}/pretext_train_ds_loss_{pretext_task}", step_metric="round")
            #     wandb.define_metric(f"test/node_{client.id}/pretext_test_acc_{pretext_task}", step_metric="round")
            for other_client in self.clients:
                wandb.define_metric(f"test/node_{client.model.id}/round_test_acc_{client.model.id}_on_{other_client.model.id}", step_metric="round")
                wandb.define_metric(f"test/node_{client.model.id}/round_test_acc_on_train_{client.model.id}_on_{other_client.model.id}", step_metric="round")

    def evaluate(self):
        stats_test, stats_test_on_train = self.test_metrics(standalone = not self.federation_grid_metrics)
        stats_train = self.train_metrics()

        if stats_test == None or stats_train == None:
            return
        self.round_test_stats.insert(self.round, stats_test)
        self.round_test_on_train_stats.insert(self.round, stats_test_on_train)
        self.round_train_stats.insert(self.round, stats_train)

        # fed_test_acc = sum(stats_test[2])*1.0 / sum(stats_test[1])
        fed_test_acc = 0
        # Somma tutti gli elementi della lista che contiene dizionari di cui prendere quello 'aggregated'
        fed_test_metrics = {}
        for node_id in stats_test:
            node_test_acc = 0
            stats = stats_test[node_id]
            for tested_node_id in stats.keys():
                sample_count = stats[tested_node_id]['samples']
                for client_metric in stats[tested_node_id].defined_metrics:
                    if client_metric not in fed_test_metrics:
                        fed_test_metrics[client_metric] = []
                    fed_test_metrics[client_metric].append(stats[tested_node_id][client_metric]['mean'])
                    node_test_acc += stats[tested_node_id][client_metric]['mean']
                # node_test_acc = node_test_acc * 1.0 / len(stats[tested_node_id].defined_metrics)
            # fed_test_accuracies.append(node_test_acc)

        fed_test_metrics_on_train = {}
        for node_id in stats_test_on_train:
            node_test_acc = 0
            stats = stats_test_on_train[node_id]
            for tested_node_id in stats.keys():
                for client_metric in stats[tested_node_id].defined_metrics:
                    if client_metric not in fed_test_metrics_on_train:
                        fed_test_metrics_on_train[client_metric] = []
                    fed_test_metrics_on_train[client_metric].append(stats[tested_node_id][client_metric]['mean'])
                # node_test_acc =  node_test_acc * 1.0 / len(stats[tested_node_id].defined_metrics)
            # fed_test_accuracies_on_train.append(node_test_acc)

        fed_train_losses = [] 
        for train_stat in stats_train:
            node_train_metric = 0
            stats = stats_train[train_stat]
            node_train_metric = stats['loss']['mean']
            # node_train_metric = node_train_metric * 1.0 / len(stats)
            fed_train_losses.append(node_train_metric)

        # fed_test_acc = sum(fed_test_metrics) * 1.0 / len(fed_test_metrics)
        # fed_test_acc_on_train = sum(fed_test_metrics_on_train) * 1.0 / len(fed_test_metrics_on_train)
        fed_train_loss = sum(fed_train_losses) * 1.0 / len(fed_train_losses)

        worst_metrics = {}
        worst_metrics_node = {} 
        worst_metrics_taskid = {} 
        best_metrics = {}
        best_metrics_node = {}
        best_metrics_taskid = {}

        for node_id in stats_test:
            node_test_acc = 0
            stats = stats_test[node_id]
            for test_node_id in stats.keys():
                client_metric = stats[test_node_id]
                defined_metrics = client_metric.defined_metrics

                for defined_metric in defined_metrics:
                    if defined_metric not in worst_metrics or worst_metrics[defined_metric] >= client_metric[defined_metric]['min']:
                        worst_metrics[defined_metric] = client_metric[defined_metric]['min']
                        worst_metrics_taskid[defined_metric] = client_metric[defined_metric]['min_index']
                        worst_metrics_node[defined_metric] = node_id

                    if defined_metric not in best_metrics or best_metrics[defined_metric] <= client_metric[defined_metric]['max']:
                        best_metrics[defined_metric] = client_metric[defined_metric]['max']
                        best_metrics_taskid[defined_metric] = client_metric[defined_metric]['max_index']
                        best_metrics_node[defined_metric] = node_id

        fed_test_metrics_string = ""
        for client_metric in fed_test_metrics:
            mean_metric = sum(fed_test_metrics[client_metric]) * 1.0 / len(fed_test_metrics[client_metric])
            fed_test_metrics_string += f" {client_metric} {mean_metric:.3f} "

        fed_test_metrics_on_train_string = ""
        for client_metric in fed_test_metrics_on_train:
            mean_metric = sum(fed_test_metrics_on_train[client_metric]) * 1.0 / len(fed_test_metrics_on_train[client_metric])
            fed_test_metrics_on_train_string += f" {client_metric} {mean_metric:.3f} "

        print ( "**Federation train loss: %.2f (%s)" % ( fed_train_loss, fed_train_losses ) )
        print ( "**Federation test metrics: %s" % ( fed_test_metrics_string ) )
        print ( "**Federation test metrics on train: %s" % ( fed_test_metrics_on_train_string ) )
        for worst_metric in worst_metrics:
            print ( f"**Federation worst {worst_metric}: {worst_metrics[worst_metric]:.3f} task {worst_metrics_taskid[worst_metric]}")
        for best_metric in best_metrics:
            print ( f"**Federation best {best_metric}: {best_metrics[best_metric]:.3f} task {best_metrics_taskid[best_metric]}")

        # train_loss = 0
        # for node_index in stats_train:
        #     train_loss += sum(stats_train[node_index]) * 1.0 / len(stats_train[node_index])

        return

    def federation_metrics(self):
        num_samples = []
        tot_correct = []
        tot_auc = []

        y_t =[]
        y_p = []
        clients_test_stats = []
        clients_train_stats = []

        for c in self.clients:
            client_test_stats = []
            for t in self.clients:
                client_test_stat = []
                ct, ns, auc, y_true, y_prob = c.test_metrics(t)
                # ct, ns, auc = c.test_metrics()
                tot_correct.append(ct*1.0)
                tot_auc.append(auc*ns)
                num_samples.append(ns)
                y_t.append(y_true)
                y_p.append(y_prob)
                client_test_stat.append(ct*1.0/ns)
                client_test_stat.append(auc*ns)
                client_test_stat.append(ns)
                client_test_stat.append(y_true)
                client_test_stat.append(y_prob)
                client_test_stats.append(client_test_stat)
            clients_test_stats.insert(c.model.id, client_test_stats)
            # clients_stats.append(client_stats)

        for c in self.clients:
            client_train_stats = []
            for t in self.clients:
                client_train_stat = []
                ct, ns, auc, y_true, y_prob = c.test_metrics(t, on_train = True )
                # ct, ns, auc = c.test_metrics()
                tot_correct.append(ct*1.0)
                tot_auc.append(auc*ns)
                num_samples.append(ns)
                y_t.append(y_true)
                y_p.append(y_prob)
                client_train_stat.append(ct*1.0/ns)
                client_train_stat.append(auc*ns)
                client_train_stat.append(ns)
                client_train_stat.append(y_true)
                client_train_stat.append(y_prob)
                client_train_stats.append(client_train_stat)
            # train_clients_stats.append(client_stats)
            clients_train_stats.insert(c.model.id, client_train_stats)
        # test_accs = []
        # train_accs = []
        # train_losses = []
        # acc_std = []
        # acc_std_on_train = []
        # for client in self.clients:
        #     test_acc, test_num, auc, test_y_true, test_y_prob = client.test_metrics()
        #     test_accs.append(test_acc)
        #     train_loss, train_num = client.train_metrics()
        #     # train_accs.append(train_acc)
        #     train_losses.append(train_loss)
        #     acc_std.append(client.test_std)
        #     acc_std_on_train.append(client.test_std_on_train)

        acc_std_mean, acc_std_on_train_mean  = self.federation_metrics_std( clients_test_stats, clients_train_stats )
        self.data_log({"federation/acc_std": acc_std_mean, "round":self.round})
        self.data_log({"federation/acc_std_on_train": acc_std_on_train_mean, "round":self.round})
        print(f"Mean standard deviation of accuracies on test sets: {acc_std_mean} {acc_std_on_train_mean}")

        return clients_test_stats, clients_train_stats
    
    def federation_metrics_std(self, clients_test_stats = None, clients_train_stats = None ):
        test_accs = []
        train_accs = []
        train_losses = []
        acc_std = []
        acc_std_on_train = []
        if ( clients_test_stats == None ):
            for client in self.clients:
                test_acc, test_num, auc, test_y_true, test_y_prob = client.test_metrics()
                test_accs.append(test_acc)
                train_loss, train_num = client.train_metrics()
                # train_accs.append(train_acc)
                train_losses.append(train_loss)
                acc_std.append(client.test_std)
                acc_std_on_train.append(client.test_std_on_train)
        else:
            for client_stats in clients_test_stats:
                for test_client_stats in client_stats:
                    test_acc = test_client_stats[0]
                    test_accs.append(test_acc)
                    test_num = test_client_stats[2]
                    auc = test_client_stats[1]/test_num
                    # train_loss = test_client_stats[3]
                    # train_losses.append(train_loss)
                    y_true = test_client_stats[3]
                    y_prob = test_client_stats[4]
                    acc_std.append(self.round_test_metric_deviation( test_accs ))
            for client_stats in clients_train_stats:
                for test_client_stats in client_stats:
                    test_acc = test_client_stats[0]
                    test_accs.append(test_acc)
                    test_num = test_client_stats[2]
                    auc = test_client_stats[1]/test_num
                    # train_loss = test_client_stats[3]
                    # train_losses.append(train_loss)
                    y_true = test_client_stats[3]
                    y_prob = test_client_stats[4]
                    acc_std_on_train.append(self.round_test_metric_deviation( test_accs ))

        acc_std_mean = np.mean(acc_std)
        acc_std_on_train_mean = np.mean(acc_std_on_train)
        return acc_std_mean, acc_std_on_train_mean


    def round_rewind_train_metrics(self, client):
        previous_node = None
        previous_loss = -1
        previous_losses = []
        previous_losses_log = ""
        for rewind_node in client.rewind_previous_node:
            previous_loss, previous_train = rewind_node.train_metrics_other(client)
            previous_loss = previous_loss/previous_train
            previous_losses.append(previous_loss)
            previous_losses_log += f"{rewind_node.id}:{previous_loss:.2f} "
        return previous_losses, previous_losses_log

    def round_train_metrics(self, client):
        train_metric = NodeMetric( phase=NodeMetric.Phase.TRAIN )
        client._move_to_gpu(self.device)
        train_metric = client.train_metrics()
        if client.downstream_task != None:
            client.downstream_task.train_metrics_log( train_metric, round = self.round )
        client._move_to_cpu()


        # print ( "Getting train metrics on model %s loss %s " % ( hex(id(client.train_model)), hex(id(client.loss) ) ))

        node_loss_aggregated = train_metric['loss']['mean']
        previous_loss = -1
         
        round_loss = node_loss_aggregated
        self.data_log({f'train/node_{client.id}/round_train_loss': round_loss, "round": self.round})
        # self.data_log({f'train/node_loss_{client.train_model_id}': round_loss, "round": self.round})
        loss_dict = {client.train_model_id: round_loss}
        client.node_data_losses.append(loss_dict)
        # node_data_loss_string = [ f"{k}:{v:.2f}" for k,v in [ node_data_loss for node_data_loss in client.node_data_losses ] ]
        node_data_loss_string = ""
        for node_data_loss in client.node_data_losses:
            k,v = [n for n in node_data_loss.items()][0]
            node_data_loss_string += f"{k}:{v:.2f} "
        
        # print("** Round %d Trained node %d using model from %d on dataset %d loss %02f (%s)" % ( client.round, client.id, client.train_model_id, client.node_data.id, round_loss, node_data_loss_string ))
        if len(client.rewind_previous_node) and self.rewind_ratio:
            previous_losses, previous_losses_log = self.round_rewind_train_metrics(client)
            previous_loss = previous_losses[-1]
            client.rewind_previous_node_loss.append(previous_loss)
            print("** Previous rewind nodes' loss %s" % ( previous_losses_log ))
            self.data_log({f"rewind/rewind_loss_{client.model.id}": previous_loss, "round": self.round})
        # else:
        #     client.rewind_previous_node_loss.append(round_loss)

        return round_loss, previous_loss
    

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        nodes_train_metrics = {}

        for client_index,c in enumerate(self.clients):
            if c.node_data.test_data == None:
                print ( "Client %d test data is None" % c.id)
                continue
            c._move_to_gpu(self.device)
            # node_train_metrics = NodeMetric( phase=NodeMetric.Phase.TRAIN )
            node_train_metrics = c.train_metrics()
            nodes_train_metrics[c.id] = node_train_metrics
            c._move_to_cpu()

        return nodes_train_metrics
    
    def test_metrics(self,standalone = False):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        test_num_samples = []
        test_tot_correct = []
        test_tot_auc = []

        test_y_t =[]
        test_y_p = []

        train_num_samples = []
        train_tot_correct = []
        train_tot_auc = []

        train_y_t =[]
        train_y_p = []
        test_clients_stats = self.num_clients * [None]
        train_clients_stats = self.num_clients * [None]

        tested_clients = 0

        # metric = ConfusionMatrix(num_classes=10)
        # metric.attach(default_evaluator, 'cm')
        test_clients_stats = {}
        test_clients_stats_on_train = {}
        train_clients_stats = {}

        for client_index,c in enumerate(self.clients):
            if c.node_data.test_data == None:
                print ( "Client %d test data is None" % c.id)
                continue
            tested_clients += 1
            test_clients_stats[c.id] = {}
            test_clients_stats_on_train[c.id] = {}
            train_clients_stats[c.id] = []
            if ( client_index < len(self.clients) - 1):
                self.clients[client_index+1].node_data.load_test_data(self.args.batch_size)
            test_clients = self.clients
            if standalone == True:
                test_clients = [c]

            for t in test_clients:
                test_metrics = None
                test_metrics_on_train = None


                if c.node_data.test_data == None:
                    print ( "Client %d test data is None" % c.id)
                    continue
                c.pretext_train = False
                node_test_metrics = c.test_metrics(t)
                # print("test_metrics:", test_metrics)
                node_test_metrics_on_train = c.test_metrics(t, on_train=True)
                # print("test_metrics_on_train:", test_metrics_on_train)
                test_clients_stats[t.id][c.id] = node_test_metrics
                test_clients_stats_on_train[t.id][c.id] = node_test_metrics_on_train

            if ( client_index > 0 and self.reduce_memory_footprint == True):
                c.node_data.unload_test_data()
            

            
            # if not self.no_wandb:
            #     wandb.log({f'test_acc_{c.id}': ct*1.0/ns})
        if tested_clients == 0:
            return None
        # ids = [c.id for c in self.clients]
        for node_id in test_clients_stats.keys():
            metric_types = test_clients_stats[node_id][node_id].defined_metrics
            # test_clients_stats[node_id][node_id]['aggregated'].keys()
            for metric_type in metric_types:
                samples = test_clients_stats[node_id][node_id]['samples']
                node_test_acc = test_clients_stats[node_id][node_id][metric_type]['mean']
                node_test_acc_on_train = test_clients_stats_on_train[node_id][node_id][metric_type]['mean']
                # print ( f"Client {node_id} test metrics: {node_test_acc:.2f} {node_test_acc_on_train:.2f}" )
        return test_clients_stats, test_clients_stats_on_train
    
    def round_test_metric_deviation (self, accuracies):
        # Calcola la deviazione standard tra le loss dei nodi e dei modelli
        standard_deviation = np.std(accuracies)

        return standard_deviation
    def round_test_metrics(self, client):
        # if len(client.rewind_previous_node) > 0:
        #     previous_node = client.rewind_previous_node[-1]
        #     previous_accuracy, previous_test = client.test_metrics_other(previous_node)
        #     previous_accuracy = previous_accuracy/previous_test

        if client.downstream_task == None:
            return 0

        node_test_metrics = client.test_metrics()
        client.downstream_task.test_metrics_log( metrics = node_test_metrics, round = self.round )
        # test_num = node_test_metrics['samples']
        # acc, test_num, auc, y_true, y_prob = client.test_metrics()

        if node_test_metrics['samples'] == 0:
            print(f"Node {client.id} has no test samples")
            return 0
        

        client_round_metric = {}
        
        for metric in node_test_metrics.defined_metrics:
            client_round_metric[metric] = []
        
        for metric in node_test_metrics.defined_metrics:
            client_round_metric[metric].append(node_test_metrics[metric]['mean'])
        # if not self.no_wandb:
        #     wandb.log({f'test/model_{client.id}/round_test_acc_{client.id}': client_round_acc, "round": self.round})

        node_chosen_metric = node_test_metrics.defined_metrics[0]

        accuracies = self.round_test_metrics_nodes(client, ignore_last=False)
        metric_list = []
        accuracies_list = []
        for k, v in accuracies.items():
            accuracies_list.append(v[node_chosen_metric]['mean'])

        acc_std = np.std(accuracies_list)
        client.test_std_on_train.append(acc_std)

        node_test_metrics = NodeMetric( phase=NodeMetric.Phase.TEST )
        node_test_metrics_on_train = NodeMetric( phase=NodeMetric.Phase.TEST )
        test_metrics = client.test_metrics( metrics=node_test_metrics )
        test_metrics_on_train = client.test_metrics( on_train = True, metrics=node_test_metrics_on_train )

        print( "Node %d test metrics: %s on train: %s" % ( client.id, test_metrics, test_metrics_on_train ) )

        accuracy_on_test = node_test_metrics[node_chosen_metric]['mean']

        accuracies_on_train = self.round_test_metrics_nodes(client, on_train = True, ignore_last=False)
        # accuracies_on_train = self.round_test_metrics_nodes(client, on_train = True, ignore_last=False)

        metric_list = []
        accuracies_on_train_list = []
        for k, v in accuracies_on_train.items():
            accuracies_on_train_list.append(v[node_chosen_metric]['mean'])

        acc_std_on_train = np.std(accuracies_on_train_list)
        client.test_std_on_train.append(acc_std_on_train)
       
        # print("\n** Round %d Trained node %d model %d accuracy %02f other %s" % (self.round, client.id, client.model.id, client_round_acc, accuracies ))
        # print("** Round %d Accuracies on test sets %.02f %s" % ( self.round, client_round_acc, accuracies ))
        # print("** Round %d Accuracies on train sets %.02f %s" % ( self.round, accuracy_on_train, accuracies_on_train ))
        # print("** Round %d std on test %.02f on train %.02f" % ( self.round, acc_std, acc_std_on_train ))
        if not self.no_wandb:    
            wandb.log({f'test/node_{client.id}/test_std': acc_std, "round": self.round})
            wandb.log({f'test/node_{client.id}/test_std_on_train': acc_std_on_train, "round": self.round})
            for metric, path in client.defined_test_metrics.items():
                if metric in test_metrics and test_metrics[metric] is not None:
                    wandb.log({f'test/{path}': test_metrics[metric]['mean'], "round": self.round}) 

        # standard_deviation = self.round_test_metric_deviation(client)
        # print(f"Standard deviation of accuracies for client {client.id}: {standard_deviation}")
        return accuracy_on_test
    
    def round_test_metrics_nodes (self, client, ignore_last = True, on_train = False):
        accuracies = []
        round_node_metrics = {}
        test_clients = self.clients
        if self.federation_grid_metrics == False:
            test_clients = [client]
            
        for test_client in test_clients:
            # if ( test_client.node_data.id != client.node_data.id or ignore_last == False ):
            node_metrics = NodeMetric( phase=NodeMetric.Phase.TEST )
            test_metrics = client.test_metrics(test_client, on_train = on_train, metrics=node_metrics)
            round_node_metrics[test_client.id] = node_metrics

            test_num = node_metrics['samples']
            if test_num == 0:
                continue
            
            node_round_metric = {}
            for node_metric in node_metrics.defined_metrics:
                node_round_metric[node_metric] = []
            for node_metric in node_metrics.defined_metrics:
                node_round_metric[node_metric].append(node_metrics[node_metric]['mean'])
            node_round_chosen_metric = node_metrics.defined_metrics[0]
            round_acc = test_metrics[node_round_chosen_metric]['mean']

            other_accuracy = { 'node_dataset': test_client.id, 'accuracy': round_acc }
            accuracies.append(other_accuracy)
            # print("Node's model %d accuracy dataset %d: %02f" % (client.id, test_client.id, round_acc )) 
            if  not self.no_wandb:
                if on_train == False:
                    wandb.log({f'test/node_{client.id}/round_test_acc_{client.id}_on_{test_client.id}': round_acc, 'round': self.round } )
                else:
                    wandb.log({f'test/node_{client.id}/round_test_acc_on_train_{client.id}_on_{test_client.id}': round_acc, 'round': self.round } )
            # if previous_node != None:
            #     client.rewind_previous_node_loss.append(previous_loss)
            #     print("Previous node %d loss %02f" % ( previous_node.id, previous_loss))
        return round_node_metrics

            
# ---------------- From FedER -----------------------------
def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def get_routes(n_nodes, clients = None):
    if clients is not None:
        idxs = [client.id for client in clients]
    else:
        idxs = [i for i in range(n_nodes)]
    random.shuffle(idxs)
    routes = {x[0]:x[1] for x in pairwise(idxs)}
    last_route = (idxs[-1],idxs[0])
    routes[last_route[0]] = last_route[1]
    return routes # k: sender, v: receiver

#---------------------------------------------