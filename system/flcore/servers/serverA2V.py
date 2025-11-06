import copy
import os
import sys
import wandb
from flcore.clients.clientA2V import clientA2V
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
from collections import OrderedDict
from collections.abc import Mapping



from torch.optim import AdamW

import time
from itertools import cycle

import torch
from torchvision import transforms

from flcore.routing.scoredrouting import ScoredRouting
from flcore.routing.randomrouting import RandomRouting
from flcore.routing.staticrouting import StaticRouting
from flcore.trainmodel.downstreamsinestesiaadapters import DownstreamSinestesiaAdapters
from flcore.trainmodel.downstreamsinestesia import DownstreamSinestesia
from flcore.trainmodel.Audio2Visual_NoData.src.models.sinestesia import SinestesiaWithClassifier

from torchinfo import summary

from utils.node_metric import NodeMetric
from utils.ballooning import GPUMemoryBalloon
from datautils.dataset_vegas import VEGASDataset
from datautils.dataset_esc50 import ESC50Dataset

import logging, sys

# Import Audio2Visual specific modules
# sys.path.append('/home/lpala/fedgfe/system/flcore/trainmodel/Audio2Visual_NoData')
from flcore.trainmodel.Audio2Visual_NoData.src.models.audio2image import Audio2Image, SDImageModel, ImageDiffusion

logger = logging.getLogger(__name__)

class FedA2V(FedRewind):
    def __init__(self, args, times, pretext_tasks=None):
        super().__init__(args, times, create_clients=False)

        self.global_model = None
        self.generator_model = None
        self.encoder_audio = None
        self.encoder_image = None
        self.encoder_text = None
        self.config = args.json_config if hasattr(args, 'json_config') else None

        self.balooning_size = 0
        self.bolooning_reserve_mb = 0.1
        if self.config.experiment.use_balooning:
            self.balooning_size = 38*1024*1024*1024
        self.baloon = GPUMemoryBalloon(gpu_id=0, chunk_size_mb=256,reserve_mb=self.bolooning_reserve_mb)
        self.baloon.allocate_memory(self.balooning_size)

        self.device1 = self.device if torch.cuda.is_available() else 'cpu'
        self.device2 = self.device if torch.cuda.is_available() else 'cpu'
        self.device3 = self.device if torch.cuda.is_available() else 'cpu'

        # Audio2Visual specific configuration
        self.diffusion_type = getattr(self.config.federation, 'diffusion_type', 'sd')
        self.use_act_loss = getattr(args, 'use_act_loss', True)
        self.audio_model_name = getattr(args, 'audio_model_name', "MIT/ast-finetuned-audioset-10-10-0.4593")
        self.img_pipe_name = getattr(args, 'img_pipe_name', "runwayml/stable-diffusion-v1-5")
        self.img_lcm_lora_id = getattr(args, 'img_lcm_lora_id', "latent-consistency/lcm-lora-sdv1-5")


        self.global_model_train = getattr(self.config.feda2v, 'global_model_train', False)
        self.global_model_train_epochs = getattr(self.config.feda2v, 'global_model_train_epochs', 1)
        self.generate_nodes_images_frequency = getattr(self.config.feda2v, 'generate_nodes_images_frequency', 0)
        self.generate_global_images_frequency = getattr(self.config.feda2v, 'generate_global_images_frequency', 0)
        self.generate_global_images_average_text_embeddings = getattr(self.config.feda2v, 'generate_global_images_average_text_embeddings', False)
        self.images_output_dir = getattr(self.config.feda2v, 'images_output_dir', 'output_images')
        self.generate_images_frequency = self.generate_nodes_images_frequency

        self.adapter_aggregation_mode = self.config.feda2v.adapter_aggregation_mode

        # Create Encoders models Audio2Visual model
        self.create_global_model(args)

        self.nodes_backbone_model = self.global_model

        self.federation_grid_metrics = args.federation_grid_metrics

        self.model_aggregation_random = args.model_aggregation_random
        self.model_aggregation = self.args.model_aggregation
        self.aggregation_method = args.model_aggregation
        self.model_backbone_save_checkpoint = args.model_backbone_save_checkpoint
        self.model_backbone_load_checkpoint = args.model_backbone_load_checkpoint
        self.model_backbone_checkpoint = args.model_backbone_checkpoint

        self.global_output_means = {}
        self.global_output_means_per_class = {}

        self.use_saved_audio_embeddings = getattr(args, 'use_saved_audio_embeddings', False)
        self.audio_embedding_file_name = getattr(args.json_config.feda2v, 'audio_embedding_file_name', None)
        self.store_audio_embeddings = getattr(args.json_config.feda2v, 'store_audio_embeddings', False)


        if self.model_backbone_load_checkpoint:
            self.load_checkpoint()
            if self.global_model != None:
                self.nodes_backbone_model = self.global_model

        self.clients = []

        self.model_aggregation_weighted = args.model_aggregation_weighted

        if self.routing_static:
            self.routing = StaticRouting(clients_count=self.num_clients, random=self.routing_random)

        # Select slow clients and create clients
        self.set_clients(clientA2V)

        self.generate_global_images_average_text_embeddings_from_nodes()

        for client in self.clients:
            print(f"\n*** Client {client.id} dataset {client.dataset}")
            client.node_data.stats_dump()

        if self.no_wandb == False:
            self.define_metrics()

        for client in self.clients:
            client.federation_clients = self.clients

        self.nodes_adapters = {}
        self.nodes_adapters_modules = {}

        logger.info("Finished creating Audio2Visual server and clients.")

        self.Budget = []
        self.num_classes = args.num_classes
        self.statistics_dataframe = None

       
                                                 
    def create_global_model(self, args):

        self.global_model = DownstreamSinestesiaAdapters( args, diffusion_type=self.diffusion_type )


        self.encoder_audio = None
        self.encoder_image = None
        self.encoder_text = None

        if self.diffusion_type == 'flux':
            img_pipe_name = 'MIT/ast-finetuned-audioset-10-10-0.4593'
        elif self.diffusion_type == 'sd':
            img_pipe_name = 'runwayml/stable-diffusion-v1-5'

        if self.generate_global_images_frequency > 0 or self.generate_nodes_images_frequency:
            self.global_model.enable_diffusion = True
            self.global_model.image_generation_frequency = self.generate_global_images_frequency
            self.global_model.start_diffusion()

            # self.generator_model = ImageDiffusion(
            #     audio_model_name=self.audio_model_name,
            #     img_pipe_name=img_pipe_name,
            #     img_lcm_lora_id=self.img_lcm_lora_id,
            #     diffusion_type=self.diffusion_type,
            #     use_act_loss=self.use_act_loss,
            #     device1=self.device1,
            #     device2=self.device2,
            #     mode='train_nodata'
            # )

        self.global_adapters = self.global_model.adapters
        # self.global_projections = self.global_model.projections
        self.global_adapters_modules = self.global_model.adapters_modules

        self.global_optimizers = self.create_global_model_optimizers()

        return self.global_model, self.generator_model
    
    # def create_global_modules_list(self):
    #     global_module_list = {}
    #     for module_name in self.global_adapters.keys():
    #         module_list = torch.nn.ModuleList()
    #         module_list.add_module ( module_name+"_adapter", self.global_adapters[module_name] )
    #         module_list.add_module ( module_name+"_projection", self.global_projections[module_name] )
    #         global_module_list[module_name] = module_list
    #     return global_module_list

    def create_global_model_optimizers(self):
        optimizers = {}
        for module_name, module in self.global_adapters.items():
            if self.config.training.optimizer == "AdamW":
                optimizers[module_name] = AdamW(params=module.parameters(), lr=self.config.training.learning_rate)

        return optimizers

    def create_nodes_model(self, model_string=None, global_model=None):
        """Create a node-specific Audio2Visual model."""
        if global_model != None:
            return copy.deepcopy(global_model)

        return self.create_global_model(self.args)

    def statistics_init(self):
        """Initialize pandas dataframe for nodes and model statistics per round."""
        self.statistics_dataframe = pd.DataFrame(
            columns=['round', 'node', 'model', 'train_loss', 'test_loss', 'generation_quality']
        )

    def statistics_update(self, round, node, model, train_loss, test_loss, generation_quality):
        """Update pandas dataframe with statistics."""
        self.statistics_dataframe = self.statistics_dataframe.append({
            'round': round,
            'node': node,
            'model': model,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'generation_quality': generation_quality
        }, ignore_index=True)

    def train_thread(self, client, device=-1, future=None, previous_node=None, training_task="both"):
        if device != -1:
            client.device = device
        thread = Thread(target=self.client_thread, args=(client, device, future, previous_node, training_task))
        thread.start()
        return thread

    def client_thread(self, client, device=-1, future=None, previous_node=None, training_task="both"):
        if device != -1:
            client.device = device
        target = client.train(rewind_train_node=previous_node, training_task=training_task)
        if future != None:
            future.set_result(-1)

    def train_nodes(self, round, training_task="both"):
        self.selected_clients = self.clients

        client_count = len(self.clients)
        node_index = 0

        while node_index < client_count:
            node = self.clients[node_index]
            self.client_round_starting_hook(node)
            node.device = self.device
            node.round = round
            node.thread = None
            node.federation_size = len(self.clients)
            node.train()

            self.client_round_ending_hook(node)
            node_index += 1

    def train(self):
        """Main training loop for Audio2Visual federated learning."""
        training_task = "both"

        if self.global_model != None:
            self.global_model.to(self.device)

        for i in range(1, self.global_rounds + 1):
            self.round = i

            if self.no_wandb == False:
                self.data_log({"round": self.round})

            s_t = time.time()
            self.selected_clients = self.clients

            if i % self.eval_gap == 0 or True:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate Audio2Visual models")
                self.evaluate()

            if self.aggregation_method != 'none' or self.adapter_aggregation_mode != 'none':
                self.send_models()

            self.train_nodes(i, training_task=training_task)

            if self.store_audio_embeddings and self.audio_embedding_file_name is not None and self.round == 1:
                self.save_audio_embeddings(file_name=self.audio_embedding_file_name)
                print(f"Saved audio embeddings to {self.audio_embedding_file_name}")

            print(self.uploaded_ids)

            self.Budget.append(time.time() - s_t)
            print('-' * 50 + "Round time: ", self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

            if self.model_aggregation != "none":
                self.receive_models()
                self.aggregate_parameters()

            if self.model_backbone_save_checkpoint:
                self.save_checkpoint()

            self.round_ending_hook()

        self.save_results()
        self.evaluate()
        wandb.finish()

    def round_ending_hook(self):
        # self.baloon.deflate()
        if self.generate_nodes_images_frequency > 0 and self.round % self.generate_nodes_images_frequency == 0:
            self.global_model.diffusion_model.to(self.global_model.diffusion_model_device)
            
            for client in self.clients:
                 self.generate_images(client)

            self.global_model.diffusion_model.to("cpu")

        if self.global_model_train and (self.config.feda2v.global_model_train_from_nodes_audio_embeddings or self.config.feda2v.global_model_train_from_nodes_adapters):
            self._move_to_gpu(self.device)

            if self.config.feda2v.global_model_train_from_nodes_audio_embeddings:
                loss = self.global_model_train_from_nodes_text_embeddings()
            if self.config.feda2v.global_model_train_from_nodes_adapters:
                loss = self.global_model_train_from_nodes_adapters_output()

            self._move_to_cpu()
            print(f"\nGlobal Audio2Visual model trained from nodes embeddings with loss {loss:.4f}")
        
        # self.baloon.inflate()

        if self.generate_global_images_frequency > 0 and self.round % self.generate_global_images_frequency == 0:
            all_audio_embeddings = {}

            for client in self.clients:
                if hasattr(client, 'audio_embedding_store') and client.audio_embedding_store is not None:
                    all_audio_embeddings.update(client.audio_embedding_store)

    def generate_global_images_average_text_embeddings_from_nodes(self):
        nodes_classes_text_embeddings = {}

        for node in self.clients:
            node_dataset = node.node_data.train_dataset.dataset
            node_classes = node_dataset.active_classes
            node_text_embeddings = node_dataset.text_embs
            print ( f"Node {node.id} classes {node_classes}")
            for node_class in node_classes:
                nodes_classes_text_embeddings.update({node_class: node_text_embeddings[node_class]})

        print ( f"Nodes text embeddings {nodes_classes_text_embeddings.keys()}")
        return
    
    def global_model_train_from_nodes_text_embeddings(self):
        self.global_model.to(self.device)

        loss = 0.0
        nodes_class_losses = {}
        for node_id, per_class_outputs in self.nodes_per_class_outputs.items():
            node = self.clients[node_id]
            node_dataset = node.node_data.train_dataset if not isinstance(node.node_data.train_dataset, torch.utils.data.Subset) else node.node_data.train_dataset.dataset


            print (f"Training global model using audio embedding from from node {node_id}")
            node_loss = 0.0
            for step in range(self.global_model_train_epochs):

                for class_output_name, per_class_output in per_class_outputs.items():
                    if class_output_name not in nodes_class_losses:
                        nodes_class_losses[class_output_name] = []

                    text_embedding = node_dataset.get_text_embeddings(class_output_name)[self.diffusion_type]
                    prompt_embeds = text_embedding['prompt_embeds'] if 'prompt_embeds' in text_embedding else None
                    pooled_prompt_embeds = text_embedding['pooled_prompt_embeds'] if 'pooled_prompt_embeds' in text_embedding else None

                    audio_embedding = torch.stack(per_class_output)

                    output = self.global_model( per_class_output, img_target_prompt_embeds=prompt_embeds, img_target_pooled_prompt_embeds=pooled_prompt_embeds, audio_embedding = audio_embedding)

                    for module_name, optimizer in self.global_optimizers.items():
                        optimizer.zero_grad()

                    losses = output['text_loss']


                    total_loss = torch.tensor(0.0)
                    for loss in losses:
                        total_loss = total_loss.to(loss.device)
                        total_loss += loss

                    total_loss.backward()
                    nodes_class_losses[class_output_name].append(total_loss.item())

                    for module_name, optimizer in self.global_optimizers.items():
                        optimizer.step()

    def global_model_train_from_nodes_adapters_output(self):
        self.global_model.to(self.device)

        loss = 0.0
        nodes_class_losses = {}
        for node_id, per_class_adapters_outputs in self.nodes_per_class_adapters_outputs_means.items():
            node = self.clients[node_id]
            node_dataset = node.node_data.train_dataset if not isinstance(node.node_data.train_dataset, torch.utils.data.Subset) else node.node_data.train_dataset.dataset

            print (f"Training global model using adapters outputs from node {node_id}")
            node_loss = 0.0
            for step in range(self.global_model_train_epochs):

                for class_output_name, per_class_output in per_class_adapters_outputs.items():
                    if class_output_name not in nodes_class_losses:
                        nodes_class_losses[class_output_name] = []

                    # for adapter_name, adapter_module in self.global_adapters_modules:

                    output = self.global_model( per_class_output, img_target_prompt_embeds=prompt_embeds, img_target_pooled_prompt_embeds=pooled_prompt_embeds, audio_embedding = audio_embedding)

                    for module_name, optimizer in self.global_optimizers.items():
                        optimizer.zero_grad()

                    losses = output['text_loss']


                    total_loss = torch.tensor(0.0)
                    for loss in losses:
                        total_loss = total_loss.to(loss.device)
                        total_loss += loss

                    total_loss.backward()
                    nodes_class_losses[class_output_name].append(total_loss.item())

                    for module_name, optimizer in self.global_optimizers.items():
                        optimizer.step()

        for class_name, class_losses in nodes_class_losses.items():
            class_loss = sum(class_losses)/len(class_losses)
            print ( f"Loss for class {class_name} {class_loss:.2f}")
                


        return loss
    
    def global_model_step_per_node (self):
        pass


    def global_model_create_images(self, audio_embeddings, num_images=1):
        """Generate images using the global Audio2Visual model."""
        self.global_model.diffusion_model.to(self.global_model.diffusion_model_device)

        prompt_embeds = audio_embeddings['t5'].to(self.global_model.diffusion_model_device)
        pooled_prompt_embeds = audio_embeddings['clip'].to(self.global_model.diffusion_model_device)

        imgs = self.global_model.diffusion_model(
                                        prompt_embeds= prompt_embeds,
                                        pooled_prompt_embeds=pooled_prompt_embeds,
                                        num_inference_steps=1,
                                        output_type="pt",
                                        ).images
        
        self.global_model.diffusion_model.to("cpu")

        return imgs

    def save_audio_embeddings(self, file_name="audio_embeddings.pt"):
        """Save audio embeddings from clients to a file."""
        all_audio_embeddings = {}

        for client in self.clients:
            if hasattr(client, 'audio_embedding_store') and client.audio_embedding_store is not None:
                all_audio_embeddings.update(client.audio_embedding_store)

        torch.save(all_audio_embeddings, file_name)
        
    def save_checkpoint(self):
        """Save Audio2Visual model checkpoint."""
        if self.save_checkpoint_enable != True and self.model_backbone_save_checkpoint != True:
            return
        if self.save_folder_name == None:
            self.save_folder_name = os.path.join(self.uuid)
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)

        filename = self.model_backbone_checkpoint
        torch.save(self.global_model, os.path.join(self.save_folder_name, filename))

    def load_checkpoint(self):
        """Load Audio2Visual model checkpoint."""
        if self.model_backbone_load_checkpoint != True:
            return

        filename = self.model_backbone_checkpoint
        if os.path.exists(filename):
            self.global_model = torch.load(filename)
        else:
            print(f"Checkpoint {filename} not found")

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        self.gathered_nodes = 0

        if self.adapter_aggregation_mode != 'none':
            self.receive_nodes_adapters()

        if self.aggregation_method == 'per_class_average':
            received_model = self.receive_models_per_class_average()
        else:
            print(f"Model aggregation method {self.aggregation_method} not recognized.")
            return
        
        self.gathered_nodes = received_model
        return received_model
    
    def receive_nodes_adapters(self):

        active_clients = self.selected_clients
        gathered_nodes = 0

        for node in active_clients:
            self.nodes_adapters[node.id] = node.adapters
            self.nodes_adapters_modules[node.id] = node.adapters_modules

            gathered_nodes += 1

        return gathered_nodes


        
    def receive_models_per_class_average(self):

        active_clients = self.selected_clients

        self.nodes_per_class_outputs = {}
        self.nodes_per_class_outputs_means = {}
        self.nodes_per_class_adapters_outputs = {}
        self.nodes_per_class_adapters_outputs_means = {}
        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        gathered_nodes = 0
        for node in active_clients:
            try:
                node_time_cost = node.train_time_cost['total_cost'] / node.train_time_cost['num_rounds'] + \
                                   node.send_time_cost['total_cost'] / node.send_time_cost['num_rounds']
            except ZeroDivisionError:
                node_time_cost = 0
            if node_time_cost <= self.time_threshold:
                tot_samples += node.train_samples
                self.nodes_per_class_outputs[node.id] = node.per_class_outputs
                self.nodes_per_class_outputs_means[node.id] = node.per_class_outputs_mean
                self.nodes_per_class_adapters_outputs[node.id] = node.training_adapter_outputs_all
                self.nodes_per_class_adapters_outputs_means[node.id] = node.training_adapter_outputs_mean

            gathered_nodes += 1

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

        return gathered_nodes

    def send_models(self):
        """Send global Audio2Visual model to clients."""
        assert (len(self.clients) > 0)

        for node in self.clients:
            start_time = time.time()
            node.update_local_adapters(self.global_adapters)

            node.send_time_cost['num_rounds'] += 1
            node.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
            node._move_to_cpu()

            if self.model_aggregation == "per_class_average":
                self.send_models_per_class_average(node)

            if self.adapter_aggregation_mode == 'avg':
                self.send_adapters(node)
        return
    
    def send_adapters(self, node):
        self.global_model.to(self.device)
        node.update_local_adapters(self.global_adapters)

        self.global_model.to("cpu")

    def send_models_per_class_average(self,node):
        return
        self.global_model.to(self.device)
        for client in self.clients:
            start_time = time.time()
            client.update_local_adapters(self.global_adapters)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
            client._move_to_cpu()

        self.global_model.to("cpu")

    def aggregate_parameters(self):
        """Aggregate Audio2Visual model parameters from clients."""
        assert (self.gathered_nodes > 0)

        if self.aggregation_method == 'per_class_average':
            self.aggregate_audio_encoder_parameters_per_class_avarage()

        if self.adapter_aggregation_mode == 'avg':
            self.aggregate_adapters_parameters_fedavg()
        else:
            print(f"Model aggregation method {self.aggregation_method} not recognized.")
            return
        
    def aggregate_audio_encoder_parameters_per_class_avarage(self):

        # Aggregate per-class outputs means collected from nodes
        # self.nodes_per_class_outputs_means is expected to be a dict: node_id -> {class: value}
        per_class_accum = {}  # class -> summed tensor
        per_class_counts = {}  # class -> count

        for node_id, node_classes in self.nodes_per_class_outputs_means.items():
            if node_classes is None:
                continue
            # node_classes might be dict-like: class -> value
            for sample_class, value in node_classes.items() if isinstance(node_classes, dict) else enumerate(node_classes):
                # Normalize value to torch tensor on CPU
                if isinstance(value, torch.Tensor):
                    v = value.detach().cpu().float()
                elif isinstance(value, np.ndarray):
                    v = torch.from_numpy(value).float()
                else:
                    # scalar or list
                    try:
                        v = torch.tensor(value).float()
                    except Exception:
                        # skip unrecognized types
                        continue

                if sample_class not in per_class_accum:
                    per_class_accum[sample_class] = v.clone()
                    per_class_counts[sample_class] = 1
                else:
                    # ensure shapes are compatible
                    try:
                        per_class_accum[sample_class] = per_class_accum[sample_class] + v
                        per_class_counts[sample_class] += 1
                    except Exception:
                        # If shapes mismatch, try to convert to scalar mean if possible
                        try:
                            per_class_accum[sample_class] = per_class_accum[sample_class] + v.mean()
                            per_class_counts[sample_class] += 1
                        except Exception:
                            # give up on this value
                            continue

        # Compute final mean per class
        global_per_class_output_means = {}
        for sample_class, summed in per_class_accum.items():
            cnt = per_class_counts.get(sample_class, 1)
            global_per_class_output_means[sample_class] = (summed / cnt)

        # Save aggregated result on the server (CPU tensors)
        self.global_per_class_output_means = global_per_class_output_means

        global_means = []
        for class_name, per_class_output_mean in global_per_class_output_means.items():
            global_means.append(per_class_output_mean)

        global_means = torch.stack(global_means)
        global_mean = torch.mean(global_means,dim=0)

        self.global_output_means = global_mean

    def aggregate_adapters_parameters_fedavg(self):
        nodes_adapters_states = {}
        for node_id, node_adapters in self.nodes_adapters_modules.items():
            if not node_adapters:
                continue
            try:
                nodes_adapters_states[node_id] = node_adapters.state_dict()
            except Exception:
                state = node_adapters if isinstance(node_adapters[node_adapters], dict) else {}
        averaged_state = self.average_state_dicts(nodes_adapters_states)

        self.global_adapters_modules.load_state_dict(averaged_state)

    @torch.no_grad()
    def average_state_dicts(self, state_dicts, weights=None):
        if isinstance(state_dicts, Mapping):
            keys = list(state_dicts.keys())
            sds = [state_dicts[k] for k in keys]
            if weights is None:
                ws = [1.0] * len(sds)
            elif isinstance(weights, Mapping):
                ws = [float(weights[k]) for k in keys]
            else:
                ws = list(weights)
        else:
            sds = list(state_dicts)
            ws = [1.0] * len(sds) if weights is None else list(weights)

        if not sds:
            raise ValueError("Nessun state_dict fornito.")
        if len(sds) != len(ws):
            raise ValueError("state_dicts e weights hanno lunghezze diverse.")

        # Detach su CPU
        sds = [
            OrderedDict((k, v.detach().cpu() if torch.is_tensor(v) else v) for k, v in sd.items())
            for sd in sds
        ]

        # Verifica chiavi
        ref_keys = list(sds[0].keys())
        for i, sd in enumerate(sds[1:], 1):
            if list(sd.keys()) != ref_keys:
                raise ValueError(f"Chiavi non allineate in state_dict indice {i}.")

        # Media pesata
        total_w = float(sum(ws))
        out = OrderedDict()
        for k in ref_keys:
            v0 = sds[0][k]
            if torch.is_tensor(v0) and v0.is_floating_point():
                acc = torch.zeros_like(v0)
                for sd, w in zip(sds, ws):
                    acc.add_(sd[k], alpha=w / total_w)
                out[k] = acc
            else:
                out[k] = v0  # es. num_batches_tracked, interi, bool
        return out

    def add_parameters(self, w, client_model):
        """Add weighted client parameters to global Audio2Visual model."""
        if hasattr(self.global_model, 'audio2image') and hasattr(client_model, 'audio2image'):
            for server_param, client_param in zip(
                    self.global_model.audio2image.parameters(),
                    client_model.audio2image.parameters()):
                if server_param.requires_grad and client_param.requires_grad:
                    server_param.data += client_param.data.clone() * w

    def client_round_starting_hook(self, client):
        """Hook called at the start of each client's training round."""
        print(f"\nStarting training round for Node {client.id}.")
        round_train_metrics = self.round_train_metrics(client)
        round_test_metrics = self.round_test_metrics(client)

        if not self.no_wandb:
            wandb_metrics = client.log_metrics(round_train_metrics, round=self.round, suffix="_start")
            wandb_metrics.update(client.log_metrics(round_test_metrics, round=self.round, suffix="_start"))

            for wandb_metric in wandb_metrics:
                metrics = {wandb_metric: wandb_metrics[wandb_metric], "round": self.round}
                self.data_log(metrics)

        print (f"Node {client.id} pre round metric train {round_train_metrics['text_loss']['mean']:.2f}. test {round_test_metrics['text_loss']['mean']:.2f}")


    def client_round_ending_hook(self, client):
        """Hook called at the end of each client's training round."""
        round_train_metrics = self.round_train_metrics(client)
        round_test_metrics = self.round_test_metrics(client)

        if not self.no_wandb:
            wandb_metrics = client.log_metrics(round_train_metrics, round=self.round, suffix="_end")
            wandb_metrics.update(client.log_metrics(round_test_metrics, round=self.round, suffix="_end"))

            for wandb_metric in wandb_metrics:
                metrics = {wandb_metric: wandb_metrics[wandb_metric], "round": self.round}
                self.data_log(metrics)

        print (f"Node {client.id} post metric train {round_train_metrics['text_loss']['mean']:.2f}. test {round_test_metrics['text_loss']['mean']:.2f}")

    def generate_images_from_diffusion(self, text_embeddings):
        if 't5' not in text_embeddings or 'clip' not in text_embeddings:
            print ('Text embeddings is missing something')
            return[]
        
        prompt_embeds = text_embeddings['t5'].to(self.global_model.diffusion_model_device)
        pooled_prompt_embeds = text_embeddings['clip'].to(self.global_model.diffusion_model_device)

        imgs = []
        for pe, ppe in zip(prompt_embeds, pooled_prompt_embeds):
            pe = pe.unsqueeze(0)
            ppe = ppe.unsqueeze(0)
            img = self.global_model.diffusion_model(
                                        prompt_embeds=pe,
                                        pooled_prompt_embeds=ppe,
                                        num_inference_steps=1,
                                        output_type="pt",
                                        ).images
            imgs.append(img)

        imgs = torch.cat(imgs,dim=0)
        return imgs
    
    def save_generated_images(self, imgs, client, embeddings, suffix=""):
        for idx, img in enumerate(imgs):
            class_name = embeddings['class_name'][idx]
            img_save_path = os.path.join(self.images_output_dir, f"round_{self.round}_node_{client.id}_img_{class_name}_{idx}{suffix}.png")
            img = img.squeeze(0)
            converted_img = transforms.ToPILImage()(img.cpu())
            converted_img.save(img_save_path)
            print(f"Saved generated image to {img_save_path}")


    def generate_images(self, client):
        """Generate images using the client's Audio2Visual model."""
        print(f"\nGenerating images for Node {client.id} using Audio2Visual model.")

        if not os.path.exists(self.images_output_dir):
            try:
                os.makedirs(self.images_output_dir)
            except FileExistsError:
                pass

        embeddings = client.get_audio_embeddings_for_generation(num_embeddings=5)
        prompt_embeds = embeddings['t5'].to(self.global_model.diffusion_model_device)
        pooled_prompt_embeds = embeddings['clip'].to(self.global_model.diffusion_model_device)
        self.global_model.diffusion_model.to(self.global_model.diffusion_model_device)

        imgs = self.generate_images_from_diffusion(embeddings)

        self.save_generated_images(imgs, client, embeddings)
        
        # for idx, img in enumerate(imgs):
        #     class_name = embeddings['class_name'][idx]
        #     img_save_path = os.path.join(self.images_output_dir, f"round_{self.round}_node_{client.id}_img_{class_name}_{idx}.png")
        #     img = img.squeeze(0)
        #     converted_img = transforms.ToPILImage()(img.cpu())
        #     converted_img.save(img_save_path)
        #     print(f"Saved generated image to {img_save_path}")

        embeddings = client.get_audio_embeddings_for_generation(num_embeddings=2, from_train=True )
        prompt_embeds = embeddings['t5'].to(self.global_model.diffusion_model_device)
        pooled_prompt_embeds = embeddings['clip'].to(self.global_model.diffusion_model_device)

        imgs = self.generate_images_from_diffusion(embeddings)
        self.save_generated_images(imgs, client, embeddings, "_train")

        # for idx, img in enumerate(imgs):
        #     class_name = embeddings['class_name'][idx]
        #     img_save_path = os.path.join(self.images_output_dir, f"round_{self.round}_node_{client.id}_img_{class_name}_{idx}_train.png")
        #     img = img.squeeze(0)
        #     converted_img = transforms.ToPILImage()(img.cpu())
        #     converted_img.save(img_save_path)
        #     print(f"Saved generated image from trainset to {img_save_path}")

        node_dataset = client.node_data.train_dataset.dataset
        text_embs = node_dataset.text_embs
        for node_class in node_dataset.active_classes.keys():
            prompt_embeds = text_embs[node_class][self.diffusion_type]['prompt_embeds']
            pooled_prompt_embeds = text_embs[node_class][self.diffusion_type]['pooled_prompt_embeds']
            imgs = self.global_model.diffusion_model(
                                        prompt_embeds= prompt_embeds,
                                        pooled_prompt_embeds=pooled_prompt_embeds,
                                        num_inference_steps=1,
                                        output_type="pt",
                                        ).images
            img_save_path = os.path.join(self.images_output_dir, f"round_{self.round}_node_{client.id}_img_{node_class}_from_textembs.png")
            converted_img = transforms.ToPILImage()(imgs[0].cpu().detach())
            converted_img.save(img_save_path)
            print(f"Saved generated image from text embeddings to {img_save_path}")
            
    def create_clients(self, clientObj):
        config = self.args.json_config if hasattr(self.args, 'json_config') else None 
        if config is None:
            print("No JSON configuration provided for Audio2Visual clients.")
            return

        for node_id, node_config in config.nodes.items():
            node_dataset = None

            print(f"Creating {node_id} Audio2Visual node from nodes_tasks configuration:, using dataset {node_config.dataset} and labels {node_config.selected_classes}")

            if node_config.dataset == "VEGAS":
                audio_embedding_file = "/home/lpala/fedgfe/dataset/Audio/vegas_audio_embs_dict.pt"
                audio_embedding_file_loaded = False
                selected_classes = getattr(node_config, 'selected_classes', None)
                excluded_classes = getattr(node_config, 'excluded_classes', None)
                node_dataset = VEGASDataset(
                    selected_classes=selected_classes,
                    excluded_classes=excluded_classes,
                    use_saved_audio_embeddings=self.use_saved_audio_embeddings
                )

                for client in self.clients:
                    if client.dataset == "VEGAS":
                        if isinstance(client.node_data.train_dataset, torch.utils.data.Subset):
                            dataset = client.node_data.train_dataset.dataset
                        else:
                            dataset = client.node_data.train_dataset
                        if isinstance(dataset, VEGASDataset) and dataset.audio_embs_from_file is not None:
                            audio_embedding_file_loaded = True
                            node_dataset.audio_embs_from_file = dataset.audio_embs_from_file
                            break

                if self.use_saved_audio_embeddings and self.audio_embedding_file_name is not None and not audio_embedding_file_loaded:
                    node_dataset.load_audio_embeddings_from_file(self.audio_embedding_file_name)
                    logger.info(f"Loaded audio embeddings for VEGAS dataset from {self.audio_embedding_file_name}")

            elif node_config.dataset == "ESC50":
                selected_classes = getattr(node_config, 'selected_classes', None)
                excluded_classes = getattr(node_config, 'excluded_classes', None)
                # split = getattr(node_config, 'dataset_split', 'train')
                # use_folds = getattr(node_config, 'use_folds', False)
                train_folds = getattr(node_config, 'train_folds', [0, 1, 2, 3])
                test_folds = getattr(node_config, 'test_folds', [4])

                node_dataset = ESC50Dataset(
                    selected_classes=selected_classes,
                    excluded_classes=excluded_classes,
                    # split=split,
                    # use_folds=use_folds,
                    train_folds=train_folds,
                    test_folds=test_folds,
                    node_id=int(node_id)
                )

            if node_dataset is None:
                logger.warn("No dataset assigned to node, skipping node creation")
                continue
            node = clientObj(self.args,
                             int(node_id),
                             node_config=node_config,
                             global_model=self.global_model,
                             dataset=node_dataset
            )
        
            self.clients.append(node)

        return self.clients

    def set_clients(self, clientObj):
        return self.create_clients(clientObj)

    def define_metrics(self):
        """Define metrics for Audio2Visual federated learning."""
        wandb.define_metric(f"round")

        for client in self.clients:
            client.define_metrics()
            wandb.define_metric(f"test/node_{client.id}/a2v_loss", step_metric="round")
            wandb.define_metric(f"train/node_{client.id}/a2v_loss", step_metric="round")
            wandb.define_metric(f"test/node_{client.id}/generation_quality", step_metric="round")
            wandb.define_metric(f"test/node_{client.id}/audio_image_alignment", step_metric="round")

    def evaluate(self):
        """Evaluate Audio2Visual models."""
        stats_test = self.test_metrics()
        stats_train = self.train_metrics()

        if stats_test == None or stats_train == None:
            return

        # Calculate average losses
        test_losses = []
        train_losses = []

        for node_id in stats_test:
            if 'loss' in stats_test[node_id]:
                test_losses.append(stats_test[node_id]['loss']['mean'])

        for node_id in stats_train:
            if 'loss' in stats_train[node_id]:
                train_losses.append(stats_train[node_id]['loss']['mean'])

        if test_losses:
            avg_test_loss = sum(test_losses) / len(test_losses)
            print(f"**Federation average test loss: {avg_test_loss:.4f}")

        if train_losses:
            avg_train_loss = sum(train_losses) / len(train_losses)
            print(f"**Federation average train loss: {avg_train_loss:.4f}")

    def train_metrics(self):
        """Calculate training metrics for Audio2Visual clients."""
        nodes_train_metrics = {}

        for client_index, c in enumerate(self.clients):
            if c.node_data.test_data == None:
                print(f"Client {c.id} test data is None")
                continue
            c._move_to_gpu(self.device)
            node_train_metrics = c.train_metrics()
            nodes_train_metrics[c.id] = node_train_metrics
            c._move_to_cpu()

        return nodes_train_metrics

    def test_metrics(self, standalone=False):
        """Calculate test metrics for Audio2Visual clients."""
        test_clients_stats = {}

        for client_index, c in enumerate(self.clients):
            if c.node_data.test_data == None:
                print(f"Client {c.id} test data is None")
                continue

            test_clients_stats[c.id] = {}

            # Pre-load next client's data if applicable
            if client_index < len(self.clients) - 1:
                self.clients[client_index + 1].node_data.load_test_data(self.args.batch_size)

            test_clients = self.clients if not standalone else [c]

            for t in test_clients:
                if c.node_data.test_data == None:
                    print(f"Client {c.id} test data is None")
                    continue

                node_test_metrics = c.test_metrics(t)
                test_clients_stats[t.id] = node_test_metrics

            # Unload test data to save memory
            if client_index > 0 and self.reduce_memory_footprint == True:
                c.node_data.unload_test_data()

        return test_clients_stats

    def round_train_metrics(self, client):
        """Calculate round training metrics for a client."""

        train_metric = NodeMetric(phase=NodeMetric.Phase.TRAIN)
        client._move_to_gpu(self.device)
        train_metric = client.train_metrics()
        client._move_to_cpu()

        node_loss_aggregated = train_metric['text_loss']['mean'] if 'text_loss' in train_metric else 0.0
        round_loss = node_loss_aggregated
        # self.data_log({f'train/node_{client.id}/a2v_round_train_loss': round_loss, "round": self.round})
        logger.debug(f"Node {client.id} round train loss: {round_loss}")
        
        return train_metric

    def round_test_metrics(self, client):
        """Calculate round test metrics for a client."""

        client._move_to_gpu(self.device)
        node_test_metrics = client.test_metrics()
        client._move_to_cpu()

        round_loss = node_test_metrics['text_loss']['mean'] if 'text_loss' in node_test_metrics else 0.0
        logger.debug(f"Node {client.id} round test loss: {round_loss}")

        return node_test_metrics
    
    def _move_to_device(self, device):
        self.global_model.to(device)

        # Move optimizer state if needed
        if hasattr(self, 'global_optimizers') and self.global_optimizers is not None:
            for optimizer_module, optimizer in self.global_optimizers.items():
                if isinstance(optimizer, torch.optim.AdamW):
                    move_optimizer_state(optimizer, device)

    def _move_to_gpu(self, device):
        print(f"Server moving to GPU: {device}")
        self._move_to_device(device)

    def _move_to_cpu(self):
        print(f"Server moving to CPU for memory optimization")
        self._move_to_device('cpu')

@torch.no_grad()
def move_optimizer_state(optimizer, device):
    for pg in optimizer.param_groups:
        for p in pg["params"]:
            st = optimizer.state.get(p, None)
            if not st:
                continue
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    st[k] = v.to(device)
    torch.cuda.empty_cache()