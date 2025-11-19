from collections import defaultdict
import copy
import os
import sys
import sklearn
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader, Subset

import wandb

from flcore.clients.clientbase import Client
from datautils.node_dataset import NodeData
from modelutils.optimizer_manager import OptimizerManager
from modelutils.pretext_trainer import PretextTrainer
from modelutils.downstream_trainer import DownstreamTrainer
from sklearn.preprocessing import label_binarize
from sklearn.metrics import balanced_accuracy_score
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torchvision import transforms
from utils.check_parameters import check_optimizer_params, print_model_gradients_status
from utils.node_metric import NodeMetric

from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR

from modelutils.model_stats import count_changed_weights

import torch.nn.functional as F

from tqdm import tqdm

from torchviz import make_dot

from collections import Counter
from transformers import ASTModel, ASTFeatureExtractor

sys.path.append('/home/lpala/fedgfe/system/flcore/trainmodel/Audio2Visual_NoData')
from flcore.trainmodel.Audio2Visual_NoData.src.models.audio2image import Audio2Image, SDImageModel, ImageDiffusion

import logging

logger = logging.getLogger(__name__)

class clientA2V(Client):
    def __init__(self, args, node_id, node_config=None, global_model=None, dataset=None, store_audio_embedding=False, **kwargs):
        super().__init__(args, node_id, None, None, **kwargs)

        logger = logging.getLogger(f"{__name__}_{node_id}")



        self.id = node_id
        self.learning_rate = args.local_learning_rate
        self.train_loader = None
        self.no_wandb = args.no_wandb
        self.train_dataloader = None
        self.test_dataloader = None
        self.global_model = global_model
        self.global_rounds = args.global_rounds

        self.store_audio_embedding = self.args.json_config.feda2v.store_audio_embeddings

        self.model = global_model

        if 'data_log' in kwargs:
            self.data_log = kwargs['data_log']
        else:
            self.data_log = None

        if node_config.dataset != None:
            self.dataset = node_config.dataset
        else:
            self.dataset = args.dataset

        self.node_data = NodeData(args, node_id,dataset=dataset)
        self.node_data.dataset = self.dataset

        self.experiment_config = getattr(args.json_config, 'experiment', None)

        self.optimize_memory_usage = getattr(self.experiment_config, 'optimize_memory_usage', False )

        self.train_optimizer = None
        self.finetuning_optimizer = None

        self.model_optimizer = args.model_optimizer

        self.defined_train_metrics = { 'loss_at_start': None, 'loss_at_end': None, 'loss_reduction': None }
        self.defined_test_metrics = {}
        self.use_saved_audio_embeddings = getattr(args, 'use_saved_audio_embeddings', False)

        self.text_losses_summed = True

        # Audio2Visual specific initialization
        self.diffusion_type = getattr(node_config, 'diffusion_type', 'sd')  # 'sd', 'flux', or 'cogx'
        self.use_act_loss = getattr(args, 'use_act_loss', True)
        self.audio_model_name = getattr(args, 'audio_model_name', "MIT/ast-finetuned-audioset-10-10-0.4593")
        self.img_pipe_name = getattr(args, 'img_pipe_name', "runwayml/stable-diffusion-v1-5")
        self.img_lcm_lora_id = getattr(args, 'img_lcm_lora_id', "latent-consistency/lcm-lora-sdv1-5")

        self.audio2image_model = self.global_model.get_audio2image_model()

        self.per_class_outputs = None  # Will be computed at the end of each round
        self.per_class_outputs_mean = None  # Will be computed at the end of each round

        # Storage for adapter outputs collected during training
        self.training_adapter_outputs_all = None   # All outputs: {class_name: {adapter_name: [outputs]}}
        self.training_adapter_outputs_mean = None  # Mean outputs: {class_name: {adapter_name: mean_tensor}}

        self.audio_encoder_output = None

        self.adapters = self.model.adapters if hasattr(self.model, 'adapters') else {}
        self.adapters_modules = self.model.adapters_modules if hasattr(self.model, 'adapters_modules') else None

        self.audio_embedding_store = {}

    def define_metrics(self):
        self.metrics_path = "node_" + str(self.id) + "/"
        if self.global_model is not None:
            self.global_model.define_metrics(metrics_path=self.metrics_path)


        return

    def log_metrics(self, metrics, round=None, prefix="", suffix=""):
        if metrics == None:
            return
        
        metrics_values = {}

        for metric_name in metrics._defined_metrics:
            if metrics.phase == NodeMetric.Phase.TRAIN:
                metric_path = f"train/{self.metrics_path}{prefix}{metric_name}{suffix}"
                # metric_path = f"train/{self.metrics_path}a2v_{prefix}{metric_name}{suffix}"
            elif metrics.phase == NodeMetric.Phase.TEST:
                metric_path = f"test/{self.metrics_path}{prefix}{metric_name}{suffix}"
                # metric_path = f"test/{self.metrics_path}a2v_{prefix}{metric_name}{suffix}"
            metric_value = metrics[metric_name]['mean']
            if round == None:
                round = self.round
            
            metrics_values[metric_path] = metric_value

        return metrics_values
    
    def test_genarated_images(self, generated_images):

        if type(generated_images) is not dict:
            print ( "Generated images is not dict")
            return
        
        if 'on_test' not in generated_images and 'on_train' not in generated_images:
            print ( "Generated images does not contain train or test images")
            return

        print ( "Stub method")
        return

    def train(self, client_device=None, rewind_train_node=None, training_task="both"):
        """Train the Audio2Visual model."""
        logger.debug(f"*** Node {self.id} memory before training {torch.cuda.memory_allocated(self.device)//(1024**2)} MB")

        self.model = self.global_model

        node_trainloader = self.load_train_data()
        node_testloader = self.load_test_data()

        trainloader = node_trainloader
        start_time = time.time()
        device = self.device
        if client_device != None:
            device = client_device

        # Move model to GPU before training
        self._move_to_gpu(self.device)

        max_local_epochs = self.local_epochs
        local_epochs = max_local_epochs

        if self.round == 1:
            print(f"Creating optimizer for node {self.id} with model optimizer {self.model_optimizer} and learning rate {self.learning_rate}")
            self.setup_optimizer()
            self.optimizer = self.train_optimizer

            print(f"Creating learning rate scheduler for node {self.id}")
            self.scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=self.global_rounds)

            if self.learning_rate_schedule:
                self.setup_learning_rate_scheduler(self.global_rounds)

            if self.no_wandb == False:
                wandb.watch(self.audio2image_model, log='all', log_freq=100, criterion=None, log_graph=False, idx=self.id)

            self.print_optimizer_info(self.id)

        self.model.train()

        # Train the Audio2Visual model
        print(f"Node {self.id} training Audio2Visual model for {local_epochs} epochs")
        self.train_a2v(local_epochs, trainloader, client_device=device)

        # Compute and store per-class mean output at the end of the round
        print(f"Node {self.id} computing per-class mean output for round {self.round}")

        self.update_per_class_mean_output(use_train=True, device=device)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        # Move model to CPU after training to save GPU memory
        self._move_to_cpu()
        print(f"*** Node {self.id} memory after training and moving to CPU {torch.cuda.memory_allocated(device)//1024**2} MB")

    def update_local_adapters(self, adapters = None, projections = None):
        for module_name, module in self.adapters.items():
            if module_name in adapters:
                local_adapter = self.adapters[module_name]
                global_adapter_state_dict = adapters[module_name].state_dict()
                local_adapter.load_state_dict(global_adapter_state_dict)
        
    def train_a2v(self, epochs, dataloader, client_device=None):
        # self.train_time_cost['num_sslrounds'] += 1

        device = client_device if client_device is not None else self.device

        a2i = self.global_model
        self.model.to(device)

        # Initialize storage for adapter outputs per class during training
        # Structure: {class_name: {adapter_name: [list of outputs]}}

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            training_adapter_outputs = defaultdict(lambda: defaultdict(list))

            # for batch_idx, (audio_data, text_embeddings) in enumerate(dataloader):
            for batch_idx, samples in enumerate(dataloader):
                # Move data to device
                if 'audio' in samples and isinstance(samples['audio'], torch.Tensor):
                    audio_data = samples['audio'].to(device)
                else:
                    raise ValueError("Audio data not found in the batch samples")

                # Handle text embeddings (target for training)
                target_prompt_embeds = None
                target_pooled_prompt_embeds = None

                text_embeddings = samples.get('text_emb', None)
                if isinstance(text_embeddings, dict) and self.diffusion_type in text_embeddings:
                    if self.diffusion_type == 'sd':
                        target_prompt_embeds = text_embeddings['sd']

                        if target_prompt_embeds is not None:
                            target_prompt_embeds = target_prompt_embeds.to(device)

                    elif self.diffusion_type == 'flux':
                        target_prompt_embeds = text_embeddings['flux'].get('prompt_embeds', None)
                        target_pooled_prompt_embeds = text_embeddings['flux'].get('pooled_prompt_embeds', None)

                        if target_prompt_embeds is not None:
                            target_prompt_embeds = target_prompt_embeds.to(device)
                        if target_pooled_prompt_embeds is not None:
                            target_pooled_prompt_embeds = target_pooled_prompt_embeds.to(device)

                audio_embedding = samples.get('audio_emb', None)

                # Zero gradients
                self.optimizer.zero_grad()



                # Forward pass
                if isinstance(audio_data, torch.Tensor) and isinstance(self.audio2image_model.feature_extractor, ASTFeatureExtractor):
                    audio_data = audio_data.to('cpu').numpy()

                # outputs = self.audio2image_model(audio_data, target_prompt_embeds, target_pooled_prompt_embeds)
                outputs = self.model( audio_data,
                                        img_target_prompt_embeds=target_prompt_embeds,
                                        img_target_pooled_prompt_embeds=target_pooled_prompt_embeds,
                                        audio_embedding=audio_embedding
                                    )

                outputs['class_name'] = samples.get('class_name', None)

                if self.store_audio_embedding:
                    outputs['audio_filename'] = samples.get('audio_filename', None)
                    self.store_audio_embeddings(audio_data, outputs)

                # Store adapter outputs per class for later mean computation
                # Adapter outputs are directly in the outputs dict with keys 'clip', 't5', etc.
                batch_class_names = outputs['class_name']

                # Group outputs by class first, then by adapter

                if epoch == epochs - 1:
                    training_adapter_outputs = self.store_adapters_output_per_class ( outputs, training_adapter_outputs )

                losses = outputs['text_loss'] 
                losses_output = ""
                if self.text_losses_summed:
                    loss = torch.tensor(0.0)
                    if outputs['text_loss'] is not None:
                        if isinstance(outputs['text_loss'], tuple):
                            for l in outputs['text_loss']:
                                loss = loss.to(l.device)
                                loss += l
                                losses_output = f"{l.item()} "
                        else:
                            loss = outputs['text_loss']

                    loss.backward()
                    losses_output = f"{loss.item()}"
                    epoch_loss += loss.item()
                else:
                    losses_count = len(losses)
                    for loss_index,loss in enumerate(outputs['text_loss']):
                        retain_graph = True
                        if loss_index >= losses_count-1:
                            retain_graph = False
                        loss.backward(retain_graph=retain_graph)
                        epoch_loss += loss.item()
                        losses_output = f"{losses_output} {loss.item():.03f} "

                self.optimizer.step()
                num_batches += 1

                if batch_idx % 10 == 0:
                    print(f"Node {self.id} Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {losses_output}")

            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                print(f"Node {self.id} Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.3f}")
            
            for class_name, output in training_adapter_outputs.items():
                for k in output:
                    if type(k) == list:
                        training_adapter_outputs[k] = torch.stack(training_adapter_outputs[k])    

        # Compute mean outputs for each class and adapter at the end of all epochs
        print(f"Node {self.id} - Computing mean adapter outputs per class from training data")
        self.training_adapter_outputs_all = dict(training_adapter_outputs)  # Store all outputs
        self.training_adapter_outputs_mean = {}

        total_samples_stored = 0
        total_adapters = set()

        # Structure: {class_name: {adapter_name: mean_output}}
        for class_name, adapters_dict in training_adapter_outputs.items():
            self.training_adapter_outputs_mean[class_name] = {}

            for adapter_name, outputs_list in adapters_dict.items():
                if adapter_name not in self.adapters.keys():
                    continue
                
                if len(outputs_list) > 0:
                    # Stack all outputs and compute mean
                    self.training_adapter_outputs_mean[class_name][adapter_name] = torch.mean(
                        torch.stack(outputs_list), dim=0
                    )
                    total_samples_stored += len(outputs_list)
                    total_adapters.add(adapter_name)
                    print(f"  - Class '{class_name}' {adapter_name}: mean from {len(outputs_list)} samples")
                else:
                    print(f"  Warning: No outputs for class '{class_name}' {adapter_name}")

        print(f"Node {self.id} - Stored outputs for {len(self.training_adapter_outputs_mean)} classes, "
              f"{len(total_adapters)} adapters, {total_samples_stored} total samples")

        if self.data_log:
            self.data_log({f"train/node_{self.id}/a2v_loss": avg_loss, "epoch": epoch, "round": self.round})

    def store_adapters_output_per_class(self, batch_output, per_class_adapters ):
        if 'class_name' in batch_output:
            batch_class_names = batch_output['class_name']
        else:
            return None

        for idx, class_name in enumerate(batch_class_names):

            if class_name not in per_class_adapters:
                per_class_adapters[class_name] = {}

            if 'audio_embeddings' not in per_class_adapters[class_name]:
                per_class_adapters[class_name]['audio_embeddings'] = []

            per_class_adapters[class_name]['audio_embeddings'].append(batch_output['audio_embeddings'][idx])

            for adapter_name in self.adapters.keys():
                if adapter_name not in per_class_adapters[class_name]:
                    per_class_adapters[class_name][adapter_name] = []

                if adapter_name in batch_output:
                    per_class_adapters[class_name][adapter_name].append(batch_output[adapter_name][idx])


        return per_class_adapters
                


    def store_audio_embeddings(self, audio_data, model_outputs):
        """Store audio embeddings from the model outputs."""
        if model_outputs is None or 'audio_embeddings' not in model_outputs:
            print(f"Node {self.id} - No audio embeddings found in model outputs to store")
            return

        audio_embeddings = model_outputs['audio_embeddings']
        if not isinstance(audio_embeddings, torch.Tensor):
            print(f"Node {self.id} - Audio embeddings are not a tensor, cannot store")
            return

        audio_filename = model_outputs.get('audio_filename', None)

        audio_embeddings_data = {}
        if audio_filename is not None and 'audio_embeddings' in model_outputs:
            embs = model_outputs['audio_embeddings']
            audio_class = model_outputs.get('class_name', None)
            # convert to numpy (N, D)
            if isinstance(embs, torch.Tensor):
                embs_np = embs.detach().cpu().numpy()
            else:
                embs_np = np.asarray(embs)


            # build mapping filename -> embedding for each sample in the batch
            if audio_filename is None:
                # no filenames provided: generate synthetic keys
                for i, emb in enumerate(embs_np):
                    audio_embeddings_data[f"{self.id}_sample_{len(self.audio_embedding_store)}_{i}"] = emb
            elif isinstance(audio_filename, (list, tuple, np.ndarray)):
                filenames = list(audio_filename)
                # if lengths mismatch, still map available filenames and suffix others
                for i, emb in enumerate(embs_np):
                    if i < len(filenames):
                        key = audio_class[i]+":"+filenames[i]
                        audio_file_data = { 'class_name': audio_class[i], 'embeddings': embs_np[i] }
                    else:
                        key = f"{self.id}_sample_{len(self.audio_embedding_store)}_{i}"
                    audio_embeddings_data[key] = audio_file_data
            else:
                # single filename string: if batch len == 1 map directly, otherwise add index suffixes
                if embs_np.shape[0] == 1:
                    audio_embeddings_data[audio_filename] = embs_np[0]
                else:
                    for i, emb in enumerate(embs_np):
                        audio_embeddings_data[f"{audio_filename}_{i}"] = emb

            if isinstance(self.audio_embedding_store, dict):
                self.audio_embedding_store.update(audio_embeddings_data)
            elif isinstance(self.audio_embedding_store, list):
                self.audio_embedding_store.append(audio_file_data)
            else:
                # fallback: convert to dict and update
                try:
                    self.audio_embedding_store = dict(self.audio_embedding_store)
                except Exception:
                    self.audio_embedding_store = {}
                self.audio_embedding_store.update(audio_file_data)

        # print(f"Node {self.id} - Stored audio embeddings of shape {audio_embeddings_data['embeddings'].shape}")



    def test_a2v(self, testloader=None):
        """Test the Audio2Visual model."""
        if testloader is None:
            testloader = self.load_test_data()

        self.model.eval()
        test_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_idx, samples in enumerate(testloader):
                # Move data to device
                if 'audio' in samples and isinstance(samples['audio'], torch.Tensor):
                    audio_data = samples['audio'].to(self.device)
                else:
                    raise ValueError("Audio data not found in the batch samples")

                # Handle text embeddings
                target_prompt_embeds = None
                target_pooled_prompt_embeds = None
                text_embeddings = samples.get('text_emb', None)
                if isinstance(text_embeddings, dict):
                    target_prompt_embeds = text_embeddings.get('prompt_embeds', None)
                    target_pooled_prompt_embeds = text_embeddings.get('pooled_prompt_embeds', None)

                    if target_prompt_embeds is not None:
                        target_prompt_embeds = target_prompt_embeds.to(self.device)
                    if target_pooled_prompt_embeds is not None:
                        target_pooled_prompt_embeds = target_pooled_prompt_embeds.to(self.device)
                
                if isinstance(audio_data, torch.Tensor) and isinstance(self.audio2image_model.feature_extractor, ASTFeatureExtractor):
                    audio_data = audio_data.to('cpu').numpy()
                # Forward pass
                outputs = self.model(audio_data, target_prompt_embeds, target_pooled_prompt_embeds)

                # Calculate loss
                if 'text_loss' in outputs and outputs['text_loss'] is not None:
                    loss = 0.0
                    if isinstance(outputs['text_loss'], tuple):
                        for l in outputs['text_loss']:
                            loss += l
                    else:
                        loss = outputs['text_loss']

                    test_loss += loss.item()
                    num_batches += 1

        if num_batches > 0:
            avg_test_loss = test_loss / num_batches
            print(f"Node {self.id} Test Loss: {avg_test_loss:.4f}")

            if self.data_log:
                self.data_log({f"test/node_{self.id}/a2v_loss": avg_test_loss, "round": self.round})

            return avg_test_loss
        
        self.model.train_
        return 0.0

    def print_optimizer_info(self, client_id):
        if self.train_optimizer is None:
            print(f"Node {client_id}: No optimizer initialized")
            return
        
        for param_group_index, param_group in enumerate(self.train_optimizer.param_groups):
            num_params = sum(p.numel() for p in param_group["params"] if p.requires_grad)
            size = sum(p.numel()*p.element_size() for p in param_group["params"] if p.requires_grad)
            print(f"Node {client_id} optimizer param group {param_group_index} "
                  f"tensors {len(param_group['params'])} parameters {num_params} "
                  f"size {size} lr {param_group['lr']}")

    def setup_optimizer(self):
        """
        Setup optimizer for Audio2Visual model.
        Only trains the local copies of adapters and projections.
        """
        # Get trainable parameters from LOCAL adapters and projections only
        trainable_params = []

        trainable_params.extend(self.model.parameters())

        if len(trainable_params) == 0:
            raise ValueError(f"Node {self.id}: No trainable parameters found in local adapters!")

        # Create optimizer
        if self.model_optimizer.lower() == "adamw":
            self.train_optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.learning_rate,
                # weight_decay=self.optimizer_weight_decay
            )
        elif self.model_optimizer.lower() == "sgd":
            self.train_optimizer = torch.optim.SGD(
                trainable_params,
                lr=self.learning_rate,
                momentum=self.optimizer_momentum,
                # weight_decay=self.optimizer_weight_decay
            )
        else:
            self.train_optimizer = torch.optim.Adam(
                trainable_params,
                lr=self.learning_rate,
                # weight_decay=self.optimizer_weight_decay
            )

        self.finetuning_optimizer = self.train_optimizer

    def setup_learning_rate_scheduler(self, rounds):
        """Setup learning rate scheduler for Audio2Visual model."""
        self.scheduler = self.optimizer_manager.setup_learning_rate_scheduler(
            optimizer=self.optimizer,
            rounds=rounds,
            use_scheduler=self.learning_rate_schedule
        )

    def set_parameters(self, global_model):
        """
        Set parameters from the global Audio2Visual model.
        Updates only the local copies of adapters and projections.
        """
        updated_parameters = 0
        not_updated_parameters = 0

        self._move_to_gpu(self.device)

        # Get the global audio2image model
        global_audio2image = global_model.get_audio2image_model() if hasattr(global_model, 'get_audio2image_model') else None

        if global_audio2image is None:
            print(f"Warning: Node {self.id} - Cannot get global audio2image model")
            self._move_to_cpu()
            return

        # Update local adapters and projections from global model
        print(f"Node {self.id} - Updating local adapters from global model")

        # Update clip_adapter
        if hasattr(self, 'local_clip_adapter') and self.local_clip_adapter is not None:
            if hasattr(global_audio2image, 'clip_adapter') and global_audio2image.clip_adapter is not None:
                for global_param, local_param in zip(global_audio2image.clip_adapter.parameters(),
                                                      self.local_clip_adapter.parameters()):
                    if not torch.equal(local_param.data, global_param.data):
                        local_param.data.copy_(global_param.data)
                        updated_parameters += 1
                    else:
                        not_updated_parameters += 1
                print(f"  - Updated clip_adapter")

        # Update clip_projection
        if hasattr(self, 'local_clip_projection') and self.local_clip_projection is not None:
            if hasattr(global_audio2image, 'clip_projection') and global_audio2image.clip_projection is not None:
                for global_param, local_param in zip(global_audio2image.clip_projection.parameters(),
                                                      self.local_clip_projection.parameters()):
                    if not torch.equal(local_param.data, global_param.data):
                        local_param.data.copy_(global_param.data)
                        updated_parameters += 1
                    else:
                        not_updated_parameters += 1
                print(f"  - Updated clip_projection")

        # Update t5_adapter
        if hasattr(self, 'local_t5_adapter') and self.local_t5_adapter is not None:
            if hasattr(global_audio2image, 't5_adapter') and global_audio2image.t5_adapter is not None:
                for global_param, local_param in zip(global_audio2image.t5_adapter.parameters(),
                                                      self.local_t5_adapter.parameters()):
                    if not torch.equal(local_param.data, global_param.data):
                        local_param.data.copy_(global_param.data)
                        updated_parameters += 1
                    else:
                        not_updated_parameters += 1
                print(f"  - Updated t5_adapter")

        # Update t5_projection
        if hasattr(self, 'local_t5_projection') and self.local_t5_projection is not None:
            if hasattr(global_audio2image, 't5_projection') and global_audio2image.t5_projection is not None:
                for global_param, local_param in zip(global_audio2image.t5_projection.parameters(),
                                                      self.local_t5_projection.parameters()):
                    if not torch.equal(local_param.data, global_param.data):
                        local_param.data.copy_(global_param.data)
                        updated_parameters += 1
                    else:
                        not_updated_parameters += 1
                print(f"  - Updated t5_projection")

        print(f"Node {self.id} - Updated {updated_parameters} parameters, {not_updated_parameters} unchanged")

        self._move_to_cpu()

    def test_metrics(self, test_client=None, on_train=False):

        if on_train:
            dataloader = self.load_train_data()
        else:
            dataloader = self.load_test_data()

        if dataloader is not None:
            node_metrics = self.model.train_metrics(dataloader, audio2image_only=True)

        node_metrics.phase = NodeMetric.Phase.TEST

        return node_metrics

        print (f"Testing metrics is called on node {self.id}, but it should be done by central node")
        return None

        """Test metrics for Audio2Visual model."""
        if test_client is None:
            test_client = self

        if on_train:
            testloader = test_client.load_train_data()
        else:
            testloader = test_client.load_test_data()

        if testloader is None:
            return NodeMetric(phase=NodeMetric.Phase.TEST)

        # Test the A2V model and return loss
        test_loss = self.test_a2v(testloader)

        # Create metrics
        node_metrics = NodeMetric(phase=NodeMetric.Phase.TEST)
        node_metrics.define_metrics(['loss'], task_count=1)
        node_metrics[0]['loss'] = test_loss
        node_metrics['samples'] = len(testloader.dataset) if hasattr(testloader, 'dataset') else 1

        return node_metrics

    def train_metrics(self, trainloader=None):
        if trainloader is None:
            trainloader = self.load_train_data()

        node_metrics = NodeMetric(phase=NodeMetric.Phase.TRAIN)
        node_metrics.define_metrics(self.model.defined_train_metrics, task_count=1)
        
        if trainloader is not None:
            node_metrics = self.model.train_metrics(trainloader, audio2image_only=True)

        return node_metrics

        """Calculate training metrics for Audio2Visual model."""
        if trainloader is None:
            trainloader = self.load_train_data()

        if trainloader is None:
            return NodeMetric(phase=NodeMetric.Phase.TRAIN)

        # Calculate training metrics
        # train_loss = self.test_a2v(trainloader)
        train_loss = 0.0
        num = 0
         
        for batch_idx, samples in enumerate(trainloader):
            # Move data to device
            if 'audio' in samples and isinstance(samples['audio'], torch.Tensor):
                audio_data = samples['audio'].to(self.device)
            else:
                raise ValueError("Audio data not found in the batch samples")

            # Handle text embeddings
            target_prompt_embeds = None
            target_pooled_prompt_embeds = None
            text_embeddings = samples.get('text_emb', None)
            if isinstance(text_embeddings, dict):
                target_prompt_embeds = text_embeddings.get('prompt_embeds', None)
                target_pooled_prompt_embeds = text_embeddings.get('pooled_prompt_embeds', None)

                if target_prompt_embeds is not None:
                    target_prompt_embeds = target_prompt_embeds.to(self.device)
                if target_pooled_prompt_embeds is not None:
                    target_pooled_prompt_embeds = target_pooled_prompt_embeds.to(self.device)
            
            if isinstance(audio_data, torch.Tensor) and isinstance(self.model.feature_extractor, ASTFeatureExtractor):
                audio_data = audio_data.to('cpu').numpy()
            # Forward pass
            outputs = self.model(audio_data, target_prompt_embeds, target_pooled_prompt_embeds)
            # Calculate loss
            if outputs['text_loss'] is not None:
                if isinstance(outputs['text_loss'], tuple):
                    for l in outputs['text_loss']:
                        train_loss += l.item()
                else:
                    train_loss = outputs['text_loss'].item()
            num += len(audio_data)


        return node_metrics

    def _move_to_device(self, device):
        """Move Audio2Visual model to specified device."""
        self.model.to(device)

        # Move optimizer state if needed
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            if isinstance(self.optimizer, torch.optim.AdamW):
                move_optimizer_state(self.optimizer, device)

    def _move_to_gpu(self, device):
        if self.optimize_memory_usage:
            print(f"Node {self.id} moving to GPU: {device}")
            self._move_to_device(device)

    def _move_to_cpu(self):
        if self.optimize_memory_usage:
            print(f"Node {self.id} moving to CPU for memory optimization")
            self._move_to_device('cpu')

    def filter_batch_by_class(self, batch_data, target_class):
        filtered_batch = {}

        if isinstance(batch_data, dict):

            # filtra il dict prendendo solo i campi relativi alla classe corrente
            if 'class_name' in batch_data:
                # Support multiple types for class_name (torch.Tensor, np.ndarray, list)
                raw_class_names = batch_data['class_name']
                if isinstance(raw_class_names, torch.Tensor):
                    class_names_arr = raw_class_names.cpu().numpy()
                else:
                    class_names_arr = np.array(raw_class_names)

                # Element-wise comparison to find samples of the current class
                class_mask = (class_names_arr == class_name)

                # If no matches skip this batch
                if not np.any(class_mask):
                    return []  # No samples for this class in the batch

                # Filter batch data to only include samples of the current class
                for key, value in batch_data.items():
                    if isinstance(value, torch.Tensor):
                        # build boolean mask on the same device as the tensor for indexing
                        try:
                            mask_tensor = torch.tensor(class_mask, dtype=torch.bool, device=value.device)
                        except Exception:
                            mask_tensor = torch.tensor(class_mask, dtype=torch.bool)
                        try:
                            filtered_batch[key] = value[mask_tensor]
                        except Exception:
                            # fallback: convert value to numpy and index, then convert back if possible
                            try:
                                filtered_np = np.array(value.cpu().numpy())[class_mask]
                                filtered_batch[key] = torch.from_numpy(filtered_np).to(value.device)
                            except Exception:
                                filtered_batch[key] = np.array(value)[class_mask]
                    elif isinstance(value, np.ndarray) or isinstance(value, list):
                        filtered_batch[key] = np.array(value)[class_mask]
                    else:
                        # unknown/unsupported type: keep as is
                        filtered_batch[key] = value
        return filtered_batch

    def get_audio_embeddings_for_generation(self, num_embeddings=1, from_train=False):
        if from_train:
            dataloader = DataLoader(self.node_data.train_dataset, batch_size=num_embeddings, shuffle=False)
        else:
            dataloader = DataLoader(self.node_data.test_dataset, batch_size=num_embeddings, shuffle=False)

        with torch.no_grad():

            for batch_idx, samples in enumerate(dataloader):
                # Move data to device
                if 'audio' in samples and isinstance(samples['audio'], torch.Tensor):
                    audio_data = samples['audio'].to(self.device)
                else:
                    raise ValueError("Audio data not found in the batch samples")

                # Forward pass through feature extractor and AST model
                if isinstance(audio_data, torch.Tensor) and isinstance(self.audio2image_model.feature_extractor, ASTFeatureExtractor):
                    audio_data = audio_data.to('cpu').numpy()

                audio_inputs = self.audio2image_model.feature_extractor(
                    audio_data,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                ).input_values.to(self.device, self.audio2image_model.torch_dtype)

                ast_model = self.audio2image_model.ast_model
                ast_model.to(self.device)
                ast_model.eval()

                audio_embeddings = ast_model(audio_inputs).last_hidden_state  # (batch, seq_len, feature_dim)
                embeddings = {}
                for module_name in self.adapters.keys():
                    adapter = self.adapters[module_name].to(self.device)
                    adapter.eval()
                    output = adapter(audio_embeddings)
                    embeddings[module_name] = output
                
                embeddings['class_name'] = samples.get('class_name', None)

        print(f"Node {self.id} - Retrieved {audio_embeddings.shape[0]} audio embeddings for generation")

        return embeddings



    def compute_class_mean_audio_features(self, dataloader=None, use_train=True, device=None):
        if device is None:
            device = self.device

        # Get dataloader
        dataset = None
        if dataloader is None:
            if use_train:
                dataloader = self.load_train_data()
                dataset = self.node_data.train_dataset
            else:
                dataloader = self.load_test_data()
                dataset = self.node_data.test_dataset

        if isinstance(dataset, Subset):
            dataset = dataset.dataset  # Unwrap Subset to get original dataset

        if dataloader is None:
            print(f"Warning: Node {self.id} - No dataloader available for class mean computation")
            return {}

        # Get AST model
        ast_model = self.audio2image_model.ast_model
        feature_extractor = self.audio2image_model.feature_extractor

        if ast_model is None:
            print(f"Warning: Node {self.id} - AST model not available")
            return {}

        # Move model to device and set to eval mode
        ast_model.to(device)
        ast_model.eval()

        # Storage for features per class
        class_features = defaultdict(list)  # {class_name: [features]}
        total_samples = 0

        print(f"Node {self.id} - Computing class mean features from {'train' if use_train else 'test'} data")

        with torch.no_grad():
            dataset_classes = dataset.active_classes

            for batch_idx, batch_data in enumerate(dataloader):
                # Extract audio and labels

                if 'audio' in batch_data:
                    audio_data = batch_data['audio']
                else:
                    print(f"Warning: No 'audio' key in batch data")
                    continue

                # Extract labels
                if 'label' in batch_data:
                    labels = batch_data['label']
                elif 'class_name' in batch_data:
                    # Map class names to IDs if needed
                    labels = batch_data.get('class_id', batch_data['class_name'])
                else:
                    print(f"Warning: No label information in batch {batch_idx}")
                    continue

                # Preprocess audio if needed
                if isinstance(audio_data, torch.Tensor):
                    # Check if we need to use feature extractor
                    if isinstance(feature_extractor, ASTFeatureExtractor):
                        audio_data = audio_data.cpu().numpy()
                        # Extract features using AST feature extractor
                        audio_inputs = feature_extractor(
                            audio_data,
                            sampling_rate=16000,
                            return_tensors="pt",
                            padding=True
                        ).input_values.to(device, self.audio2image_model.torch_dtype)
                    else:
                        audio_inputs = audio_data.to(device)
                else:
                    # Already numpy, use feature extractor
                    audio_inputs = feature_extractor(
                        audio_data,
                        sampling_rate=16000,
                        return_tensors="pt",
                        padding=True
                    ).input_values.to(device, self.audio2image_model.torch_dtype)

                # Forward through AST model to get features
                ast_output = ast_model(audio_inputs).last_hidden_state  # (batch, seq_len, feature_dim)

                # Get class names for this batch
                batch_class_names = batch_data.get('class_name', [])

                # Process each sample in the batch
                for idx, class_name in enumerate(batch_class_names):
                    # Store the output for this specific sample
                    class_features[class_name].append(ast_output[idx])
                    total_samples += 1

                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches, {total_samples} samples")

        print(f"Node {self.id} - Finished processing batches")
        print(f"  Classes found: {list(class_features.keys())}")
        for class_name, features_list in class_features.items():
            print(f"    Class {class_name}: {len(features_list)} samples")

        return class_features

    def get_class_prototypes(self, dataloader=None, use_train=True, normalize=False, device=None):
        """
        Get class prototypes (normalized mean features per class).

        This is a wrapper around compute_class_mean_features that optionally normalizes
        the prototypes, which is useful for prototypical networks and metric learning.

        Args:
            dataloader: DataLoader to use
            use_train: Whether to use training data
            normalize: If True, L2-normalize the prototype vectors
            device: Device to run on

        Returns:
            dict: Dictionary mapping class_id -> prototype (torch.Tensor)
        """
        class_means = self.compute_class_mean_audio_features(dataloader, use_train, device)

        if normalize:
            # L2 normalize each prototype
            for class_id in class_means:
                prototype = class_means[class_id]
                # L2 normalization
                norm = torch.norm(prototype, p=2)
                if norm > 0:
                    class_means[class_id] = prototype / norm

            print(f"Node {self.id} - Normalized {len(class_means)} class prototypes")

        return class_means

    def compute_per_class_adapter_outputs(self, dataloader=None, use_train=True, device=None):
        """
        Compute outputs for all adapters, storing ALL outputs per class (not just mean).

        Returns:
            dict: {adapter_name: {class_name: [list of outputs]}}
        """
        if device is None:
            device = self.device

        # Get AST features per class
        ast_features_per_class = self.compute_class_mean_audio_features(
            dataloader=dataloader,
            use_train=use_train,
            device=device
        )

        # Storage for adapter outputs
        adapter_outputs = {}

        # Process through each adapter
        for adapter_name, adapter in self.adapters.items():
            print(f"Node {self.id} - Computing {adapter_name} outputs for all classes")
            adapter.to(device)
            adapter.eval()

            adapter_outputs[adapter_name] = {}

            with torch.no_grad():
                for class_name, features_list in ast_features_per_class.items():
                    # Process all features for this class through the adapter
                    class_outputs = []
                    for features in features_list:
                        # features shape: (seq_len, feature_dim)
                        # Add batch dimension if needed
                        if features.dim() == 2:
                            features = features.unsqueeze(0)  # (1, seq_len, feature_dim)

                        features = features.to(device)
                        output = adapter(features)
                        # Remove batch dimension and store
                        class_outputs.append(output.squeeze(0))

                    adapter_outputs[adapter_name][class_name] = class_outputs
                    print(f"  - {adapter_name} Class {class_name}: {len(class_outputs)} outputs")

        return adapter_outputs

    def update_per_class_mean_output(self, use_train=True, device=None):
        print(f"Node {self.id} - Updating per-class mean output")

        # Compute class means as dictionary (this stores ALL outputs per class)
        self.per_class_outputs = self.compute_class_mean_audio_features(
            dataloader=None,
            use_train=use_train,
            device=device
        )

        # Calculate mean for each class
        self.per_class_outputs_mean = {}

        for class_name, features_list in self.per_class_outputs.items():
            if len(features_list) > 0:
                # Stack all features and compute mean
                self.per_class_outputs_mean[class_name] = torch.mean(torch.stack(features_list), dim=0)
            else:
                print(f"Warning: No features for class {class_name}")

        present_classes = len(self.per_class_outputs_mean)
        total_samples = sum(len(features_list) for features_list in self.per_class_outputs.values())
        print(f"Node {self.id} - Stored mean output for {present_classes} classes ({total_samples} total samples)")

        return self.per_class_outputs_mean

    def compute_adapter_mean_outputs(self, adapter_outputs):
        """
        Compute mean outputs for each adapter and class from all stored outputs.

        Args:
            adapter_outputs: dict from compute_per_class_adapter_outputs()
                            {adapter_name: {class_name: [list of outputs]}}

        Returns:
            dict: {adapter_name: {class_name: mean_output}}
        """
        adapter_means = {}

        for adapter_name, class_outputs_dict in adapter_outputs.items():
            adapter_means[adapter_name] = {}

            for class_name, outputs_list in class_outputs_dict.items():
                if len(outputs_list) > 0:
                    # Stack all outputs and compute mean
                    adapter_means[adapter_name][class_name] = torch.mean(
                        torch.stack(outputs_list), dim=0
                    )
                    print(f"  - {adapter_name} Class {class_name}: mean computed from {len(outputs_list)} samples")
                else:
                    print(f"Warning: No outputs for {adapter_name} class {class_name}")

        return adapter_means

    def get_per_class_mean_output_dict(self):
        """
        Get per_class_mean_output as a dictionary (compact representation).

        Returns:
            dict: {class_id: mean_output} for classes present in the node
        """
        if self.per_class_outputs_mean is None:
            return {}

        result = {}
        for class_id, mean_output in enumerate(self.per_class_outputs_mean):
            if mean_output is not None:
                result[class_id] = mean_output

        return result

    def get_training_adapter_outputs(self, class_name=None, adapter_name=None):
        """
        Get adapter outputs collected during training.

        Args:
            class_name: Optional. If specified, returns outputs only for this class.
                       If None, returns outputs for all classes.
            adapter_name: Optional. If specified (along with class_name), returns outputs
                         only for this adapter. If None, returns outputs for all adapters.

        Returns:
            - If both class_name and adapter_name specified: list of outputs for that class/adapter
            - If only class_name specified: {adapter_name: [outputs]} for that class
            - If neither specified: {class_name: {adapter_name: [outputs]}} for all

        Examples:
            # Get all outputs
            all_outputs = client.get_training_adapter_outputs()

            # Get outputs for specific class
            dog_outputs = client.get_training_adapter_outputs('dog')

            # Get outputs for specific class and adapter
            dog_clip_outputs = client.get_training_adapter_outputs('dog', 'clip')
        """
        if self.training_adapter_outputs_all is None:
            print(f"Warning: No training adapter outputs available. Training may not have completed yet.")
            return None

        # Return all outputs
        if class_name is None:
            return self.training_adapter_outputs_all

        # Return outputs for specific class
        if class_name not in self.training_adapter_outputs_all:
            print(f"Warning: Class '{class_name}' not found in training outputs")
            return None

        class_outputs = self.training_adapter_outputs_all[class_name]

        # Return all adapters for this class
        if adapter_name is None:
            return class_outputs

        # Return outputs for specific adapter
        if adapter_name not in class_outputs:
            print(f"Warning: Adapter '{adapter_name}' not found for class '{class_name}'")
            return None

        return class_outputs[adapter_name]

    def get_training_adapter_means(self, class_name=None, adapter_name=None):
        """
        Get mean adapter outputs computed from training.

        Args:
            class_name: Optional. If specified, returns mean only for this class.
            adapter_name: Optional. If specified (along with class_name), returns mean
                         only for this adapter.

        Returns:
            Mean tensor(s) based on the specified filters.

        Examples:
            # Get all means
            all_means = client.get_training_adapter_means()

            # Get means for specific class
            dog_means = client.get_training_adapter_means('dog')

            # Get mean for specific class and adapter
            dog_clip_mean = client.get_training_adapter_means('dog', 'clip')
        """
        if self.training_adapter_outputs_mean is None:
            print(f"Warning: No training adapter means available. Training may not have completed yet.")
            return None

        # Return all means
        if class_name is None:
            return self.training_adapter_outputs_mean

        # Return means for specific class
        if class_name not in self.training_adapter_outputs_mean:
            print(f"Warning: Class '{class_name}' not found in training means")
            return None

        class_means = self.training_adapter_outputs_mean[class_name]

        # Return all adapters for this class
        if adapter_name is None:
            return class_means

        # Return mean for specific adapter
        if adapter_name not in class_means:
            print(f"Warning: Adapter '{adapter_name}' not found for class '{class_name}'")
            return None

        return class_means[adapter_name]


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