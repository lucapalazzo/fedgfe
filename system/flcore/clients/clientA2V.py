from collections import defaultdict
import os
import copy
import sys
import sklearn
import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader, Subset

import wandb

sys.path.append('/home/lpala/fedgfe/system/flcore/trainmodel/Audio2Visual_NoData')

from flcore.clients.clientbase import Client
from datautils.node_dataset import NodeData
from sklearn.preprocessing import label_binarize
from sklearn.metrics import balanced_accuracy_score
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from torchvision.ops.focal_loss import sigmoid_focal_loss
from torchvision import transforms
from utils.check_parameters import check_optimizer_params, print_model_gradients_status
from utils.node_metric import NodeMetric
from utils.check_loss_optimizer import check_params_graph_vs_optimizer


from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, ConstantLR, CosineAnnealingWarmRestarts

from modelutils.model_stats import count_changed_weights

import torch.nn.functional as F

from tqdm import tqdm

from torchviz import make_dot

from collections import Counter
from transformers import ASTModel, ASTFeatureExtractor

from flcore.trainmodel.Audio2Visual_NoData.src.models.audio2image import Audio2Image, SDImageModel, ImageDiffusion
from flcore.trainmodel.downstreamsinestesiaadapters import DownstreamSinestesiaAdapters

# Import generators
from flcore.trainmodel.generators import ConditionedVAEGenerator, VAEGenerator, VAELoss, GANGenerator, GANDiscriminator, MultiModalVAEGenerator

import logging

logger = logging.getLogger(__name__)

class clientA2V(Client):
    def __init__(self, args, node_id, node_config=None, global_model=None, dataset=None, store_audio_embedding=False, **kwargs):
        super().__init__(args, node_id, None, None, dataset = dataset, **kwargs)

        self.logger = logging.getLogger(f"{__name__}_{node_id}")

        self.id = node_id
        self.learning_rate = args.local_learning_rate
        self.train_loader = None
        self.no_wandb = args.no_wandb
        self.train_dataloader = None
        self.test_dataloader = None
        self.global_model = global_model
        self.global_rounds = args.global_rounds
        self.server = None  # Reference to server, will be set later


        self.feda2v_config = getattr(self.args.json_config, 'feda2v', None)
        self.t5_adapter_learning_rate = 0.001
        self.clip_adapter_learning_rate = 0.001
        self.t5_adapter_weight_decay = 0.0001
        self.clip_adapter_weight_decay = 0.0001
        self.adapters_learning_rate = 0.001
        self.adapters_weight_decay = 0.0001

        self.adapter_dropout = getattr(node_config, "adapter_dropout", None )
 

        if self.feda2v_config is not None:
            self.t5_adapter_learning_rate = self.feda2v_config.get('t5_adapter_learning_rate', self.t5_adapter_learning_rate)
            self.clip_adapter_learning_rate = self.feda2v_config.get('clip_adapter_learning_rate', self.clip_adapter_learning_rate)
            self.t5_adapter_weight_decay = self.feda2v_config.get('t5_adapter_weight_decay', self.t5_adapter_weight_decay)
            self.clip_adapter_weight_decay = self.feda2v_config.get('clip_adapter_weight_decay', self.clip_adapter_weight_decay)
            self.adapters_learning_rate = self.feda2v_config.get('adapters_learning_rate', self.adapters_learning_rate)
            self.adapters_weight_decay = self.feda2v_config.get('adapters_weight_decay', self.adapters_weight_decay)
            self.adapters_learning_rate_schedule = self.feda2v_config.get('adapters_learning_rate_schedule', False)
            self.t5_adapter_learning_rate_schedule = self.feda2v_config.get('t5_adapter_learning_rate_schedule', False)
            self.clip_adapter_learning_rate_schedule = self.feda2v_config.get('clip_adapter_learning_rate_schedule', False)

        if node_config is not None:
            self.t5_adapter_learning_rate = node_config.t5_adapter_learning_rate if node_config.t5_adapter_learning_rate is not None else self.t5_adapter_learning_rate
            self.t5_adapter_weight_decay = node_config.t5_adapter_weight_decay if node_config.t5_adapter_weight_decay is not None else self.t5_adapter_weight_decay
            self.clip_adapter_learning_rate = node_config.clip_adapter_learning_rate if node_config.clip_adapter_learning_rate is not None else self.clip_adapter_learning_rate
            self.clip_adapter_weight_decay = node_config.clip_adapter_weight_decay if node_config.clip_adapter_weight_decay is not None else self.clip_adapter_weight_decay

        self.store_audio_embedding = self.args.json_config.feda2v.store_audio_embeddings

        self.save_generated_images_splits = self.feda2v_config.save_generated_images_splits if self.feda2v_config else []

        # Get use_cls_token_only configuration
        # Priority: node_config > feda2v global config > default (False)
        global_use_cls_token = self.feda2v_config.get('use_cls_token_only', False) if self.feda2v_config else False
        node_use_cls_token = getattr(node_config, 'use_cls_token_only', None)

        if node_use_cls_token is not None:
            # Node-specific configuration takes priority
            self.use_cls_token_only = node_use_cls_token
            logger.info(f"Node {node_id}: Using node-specific use_cls_token_only={self.use_cls_token_only}")
        else:
            # Fall back to global configuration
            self.use_cls_token_only = global_use_cls_token
            logger.info(f"Node {node_id}: Using global use_cls_token_only={self.use_cls_token_only}")

        # Create local instance using DownstreamSinestesiaAdapters without diffusion

        self.model = DownstreamSinestesiaAdapters(
            args=args,
            wandb_log=not args.no_wandb,
            device=self.device,
            use_classifier_loss=False,
            diffusion_type=self.global_model.diffusion_type,
            enable_diffusion=False,  # Explicitly disable diffusion for local nodes
            use_cls_token_only=self.use_cls_token_only,
            adapter_dropout=self.adapter_dropout
        )

        # Share the AST model from global model (read-only, not trainable)
        self.model.audio2image_model.ast_model = self.global_model.audio2image_model.ast_model.to(self.device)
        self.model.audio2image_model.feature_extractor = self.global_model.audio2image_model.feature_extractor
        self.audio2image_model = self.model.get_audio2image_model()


        self.epoch_audio_embedding_cache = {}

        if 'data_log' in kwargs:
            self.data_log = kwargs['data_log']
        else:
            self.data_log = None

        if node_config.dataset != None:
            self.dataset_name = node_config.dataset
        else:
            self.dataset_name = args.dataset

        self.dataset = self.dataset_name

        self.node_data = NodeData(args, node_id, dataset=dataset)
        self.node_data.dataset_name = self.dataset_name

        self.experiment_config = getattr(args.json_config, 'experiment', None)

        self.optimize_memory_usage = getattr(self.experiment_config, 'optimize_memory_usage', False )

        self.train_optimizer = None
        self.finetuning_optimizer = None
        self.train_optimizers = {}
        self.finetuning_optimizers = {}
        self.adapters_learning_rate_scheduler = {}

        self.model_optimizer = args.model_optimizer

        self.defined_train_metrics = { 'loss_at_start': None, 'loss_at_end': None, 'loss_reduction': None }
        self.defined_test_metrics = {}
        self.use_saved_audio_embeddings = getattr(args, 'use_saved_audio_embeddings', False)

        self.text_losses_summed = False

        # Audio2Visual specific initialization
        self.diffusion_type = getattr(node_config, 'diffusion_type', 'sd')  # 'sd', 'flux', or 'cogx'
        self.use_act_loss = getattr(args, 'use_act_loss', True)
        self.audio_model_name = getattr(args, 'audio_model_name', "MIT/ast-finetuned-audioset-10-10-0.4593")
        self.img_pipe_name = getattr(args, 'img_pipe_name', "runwayml/stable-diffusion-v1-5")
        self.img_lcm_lora_id = getattr(args, 'img_lcm_lora_id', "latent-consistency/lcm-lora-sdv1-5")

        self.per_class_outputs = None  # Will be computed at the end of each round
        self.per_class_outputs_mean = None  # Will be computed at the end of each round

        # Storage for adapter outputs collected during training
        self.training_adapter_outputs_all = None   # All outputs: {class_name: {adapter_name: [outputs]}}
        self.training_adapter_outputs_mean = None  # Mean outputs: {class_name: {adapter_name: mean_tensor}}
        self.per_class_embeddings = None

        self.audio_encoder_output = None

        self.adapters = self.model.adapters if hasattr(self.model, 'adapters') else {}
        self.adapters_modules = self.model.adapters_modules if hasattr(self.model, 'adapters_modules') else None

        # Generator configuration
        self.use_generator = getattr(self.feda2v_config, 'use_generator', False) if self.feda2v_config else False
        self.generator_type = getattr(self.feda2v_config, 'generator_type', 'vae') if self.feda2v_config else 'vae'
        self.use_conditioned_vae = getattr(self.feda2v_config, 'use_conditioned_vae', True) if self.feda2v_config else True
        self.generator_training_mode = getattr(self.feda2v_config, 'generator_training_mode', False) if self.feda2v_config else False
        self.generator_only_mode = getattr(self.feda2v_config, 'generator_only_mode', False) if self.feda2v_config else False
        self.use_pretrained_generators = getattr(self.feda2v_config, 'use_pretrained_generators', False) if self.feda2v_config else False

        # Generator granularity: 'unified', 'per_class', 'per_group'
        self.generator_granularity = getattr(self.feda2v_config, 'generator_granularity', 'unified') if self.feda2v_config else 'unified'
        self.generator_class_groups = getattr(self.feda2v_config, 'generator_class_groups', None) if self.feda2v_config else None

        # Per-node generator configuration overrides
        # 1. Override granularity for this specific node
        node_granularity = getattr(node_config, 'generator_granularity', None)
        if node_granularity:
            print(f"[Client {self.id}] Overriding generator_granularity: {self.generator_granularity} → {node_granularity}")
            self.generator_granularity = node_granularity

        # 2. Specify which classes to train the generator on (must be subset of selected_classes)
        self.node_generator_train_classes = getattr(node_config, 'generator_train_classes', None)
        if self.node_generator_train_classes:
            # Validate that train_classes is a subset of selected_classes
            node_classes = set(self.node_data.dataset.selected_classes if hasattr(self.node_data, 'dataset') else [])
            train_classes = set(self.node_generator_train_classes)
            if not train_classes.issubset(node_classes):
                invalid = train_classes - node_classes
                print(f"[Client {self.id}] Warning: generator_train_classes contains invalid classes: {invalid}")
                print(f"[Client {self.id}] Will use only valid classes from selected_classes")
                self.node_generator_train_classes = list(train_classes & node_classes)

        # 3. Specify class group for per_group mode
        self.node_generator_class_group = getattr(node_config, 'generator_class_group', None)

        # Generator checkpoint configuration with flexible naming
        self.generator_checkpoint_dir = getattr(self.feda2v_config, 'generator_checkpoint_dir', 'checkpoints/generators') if self.feda2v_config else 'checkpoints/generators'
        self.generator_checkpoint_base_name = getattr(self.feda2v_config, 'generator_checkpoint_base_name', 'client_generator') if self.feda2v_config else 'client_generator'

        self.generator_training_epochs = getattr(self.feda2v_config, 'generator_training_epochs', 5) if self.feda2v_config else 5
        self.generator_augmentation = getattr(self.feda2v_config, 'generator_augmentation', True) if self.feda2v_config else True
        self.generator_augmentation_noise = getattr(self.feda2v_config, 'generator_augmentation_noise', 0.1) if self.feda2v_config else 0.1
        self.synthetic_samples_per_class = getattr(self.feda2v_config, 'synthetic_samples_per_class', 5) if self.feda2v_config else 5

        # Target sequence length for generated samples (None = use generator's default of 4, 1214 = full AST output length)
        # Sequence length for generator training (should be small to fit in GPU memory, e.g., 4 or 8)
        self.generator_training_sequence_length = self.feda2v_config.get('generator_training_sequence_length', 4) if self.feda2v_config else 4
        # Target sequence length for generation/output (can be full AST length, e.g., 1214)
        self.generator_output_sequence_length = self.feda2v_config.get('generator_output_sequence_length', None) if self.feda2v_config else None

        # Initialize generators (can be single or multiple based on granularity)
        self.prompt_generator = None  # For unified generator
        self.prompt_generators = {}   # For per_class or per_group generators: {class_name/group_name: generator}
        self.prompt_generator_clip = None
        self.prompt_generator_t5 = None
        self.generator_optimizer = None
        self.generator_optimizers = {}  # For multiple generators
        self.generator_loss_fn = None

        # Initialize generators if needed
        if self.use_generator or self.generator_training_mode or self.generator_only_mode:
            self.initialize_generators()

        # Load pretrained generators if specified
        if self.use_pretrained_generators and not self.generator_training_mode:
            self.load_generator_checkpoint()

        self._move_to_gpu(self.device)

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

        # Generator-only mode: skip adapter training, only train generators
        if self.generator_only_mode:
            print(f"\n{'='*60}")
            print(f"[Client {self.id}] GENERATOR-ONLY MODE")
            print(f"{'='*60}")
            print(f"Skipping adapter training, will only train generator(s)")
            print(f"Granularity: {self.generator_granularity}")

            # Still need to load data and run forward passes to collect adapter outputs
            # But we use frozen adapters (from global model)
            node_trainloader = self.load_train_data()
            self._move_to_gpu(self.device)

            self.train_node_generator()

            print(f"{'='*60}\n")
            return

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

            self.setup_learning_rate_scheduler(self.global_rounds)

            if self.no_wandb == False:
                wandb.watch(self.audio2image_model, log='all', log_freq=100, criterion=None, log_graph=False, idx=self.id)

            self.print_optimizer_info(self.id)

        # self.model.train()

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
        logger.debug(f"*** Node {self.id} memory after training and moving to CPU {torch.cuda.memory_allocated(device)//1024**2} MB")

    def update_local_adapters(self, adapters = None, projections = None):
        for module_name, module in self.adapters.items():
            if module_name in adapters:
                local_adapter = self.adapters[module_name]
                global_adapter_state_dict = adapters[module_name].state_dict()
                local_adapter.load_state_dict(global_adapter_state_dict)

    def audio_emebedding_cache_store(self, samples, outputs):
        audio_embeddings_batch = outputs['audio_embeddings']
        file_ids = samples.get('file_id', None)

        if file_ids is not None:
            # Ottieni riferimento al dataset sottostante
            train_dataset = self.node_data.train_dataset
            # Unwrap Subset se necessario
            if hasattr(train_dataset, 'dataset'):
                base_dataset = train_dataset.dataset
            else:
                base_dataset = train_dataset

            #  audio_embs se non esiste
            if not hasattr(base_dataset, 'audio_embs') or base_dataset.audio_embs is None:
                base_dataset.audio_embs = {}

            for idx, file_id in enumerate(file_ids):
                base_dataset.audio_embs[file_id] = audio_embeddings_batch[idx].detach().cpu()
                self.epoch_audio_embedding_cache[file_id] = audio_embeddings_batch[idx].detach().cpu()
        logger.debug(f"Node {self.id}: stored {len(file_ids)} audio embeddings int dataset  {len(base_dataset.audio_embs)})")

    def audio_embedding_cache_load(self, samples ):
        audio_embedding = None
        file_ids = samples.get('file_id', None)
        if file_ids is not None:
            train_dataset = self.node_data.train_dataset
            if hasattr(train_dataset, 'dataset'):
                base_dataset = train_dataset.dataset
            else:
                base_dataset = train_dataset

            if hasattr(base_dataset, 'audio_embs') and base_dataset.audio_embs is not None:
                cached_embeddings = []
                all_cached = True

                for file_id in file_ids:
                    if file_id in base_dataset.audio_embs:
                        cached_embeddings.append(base_dataset.audio_embs[file_id])
                    else:
                        all_cached = False
                        break

                if all_cached:
                    # Usa gli embeddings dalla cache
                    audio_embedding = torch.stack(cached_embeddings)
                    # print(f"Node {self.id} Epoch {epoch+1} 
        return audio_embedding
        
    def train_a2v(self, epochs, dataloader, client_device=None):
        device = client_device if client_device is not None else self.device

        self.model.train()

        epoch_audio_embedding_cache = {}

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            training_adapter_outputs = defaultdict(lambda: defaultdict(list))

            # Create progress bar for batches
            pbar = tqdm(enumerate(dataloader), total=len(dataloader),
                       desc=f"Node {self.id} Epoch {epoch+1}/{epochs}",
                       unit="batch")

            # for batch_idx, (audio_data, text_embeddings) in enumerate(dataloader):
            for batch_idx, samples in pbar:
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

                audio_filename = samples.get('audio_filename', None)
                audio_embedding = None

                if audio_embedding is None:
                    audio_embedding = samples.get('audio_emb', None)

                if epoch > 0 or self.round > 1 and audio_embedding is None:
                    audio_embedding = self.audio_embedding_cache_load( samples )
                    if audio_embedding is not None:
                        audio_embedding = audio_embedding.to(device)
               

                if len(self.train_optimizers) > 0:
                    for optimizer_name, optimizer in self.train_optimizers.items():
                        optimizer.zero_grad()
                else:
                    self.optimizer.zero_grad()

                if audio_embedding is None and isinstance(audio_data, torch.Tensor) and isinstance(self.audio2image_model.feature_extractor, ASTFeatureExtractor):
                    audio_data = audio_data.to('cpu').numpy()
                else:
                    audio_data = None

                outputs = self.model( audio_data,
                                        img_target_prompt_embeds=target_prompt_embeds,
                                        img_target_pooled_prompt_embeds=target_pooled_prompt_embeds,
                                        audio_embedding=audio_embedding
                                    )

                outputs['class_name'] = samples.get('class_name', None)

                # Salva gli audio embeddings nel dataset per riutilizzo nelle epoche successive (epoch 0)
                if epoch == 0 and self.round == 1 and 'audio_embeddings' in outputs:
                    self.audio_emebedding_cache_store( samples, outputs )

                if self.store_audio_embedding:
                    outputs['audio_filename'] = samples.get('audio_filename', None)
                    self.store_audio_embeddings(audio_data, outputs)

                if epoch == epochs - 1:
                    training_adapter_outputs = self.store_adapters_output_per_class ( outputs, training_adapter_outputs )

                losses = outputs['text_loss']
                losses_dict = {}  # Dictionary to hold individual losses for display

                if self.text_losses_summed:
                    loss = torch.tensor(0.0)
                    if outputs['text_loss'] is not None:
                        if isinstance(outputs['text_loss'], tuple):
                            # Assume order: [clip_loss, t5_loss] or similar
                            loss_names = ['clip', 't5']  # Adapter names
                            for idx, l in enumerate(outputs['text_loss']):
                                loss = loss.to(l.device)
                                loss += l
                                # Store individual loss values
                                if idx < len(loss_names):
                                    losses_dict[loss_names[idx]] = f"{l.item():.3f}"
                            losses_dict['total'] = f"{loss.item():.3f}"
                        elif isinstance(outputs['text_loss'], dict):
                            for name, l in outputs['text_loss'].items():
                                loss = loss.to(l.device)
                                loss += l
                                # Store individual loss values
                                losses_dict[name] = f"{l.item():.3f}"
                            losses_dict['total'] = f"{loss.item():.3f}"
                    loss.backward()
                    epoch_loss += loss.item()
                else:
                    adapters_loss = outputs['text_loss']
                    losses_count = len(losses)
                    loss_names = ['clip', 't5']  # Adapter names
                    for loss_names, loss in adapters_loss.items():
                        loss.backward()
                        epoch_loss += loss.item()
                        losses_dict[loss_names] = f"{loss.item():.3f}"


                # Gradient clipping to stabilize training
                if len(self.train_optimizers) > 0:
                    grad_norms = {}
                    for optimizer_name, optimizer in self.train_optimizers.items():
                        # Clip gradients for this optimizer's parameters
                        params = [p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
                        if len(params) > 0:
                            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                            grad_norms[optimizer_name] = grad_norm.item()
                        optimizer.step()

                    # Log gradient norms occasionally for debugging
                    if num_batches % 50 == 0 and epoch == 0:
                        grad_info = " ".join([f"{k}_grad:{v:.4f}" for k, v in grad_norms.items()])
                        logger.debug(f"Node {self.id} Batch {num_batches}: {grad_info}")
                else:
                    # Clip gradients for all model parameters
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                num_batches += 1

                # Update progress bar with loss information (clip, t5, and total)
                pbar.set_postfix(losses_dict)
            for learning_rate_scheduler_name, learning_rate_scheduler in self.adapters_learning_rate_scheduler.items():
                learning_rate_scheduler.step(self.round*self.local_epochs+epoch)

            # Close progress bar and print summary
            pbar.close()

            # if num_batches > 0:
            #     avg_loss = epoch_loss / num_batches
            #     tqdm.write(f"Node {self.id} Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.3f}")
            
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

        # Train generator if in generator training mode
        if self.generator_training_mode and self.prompt_generator is not None:
            self.train_node_generator()

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
        
        self.model.train()
        return 0.0

    def print_optimizer_info(self, client_id):
        if self.train_optimizer is None:
            print(f"Node {client_id}: No optimizer initialized")
            return
        
        for optimizer_name, optimizer in self.train_optimizers.items():
            print(f"Node {client_id} optimizer '{optimizer_name}': {optimizer}")

            for param_group_index, param_group in enumerate(optimizer.param_groups):
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
        trainable_params_dict = {}

        for adapter_name, adapter in self.adapters.items():
            trainable_params_dict[adapter_name] = adapter.parameters()

        if len(trainable_params) == 0:
            raise ValueError(f"Node {self.id}: No trainable parameters found in local adapters!")

        # Create optimizer
        if self.model_optimizer.lower() == "adamw":
            self.train_optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.learning_rate,
                weight_decay=self.adapters_weight_decay
            )
            for module_name, params in trainable_params_dict.items():
                learning_rate = self.adapters_learning_rate
                weight_decay = self.adapters_weight_decay
                if module_name == 't5':
                    learning_rate = self.t5_adapter_learning_rate
                    weight_decay = self.t5_adapter_weight_decay
                elif module_name == 'clip':
                    learning_rate = self.clip_adapter_learning_rate
                    weight_decay = self.clip_adapter_weight_decay
                self.train_optimizers[module_name] = torch.optim.AdamW(
                    params,
                    lr=learning_rate,
                    weight_decay=weight_decay
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
        self.optimizer = self.train_optimizer
        self.optimizers = self.train_optimizers

    def setup_learning_rate_scheduler(self, rounds):
        for module_name, optimizer in self.train_optimizers.items():
            if module_name == 't5' and self.t5_adapter_learning_rate_schedule:
                optimizer = self.train_optimizers[module_name]
                self.adapters_learning_rate_scheduler[module_name] = CosineAnnealingLR( optimizer=optimizer,
                                                                                        T_max=rounds,
                                                                                        eta_min=1e-6,
                                                                                        verbose=True)
            elif module_name == 'clip' and self.clip_adapter_learning_rate_schedule:
                optimizer = self.train_optimizers[module_name]
                for param_group in optimizer.param_groups:
                    param_group['initial_lr'] = self.clip_adapter_learning_rate
                self.adapters_learning_rate_scheduler[module_name] = CosineAnnealingWarmRestarts(
                                                                    optimizer=optimizer,
                                                                    T_0=self.local_epochs * self.global_rounds // 10,
                                                                    T_mult=1,
                                                                    last_epoch=self.global_rounds * self.local_epochs // 2,
                                                                    eta_min=1e-8,
                                                                    verbose=True
                                                                )
               
            
        # self.scheduler = self.optimizermanager.setup_learning_rate_scheduler(
        #     optimizer=self.optimizer,
        #     rounds=rounds,
        #     use_scheduler=self.learning_rate_schedule
        # )

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

    def test_metrics(self, test_client=None, on_train=False, use_generated_images=False, generation_split='test', round_num=None):
        """
        Calculate test metrics for the client.

        Args:
            test_client: Optional client to test (for cross-client evaluation)
            on_train: If True, use training data instead of test data
            use_generated_images: If True, generate images and compute metrics on them
            generation_split: Which split to use for image generation ('train', 'test', 'val', 'all')
            round_num: Optional round number for naming generated images

        Returns:
            NodeMetric object with computed metrics
        """

        if round_num == None:
            round_num = self.round

        if use_generated_images:
            # Generate images from the specified split
            print(f"Client {self.id}: Generating images from split '{generation_split}' for metrics computation")
            generated_images = self.generate_images(split=generation_split, round_num=round_num)

            if not generated_images or all(v is None or len(v) == 0 for v in generated_images.values()):
                logger.warning(f"Client {self.id}: No images were generated, falling back to standard metrics")
                use_generated_images = False
            else:
                # Compute metrics from generated images
                node_metrics = self.test_node_metrics_from_images(generated_images)

                # Cleanup temporary images that shouldn't be kept
                self.cleanup_temporary_images(generated_images)

                if node_metrics is not None:
                    return node_metrics
                else:
                    logger.warning(f"Client {self.id}: Failed to compute metrics from generated images, falling back to standard metrics")
                    use_generated_images = False

        # Standard metrics computation (fallback or when use_generated_images=False)
        if on_train:
            dataloader = self.load_train_data()
        else:
            dataloader = self.load_test_data()

        self.model.eval()
        if dataloader is not None:
            node_metrics = self.model.train_metrics(dataloader, audio2image_only=True)

        node_metrics.phase = NodeMetric.Phase.TEST

        return node_metrics
      
    def train_metrics(self, trainloader=None,split='train'):
        if trainloader is None:
            if split == 'train':
                trainloader = self.load_train_data()
            elif split == 'val':
                trainloader = self.load_val_data()
            elif split == 'test':
                trainloader = self.load_test_data()

        node_metrics = NodeMetric(phase=NodeMetric.Phase.TRAIN)
        node_metrics.define_metrics(self.model.defined_train_metrics, task_count=1)
        
        self.model.eval()
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
        if hasattr(self, 'optimizers') and self.optimizers is not None:
            for optimizer_name, optimizer in self.optimizers.items():
                if isinstance(optimizer, torch.optim.AdamW):
                    move_optimizer_state(optimizer, device)

        # Move generators and their optimizers if in generator training mode
        if self.generator_training_mode or self.generator_only_mode:
            # Move unified generator if exists
            if hasattr(self, 'prompt_generator') and self.prompt_generator is not None:
                self.prompt_generator.to(device)
                # Move unified generator optimizer
                if hasattr(self, 'generator_optimizer') and self.generator_optimizer is not None:
                    if isinstance(self.generator_optimizer, torch.optim.AdamW):
                        move_optimizer_state(self.generator_optimizer, device)

            # Move per-class or per-group generators if exist
            if hasattr(self, 'prompt_generators') and self.prompt_generators is not None:
                for gen_key, generator in self.prompt_generators.items():
                    if generator is not None:
                        generator.to(device)

            # Move per-class or per-group generator optimizers if exist
            if hasattr(self, 'generator_optimizers') and self.generator_optimizers is not None:
                for gen_key, optimizer in self.generator_optimizers.items():
                    if optimizer is not None and isinstance(optimizer, torch.optim.AdamW):
                        move_optimizer_state(optimizer, device)

            # Move GAN discriminator if exists (for GAN generator type)
            if hasattr(self, 'generator_discriminator') and self.generator_discriminator is not None:
                self.generator_discriminator.to(device)

    def _move_to_gpu(self, device, force = True ):
        if self.optimize_memory_usage or force or self.round <= 1:
            self.logger.debug(f"Node {self.id} moving to GPU: {device}")
            self._move_to_device(device)

    def _move_to_cpu(self, force = True):
        if self.optimize_memory_usage or force or self.round <= 1:
            self.logger.debug(f"Node {self.id} moving to CPU for memory optimization")
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

    def get_audio_embeddings_from_dataset(self, dataset):
        """Generate audio embeddings for all samples in a given dataset."""
        # Create dataloader for the entire dataset
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

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
                ast_model.eval()
                ast_model.to(self.device)

                audio_embeddings = ast_model(audio_inputs).last_hidden_state  # (batch, seq_len, feature_dim)
                embeddings = {}
                for module_name in self.adapters.keys():
                    adapter = self.adapters[module_name].to(self.device)
                    output = adapter(audio_embeddings)
                    embeddings[module_name] = output

                embeddings['class_name'] = samples.get('class_name', None)

        print(f"Node {self.id} - Retrieved {audio_embeddings.shape[0]} audio embeddings from dataset")

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

    def get_generator_for_class(self, class_name):
        """
        Get the appropriate generator for a given class.

        Args:
            class_name: Name of the class

        Returns:
            tuple: (generator, optimizer, generator_key) or (None, None, None) if not found
        """
        if not hasattr(self, 'class_to_generator_map'):
            print(f"[Client {self.id}] Warning: class_to_generator_map not initialized")
            return None, None, None

        if class_name not in self.class_to_generator_map:
            print(f"[Client {self.id}] Warning: class '{class_name}' not in mapping")
            return None, None, None

        gen_key = self.class_to_generator_map[class_name]

        if self.generator_granularity == 'unified':
            return self.prompt_generator, self.generator_optimizer, 'unified'
        else:
            if gen_key in self.prompt_generators:
                generator = self.prompt_generators[gen_key]
                optimizer = self.generator_optimizers.get(gen_key, None)
                return generator, optimizer, gen_key
            else:
                print(f"[Client {self.id}] Warning: generator '{gen_key}' not found")
                return None, None, None

    def _get_class_to_generator_mapping(self):
        """
        Build mapping from class name to generator identifier based on granularity.

        Returns:
            dict: {class_name: generator_key}
        """
        mapping = {}

        # Get classes for this node
        node_classes = []
        if hasattr(self, 'node_data') and hasattr(self.node_data, 'train_dataset'):
            train_dataset = self.node_data.train_dataset
            if hasattr(train_dataset, 'selected_classes'):
                node_classes = train_dataset.selected_classes
            elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'selected_classes'):
                node_classes = train_dataset.dataset.selected_classes

        # If node specifies which classes to train generators on, use only those
        if self.node_generator_train_classes:
            print(f"[Client {self.id}] Using node-specific generator training classes: {self.node_generator_train_classes}")
            generator_train_classes = self.node_generator_train_classes
        else:
            generator_train_classes = node_classes

        if self.generator_granularity == 'unified':
            # All classes (or subset if specified) use the same generator
            for class_name in generator_train_classes:
                mapping[class_name] = 'unified'

        elif self.generator_granularity == 'per_class':
            # Each class has its own generator (only for specified classes)
            for class_name in generator_train_classes:
                mapping[class_name] = class_name

        elif self.generator_granularity == 'per_group':
            # Map classes to groups
            if not self.generator_class_groups:
                print(f"[Client {self.id}] Warning: per_group mode but no class_groups defined. Falling back to unified.")
                for class_name in generator_train_classes:
                    mapping[class_name] = 'unified'
            else:
                # If node specifies a specific group, use only that group
                if self.node_generator_class_group:
                    print(f"[Client {self.id}] Using node-specific generator group: '{self.node_generator_class_group}'")
                    for class_name in generator_train_classes:
                        mapping[class_name] = self.node_generator_class_group
                else:
                    # Auto-detect groups from global class_groups mapping
                    for class_name in generator_train_classes:
                        # Find which group this class belongs to
                        group_found = False
                        for group_name, group_classes in self.generator_class_groups.items():
                            if class_name in group_classes:
                                mapping[class_name] = group_name
                                group_found = True
                                break

                        if not group_found:
                            print(f"[Client {self.id}] Warning: Class '{class_name}' not found in any group. Using 'other' group.")
                            mapping[class_name] = 'other'

        return mapping

    def _reduce_sequence_adaptive(self, x, target_length=4):
        """
        Reduce sequence length using adaptive average pooling.

        Args:
            x: Input tensor of shape [batch, seq_len, dim]
            target_length: Target sequence length (default: 4)

        Returns:
            Tensor of shape [batch, target_length, dim]
        """
        batch_size, seq_len, dim = x.shape

        # If already at target length, return as is
        if seq_len == target_length:
            return x

        # Transpose for adaptive pooling: [batch, dim, seq_len]
        x_transposed = x.transpose(1, 2)

        # Adaptive pooling
        pooling = torch.nn.AdaptiveAvgPool1d(target_length)
        x_pooled = pooling(x_transposed)  # [batch, dim, target_length]

        # Transpose back: [batch, target_length, dim]
        return x_pooled.transpose(1, 2)

    def initialize_generators(self):
        """Initialize VAE/GAN generators based on granularity setting."""
        print(f"[Client {self.id}] Initializing {self.generator_type} generator(s) with granularity={self.generator_granularity}")

        # Get class-to-generator mapping
        class_mapping = self._get_class_to_generator_mapping()
        generator_keys = set(class_mapping.values())

        print(f"[Client {self.id}] Will create {len(generator_keys)} generator(s): {sorted(generator_keys)}")

        # Determine visual_dim based on diffusion_type
        if self.diffusion_type == 'flux':
            # FLUX uses T5 (4096) + CLIP (768) = 4864
            visual_dim = 4864
        else:
            # SD uses only CLIP (768)
            visual_dim = 768

        if self.generator_granularity == 'unified':
            # Single generator for all classes
            if self.generator_type == 'vae':
                if self.use_conditioned_vae:
                    self.prompt_generator = ConditionedVAEGenerator(
                        input_dim=768,  # AST output feature dimension per patch (after pooling to seq_len=4)
                        hidden_dim=1024,
                        latent_dim=256,
                        visual_dim=visual_dim,  # Adjusted based on diffusion_type
                        sequence_length=self.generator_training_sequence_length
                    ).to(self.device)
                    print(f"[Client {self.id}] Using conditioned VAE generator (training_seq_len={self.generator_training_sequence_length}, output_seq_len={self.generator_output_sequence_length or 'default'})")
                else:
                    self.prompt_generator = VAEGenerator(
                        input_dim=768,  # AST output feature dimension per patch (generates audio embeddings, not text prompts)
                        hidden_dim=1024,
                        latent_dim=256,
                        sequence_length=self.generator_training_sequence_length
                    ).to(self.device)
                    print(f"[Client {self.id}] Using unconditioned VAE generator (training_seq_len={self.generator_training_sequence_length}, output_seq_len={self.generator_output_sequence_length or 'default'})")

                # Initialize loss with adaptive beta scheduling based on configured training epochs
                self.generator_loss_fn = VAELoss(
                    total_epochs=self.generator_training_epochs,
                    beta_warmup_ratio=0.5  # Beta reaches 1.0 at 50% of total epochs
                )
                self.generator_optimizer = torch.optim.AdamW(
                    self.prompt_generator.parameters(),
                    lr=1e-3,
                    weight_decay=1e-5
                )

            elif self.generator_type == 'gan':
                self.prompt_generator_clip = GANGenerator(
                    latent_dim=128,
                    hidden_dim=512,
                    output_dim=768
                ).to(self.device)

                if self.generator_training_mode or self.generator_only_mode:
                    self.generator_discriminator = GANDiscriminator(
                        input_dim=768,
                        hidden_dim=512
                    ).to(self.device)

        else:
            # Multiple generators (per_class or per_group)
            if self.generator_type == 'vae':
                # Initialize loss with adaptive beta scheduling based on configured training epochs
                self.generator_loss_fn = VAELoss(
                    total_epochs=self.generator_training_epochs,
                    beta_warmup_ratio=0.5  # Beta reaches 1.0 at 50% of total epochs
                )

                for gen_key in generator_keys:
                    # Create a generator for each key (class name or group name)
                    if self.use_conditioned_vae:
                        # Use MultiModalVAEGenerator for conditioning
                        generator = MultiModalVAEGenerator(
                            input_dim=768,  # AST output feature dimension per patch (after pooling to seq_len=4)
                            hidden_dim=1024,
                            latent_dim=256,
                            sequence_length=self.generator_training_sequence_length
                        ).to(self.device)
                        print(f"[Client {self.id}]   Created conditioned VAE generator for '{gen_key}' (training_seq_len={self.generator_training_sequence_length})")
                    else:
                        # Use standard VAEGenerator without conditioning
                        generator = VAEGenerator(
                            input_dim=768,  # AST output feature dimension per patch (generates audio embeddings, not text prompts)
                            hidden_dim=1024,
                            latent_dim=256,
                            sequence_length=self.generator_training_sequence_length
                        ).to(self.device)
                        print(f"[Client {self.id}]   Created unconditioned VAE generator for '{gen_key}' (training_seq_len={self.generator_training_sequence_length})")

                    self.prompt_generators[gen_key] = generator

                    # Create optimizer for this generator
                    optimizer = torch.optim.AdamW(
                        generator.parameters(),
                        lr=1e-3,
                        weight_decay=1e-5
                    )
                    self.generator_optimizers[gen_key] = optimizer

            elif self.generator_type == 'gan':
                for gen_key in generator_keys:
                    generator = GANGenerator(
                        latent_dim=128,
                        hidden_dim=512,
                        output_dim=768
                    ).to(self.device)

                    self.prompt_generators[gen_key] = generator
                    print(f"[Client {self.id}]   Created GAN generator for '{gen_key}'")

        # Store the mapping for later use
        self.class_to_generator_map = class_mapping
        print(f"[Client {self.id}] Generator initialization complete")

    def save_generator_checkpoint(self, round_num=None):
        """
        Save generator checkpoint(s) for this client with comprehensive metadata.
        Handles unified, per_class, and per_group granularities.

        Returns:
            list: List of saved checkpoint paths
        """
        import os
        import datetime

        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.generator_checkpoint_dir, exist_ok=True)

        saved_paths = []

        if self.generator_granularity == 'unified':
            # Save single generator checkpoint
            if round_num is not None:
                checkpoint_path = os.path.join(
                    self.generator_checkpoint_dir,
                    f'{self.generator_checkpoint_base_name}_node{self.id}_round_{round_num}.pt'
                )
            else:
                checkpoint_path = os.path.join(
                    self.generator_checkpoint_dir,
                    f'{self.generator_checkpoint_base_name}_node{self.id}.pt'
                )

            saved_paths.append(self._save_single_generator_checkpoint(checkpoint_path, round_num, None, None))

        else:
            # Save multiple generator checkpoints (per_class or per_group)
            for gen_key, generator in self.prompt_generators.items():
                # Build filename based on granularity
                if self.generator_granularity == 'per_class':
                    suffix = f"class_{gen_key}"
                elif self.generator_granularity == 'per_group':
                    suffix = f"group_{gen_key}"
                else:
                    suffix = gen_key

                if round_num is not None:
                    checkpoint_path = os.path.join(
                        self.generator_checkpoint_dir,
                        f'{self.generator_checkpoint_base_name}_node{self.id}_{suffix}_round_{round_num}.pt'
                    )
                else:
                    checkpoint_path = os.path.join(
                        self.generator_checkpoint_dir,
                        f'{self.generator_checkpoint_base_name}_node{self.id}_{suffix}.pt'
                    )

                saved_paths.append(self._save_single_generator_checkpoint(checkpoint_path, round_num, gen_key, generator))

        return saved_paths

    def _save_single_generator_checkpoint(self, checkpoint_path, round_num, gen_key, generator=None):
        """
        Save a single generator checkpoint with metadata.

        Args:
            checkpoint_path: Path where to save the checkpoint
            round_num: Training round number
            gen_key: Generator key (class name, group name, or None for unified)
            generator: Generator instance (or None to use self.prompt_generator)

        Returns:
            str: Path where checkpoint was saved
        """
        import datetime

        # Use provided generator or default unified one
        if generator is None:
            generator = self.prompt_generator

        # Get optimizer for this generator
        if gen_key and gen_key in self.generator_optimizers:
            optimizer = self.generator_optimizers[gen_key]
        else:
            optimizer = self.generator_optimizer

        # Extract metadata from dataset
        selected_classes = None
        train_folds = None
        test_folds = None
        num_train_samples = 0
        num_test_samples = 0

        if hasattr(self.node_data, 'train_dataset') and self.node_data.train_dataset is not None:
            train_dataset = self.node_data.train_dataset
            num_train_samples = len(train_dataset)

            # Try to get selected_classes from dataset
            if hasattr(train_dataset, 'selected_classes'):
                selected_classes = train_dataset.selected_classes
            elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'selected_classes'):
                selected_classes = train_dataset.dataset.selected_classes

            # Try to get folds information for ESC50
            if hasattr(train_dataset, 'train_folds'):
                train_folds = train_dataset.train_folds
            elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'train_folds'):
                train_folds = train_dataset.dataset.train_folds

        if hasattr(self.node_data, 'test_dataset') and self.node_data.test_dataset is not None:
            test_dataset = self.node_data.test_dataset
            num_test_samples = len(test_dataset)

            if hasattr(test_dataset, 'test_folds'):
                test_folds = test_dataset.test_folds
            elif hasattr(test_dataset, 'dataset') and hasattr(test_dataset.dataset, 'test_folds'):
                test_folds = test_dataset.dataset.test_folds

        # Get classes for this specific generator (if applicable)
        generator_classes = None
        if gen_key and self.generator_granularity == 'per_class':
            generator_classes = [gen_key]
        elif gen_key and self.generator_granularity == 'per_group' and self.generator_class_groups:
            generator_classes = self.generator_class_groups.get(gen_key, [])

        # Prepare comprehensive checkpoint metadata
        checkpoint = {
            # Node identification
            'client_id': self.id,
            'node_id': self.id,

            # Training state
            'round': round_num if round_num is not None else self.round,
            'timestamp': datetime.datetime.now().isoformat(),

            # Generator configuration
            'generator_type': self.generator_type,
            'generator_granularity': self.generator_granularity,
            'diffusion_type': self.diffusion_type,
            'generator_training_epochs': self.generator_training_epochs,
            'synthetic_samples_per_class': self.synthetic_samples_per_class,

            # Granularity-specific metadata
            'generator_key': gen_key,
            'class_name': gen_key if self.generator_granularity == 'per_class' else None,
            'group_name': gen_key if self.generator_granularity == 'per_group' else None,
            'generator_classes': generator_classes,

            # Dataset metadata
            'dataset_name': self.dataset_name,
            'selected_classes': selected_classes,
            'train_folds': train_folds,
            'test_folds': test_folds,
            'num_train_samples': num_train_samples,
            'num_test_samples': num_test_samples,

            # Model architecture info
            'audio_model_name': self.audio_model_name,
            'img_pipe_name': self.img_pipe_name,
        }

        # Save generator state
        if self.generator_type == 'vae':
            if generator:
                checkpoint['generator_state_dict'] = generator.state_dict()
            else:
                checkpoint['generator_state_dict'] = self.prompt_generator.state_dict()

            if optimizer:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        elif self.generator_type == 'gan':
            if generator:
                checkpoint['generator_clip_state_dict'] = generator.state_dict()
            elif self.prompt_generator_clip:
                checkpoint['generator_clip_state_dict'] = self.prompt_generator_clip.state_dict()

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Create concise log message
        if gen_key:
            print(f"[Client {self.id}] Saved {self.generator_granularity} generator '{gen_key}' to {os.path.basename(checkpoint_path)}")
        else:
            print(f"[Client {self.id}] Saved generator checkpoint to {checkpoint_path}")

        return checkpoint_path

    def load_generator_checkpoint(self, checkpoint_path=None, strict_validation=True, warn_only=False):
        """
        Load generator checkpoint(s) for this client with metadata validation.
        Automatically detects and loads multiple checkpoints for per_class/per_group modes.

        Args:
            checkpoint_path: Optional path to checkpoint file (for unified mode)
            strict_validation: If True, reject checkpoint on metadata mismatch
            warn_only: If True, only print warnings without rejecting checkpoint

        Returns:
            bool: True if all checkpoints loaded successfully, False otherwise
        """
        import os
        import glob

        if self.generator_granularity == 'unified':
            # Load single checkpoint
            if checkpoint_path is None:
                checkpoint_path = os.path.join(
                    self.generator_checkpoint_dir,
                    f'{self.generator_checkpoint_base_name}_node{self.id}.pt'
                )

            if not os.path.exists(checkpoint_path):
                print(f"[Client {self.id}] Warning: Generator checkpoint not found at {checkpoint_path}")
                return False

            return self._load_single_generator_checkpoint(checkpoint_path, strict_validation, warn_only)

        else:
            # Load multiple checkpoints for per_class or per_group
            print(f"\n[Client {self.id}] Loading {self.generator_granularity} generators from {self.generator_checkpoint_dir}")

            # Build pattern to find all checkpoint files for this node
            if self.generator_granularity == 'per_class':
                pattern = os.path.join(
                    self.generator_checkpoint_dir,
                    f'{self.generator_checkpoint_base_name}_node{self.id}_class_*.pt'
                )
            else:  # per_group
                pattern = os.path.join(
                    self.generator_checkpoint_dir,
                    f'{self.generator_checkpoint_base_name}_node{self.id}_group_*.pt'
                )

            # Find all matching checkpoint files
            checkpoint_files = glob.glob(pattern)

            if not checkpoint_files:
                print(f"[Client {self.id}] Warning: No {self.generator_granularity} checkpoints found matching pattern: {pattern}")
                return False

            print(f"[Client {self.id}] Found {len(checkpoint_files)} checkpoint files")

            # Load each checkpoint
            success_count = 0
            for checkpoint_file in sorted(checkpoint_files):
                if self._load_single_generator_checkpoint(checkpoint_file, strict_validation, warn_only, multi_mode=True):
                    success_count += 1

            print(f"\n[Client {self.id}] Loaded {success_count}/{len(checkpoint_files)} generators successfully")
            return success_count == len(checkpoint_files)

    def _load_single_generator_checkpoint(self, checkpoint_path, strict_validation=True, warn_only=False, multi_mode=False):
        """
        Load a single generator checkpoint with validation.

        Args:
            checkpoint_path: Path to checkpoint file
            strict_validation: If True, reject on critical errors
            warn_only: If True, only warn without rejecting
            multi_mode: If True, loading multiple generators (suppress some output)

        Returns:
            bool: True if loaded successfully
        """
        import os

        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Extract generator key from checkpoint or filename
            gen_key = checkpoint.get('generator_key', None)
            if gen_key is None and multi_mode:
                # Extract from filename
                basename = os.path.basename(checkpoint_path)
                if '_class_' in basename:
                    gen_key = basename.split('_class_')[1].replace('.pt', '').split('_round_')[0]
                elif '_group_' in basename:
                    gen_key = basename.split('_group_')[1].replace('.pt', '').split('_round_')[0]

            # Display checkpoint metadata (compact in multi_mode)
            if not multi_mode:
                print(f"\n[Client {self.id}] Loading checkpoint from: {checkpoint_path}")
                print(f"  Checkpoint metadata:")
                print(f"    - Node ID: {checkpoint.get('node_id', 'N/A')}")
                print(f"    - Round: {checkpoint.get('round', 'N/A')}")
                print(f"    - Granularity: {checkpoint.get('generator_granularity', 'N/A')}")
                print(f"    - Generator key: {gen_key or 'unified'}")
                print(f"    - Generator type: {checkpoint.get('generator_type', 'N/A')}")
            else:
                print(f"  Loading '{gen_key}' from {os.path.basename(checkpoint_path)}")

            # Validation flags
            validation_errors = []
            validation_warnings = []

            # Validate node ID
            if 'client_id' in checkpoint or 'node_id' in checkpoint:
                checkpoint_node_id = checkpoint.get('node_id', checkpoint.get('client_id'))
                if checkpoint_node_id != self.id:
                    msg = f"Node ID mismatch: checkpoint={checkpoint_node_id}, current={self.id}"
                    validation_errors.append(msg)

            # Validate generator type
            if 'generator_type' in checkpoint:
                if checkpoint['generator_type'] != self.generator_type:
                    msg = f"Generator type mismatch: checkpoint={checkpoint['generator_type']}, current={self.generator_type}"
                    validation_errors.append(msg)

            # Validate dataset
            if 'dataset_name' in checkpoint:
                if checkpoint['dataset_name'] != self.dataset_name:
                    msg = f"Dataset mismatch: checkpoint={checkpoint['dataset_name']}, current={self.dataset_name}"
                    validation_warnings.append(msg)

            # Validate selected classes
            if 'selected_classes' in checkpoint and checkpoint['selected_classes'] is not None:
                current_classes = None
                if hasattr(self.node_data, 'train_dataset') and self.node_data.train_dataset is not None:
                    train_dataset = self.node_data.train_dataset
                    if hasattr(train_dataset, 'selected_classes'):
                        current_classes = train_dataset.selected_classes
                    elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'selected_classes'):
                        current_classes = train_dataset.dataset.selected_classes

                if current_classes is not None:
                    checkpoint_classes = set(checkpoint['selected_classes'])
                    current_classes_set = set(current_classes)
                    if checkpoint_classes != current_classes_set:
                        msg = f"Selected classes mismatch: checkpoint={sorted(checkpoint_classes)}, current={sorted(current_classes_set)}"
                        validation_warnings.append(msg)

            # Print validation results
            if validation_errors:
                print(f"\n[Client {self.id}] Validation ERRORS:")
                for error in validation_errors:
                    print(f"    ✗ {error}")

            if validation_warnings:
                print(f"\n[Client {self.id}] Validation WARNINGS:")
                for warning in validation_warnings:
                    print(f"    ⚠ {warning}")

            # Decide whether to reject checkpoint
            if validation_errors and strict_validation and not warn_only:
                print(f"\n[Client {self.id}] Checkpoint validation failed. Set strict_validation=False to load anyway.")
                return False

            if not validation_errors and not validation_warnings:
                print(f"[Client {self.id}] ✓ Checkpoint validation passed")

            # Initialize generators if not already initialized
            if self.generator_granularity == 'unified':
                if self.prompt_generator is None and self.prompt_generator_clip is None:
                    self.initialize_generators()
            else:
                # For multi-generator mode, ensure generators dict exists
                if not hasattr(self, 'prompt_generators') or not self.prompt_generators:
                    self.initialize_generators()

            # Load generator state based on granularity
            if self.generator_granularity == 'unified':
                # Load into single generator
                if self.generator_type == 'vae' and 'generator_state_dict' in checkpoint:
                    try:
                        self.prompt_generator.load_state_dict(checkpoint['generator_state_dict'])
                    except RuntimeError as e:
                        if 'size mismatch' in str(e):
                            print(f"[Client {self.id}] ⚠ Warning: Checkpoint has incompatible dimensions (likely old visual_dim).")
                            print(f"    Checkpoint was created with different architecture. Skipping load.")
                            return False
                        else:
                            raise

                    if 'optimizer_state_dict' in checkpoint and self.generator_optimizer:
                        try:
                            self.generator_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        except:
                            print(f"[Client {self.id}] ⚠ Warning: Could not load optimizer state (architecture changed)")

                    if not multi_mode:
                        print(f"[Client {self.id}] ✓ Loaded VAE generator from round {checkpoint.get('round', 'N/A')}")

                elif self.generator_type == 'gan' and 'generator_clip_state_dict' in checkpoint:
                    self.prompt_generator_clip.load_state_dict(checkpoint['generator_clip_state_dict'])
                    if not multi_mode:
                        print(f"[Client {self.id}] ✓ Loaded GAN generator from round {checkpoint.get('round', 'N/A')}")

            else:
                # Load into specific generator (per_class or per_group)
                if gen_key and gen_key in self.prompt_generators:
                    generator = self.prompt_generators[gen_key]

                    if self.generator_type == 'vae' and 'generator_state_dict' in checkpoint:
                        try:
                            generator.load_state_dict(checkpoint['generator_state_dict'])
                        except RuntimeError as e:
                            if 'size mismatch' in str(e):
                                print(f"    ⚠ Warning: Checkpoint for '{gen_key}' has incompatible dimensions. Skipping load.")
                                return False
                            else:
                                raise

                        if 'optimizer_state_dict' in checkpoint and gen_key in self.generator_optimizers:
                            try:
                                self.generator_optimizers[gen_key].load_state_dict(checkpoint['optimizer_state_dict'])
                            except:
                                print(f"    ⚠ Warning: Could not load optimizer state for '{gen_key}'")

                        print(f"    ✓ Loaded VAE generator '{gen_key}' from round {checkpoint.get('round', 'N/A')}")

                    elif self.generator_type == 'gan' and 'generator_clip_state_dict' in checkpoint:
                        generator.load_state_dict(checkpoint['generator_clip_state_dict'])
                        print(f"    ✓ Loaded GAN generator '{gen_key}' from round {checkpoint.get('round', 'N/A')}")
                else:
                    print(f"    ⚠ Warning: Generator key '{gen_key}' not found in initialized generators")
                    return False

            return True

        except Exception as e:
            print(f"[Client {self.id}] Error loading generator checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False

    def collect_embeddings_for_generator_training(self):
        """
        Collect audio embeddings (from AST model) and text embeddings (pre-generated CLIP/T5)
        for generator training. This is used in generator_only_mode.
        For unconditioned VAE, only audio embeddings are collected.
        """
        if self.use_conditioned_vae:
            embeddings_desc = f"audio + CLIP + T5 ({self.diffusion_type.upper()})" if self.diffusion_type == 'flux' else f"audio + CLIP (SD)"
        else:
            embeddings_desc = "audio only (unconditioned VAE)"
        print(f"[Client {self.id}] Processing batches to extract {embeddings_desc} embeddings...")

        # Initialize storage
        per_class_embeddings = {}

        # Get training data
        trainloader = self.load_train_data()
        self._move_to_gpu(self.device)

        # Set model to eval mode (we're not training adapters here)
        self.model.eval()

        with torch.no_grad():
            for batch_idx, samples in enumerate(trainloader):
                # Get audio data and class names
                audio_data = samples.get('audio', None)
                class_names = samples.get('class_name', None)
                text_embeddings = samples.get('text_emb', None) if self.use_conditioned_vae else None

                # For unconditioned VAE, we only need audio and class names
                if audio_data is None or class_names is None:
                    continue

                # For conditioned VAE, we also need text embeddings
                if self.use_conditioned_vae and text_embeddings is None:
                    continue

                # Prepare audio for AST model
                if isinstance(audio_data, torch.Tensor):
                    audio_data_np = audio_data.cpu().numpy()
                else:
                    audio_data_np = audio_data

                # Process through AST model to get audio embeddings
                audio_inputs = self.audio2image_model.feature_extractor(
                    audio_data_np,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                ).input_values.to(self.device)

                audio_embeddings = self.audio2image_model.ast_model(audio_inputs).last_hidden_state

                # Extract pre-generated text embeddings only for conditioned VAE
                clip_embeddings = None
                t5_embeddings = None

                if self.use_conditioned_vae:
                    # Extract pre-generated text embeddings based on diffusion type
                    if self.diffusion_type == 'sd':
                        # For SD: use CLIP embeddings only
                        clip_embeddings = text_embeddings.get('sd', None)
                        if clip_embeddings is None:
                            print(f"[Client {self.id}] Warning: Missing SD text embeddings in batch {batch_idx}")
                            continue

                    elif self.diffusion_type == 'flux':
                        # For FLUX: use both T5 and CLIP (pooled) embeddings
                        flux_embeddings = text_embeddings.get('flux', None)
                        if flux_embeddings is None:
                            print(f"[Client {self.id}] Warning: Missing FLUX text embeddings in batch {batch_idx}")
                            continue

                        t5_embeddings = flux_embeddings.get('prompt_embeds', None)  # T5 embeddings
                        clip_embeddings = flux_embeddings.get('pooled_prompt_embeds', None)  # CLIP pooled embeddings

                        if t5_embeddings is None or clip_embeddings is None:
                            print(f"[Client {self.id}] Warning: Incomplete FLUX embeddings in batch {batch_idx}")
                            continue

                # Store per-class
                for idx, class_name in enumerate(class_names):
                    if class_name not in per_class_embeddings:
                        per_class_embeddings[class_name] = {
                            'audio_embeddings': []
                        }
                        # Add visual embedding fields only for conditioned VAE
                        if self.use_conditioned_vae:
                            per_class_embeddings[class_name]['clip'] = []
                            # Add T5 field only for FLUX
                            if self.diffusion_type == 'flux':
                                per_class_embeddings[class_name]['t5'] = []

                    # Always store audio embeddings
                    per_class_embeddings[class_name]['audio_embeddings'].append(
                        audio_embeddings[idx].detach().cpu()
                    )

                    # Store visual embeddings only for conditioned VAE
                    if self.use_conditioned_vae:
                        per_class_embeddings[class_name]['clip'].append(
                            clip_embeddings[idx].detach().cpu()
                        )

                        # Store T5 embeddings for FLUX
                        if self.diffusion_type == 'flux':
                            per_class_embeddings[class_name]['t5'].append(
                                t5_embeddings[idx].detach().cpu()
                            )

                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1} batches...")

        # Store in per_class_embeddings
        self.per_class_embeddings = per_class_embeddings

        # Print summary
        total_samples = sum(len(data['audio_embeddings']) for data in per_class_embeddings.values())
        if self.use_conditioned_vae:
            embeddings_info = "audio, CLIP, T5" if self.diffusion_type == 'flux' else "audio, CLIP"
        else:
            embeddings_info = "audio only"
        print(f"[Client {self.id}] ✓ Collected {total_samples} samples ({embeddings_info}) across {len(per_class_embeddings)} classes")
        for class_name, data in per_class_embeddings.items():
            if self.use_conditioned_vae:
                embedding_types = ['audio', 'CLIP']
                if self.diffusion_type == 'flux' and 't5' in data:
                    embedding_types.append('T5')
            else:
                embedding_types = ['audio']
            print(f"  Class '{class_name}': {len(data['audio_embeddings'])} samples ({', '.join(embedding_types)} embeddings)")

    def train_node_generator(self):
        """
        Train the generator for this node using the stored adapter outputs.
        This method is called at the end of train_a2v() when in generator_training_mode.
        Handles unified, per_class, and per_group granularities.

        Uses ALL individual samples from training, not averaged outputs.
        """
        print(f"\n[Client {self.id}] Starting generator training mode (granularity={self.generator_granularity})")

        # If in generator_only_mode or no adapter outputs available, we need to collect embeddings first
        if self.generator_only_mode or self.per_class_embeddings is None:
            if self.use_conditioned_vae:
                embeddings_type = "audio, CLIP, and T5" if self.diffusion_type == 'flux' else "audio and CLIP"
            else:
                embeddings_type = "audio only (unconditioned VAE)"
            print(f"[Client {self.id}] Collecting {embeddings_type} embeddings from training data...")
            self.collect_embeddings_for_generator_training()

        # Prepare class outputs for generator training using ALL samples (not means)
        class_outputs_for_generator = {}

        # Get available classes from training data
        if self.per_class_embeddings is None:
            print(f"[Client {self.id}] WARNING: No training embeddings available")
            return 0.0

        classes_to_train = list(self.per_class_embeddings.keys())

        # If node has specific generator training classes, use only those
        if self.node_generator_train_classes:
            classes_to_train = [c for c in classes_to_train if c in self.node_generator_train_classes]
            print(f"[Client {self.id}] Training on subset of classes: {classes_to_train}")

        # Build class outputs dict with ALL individual samples
        total_samples = 0
        for class_name in classes_to_train:
            if class_name not in self.per_class_embeddings:
                continue

            class_data = self.per_class_embeddings[class_name]

            # Check required embeddings based on VAE type and diffusion type
            required_fields = ['audio_embeddings']

            # Add visual embeddings only if using conditioned VAE
            if self.use_conditioned_vae:
                required_fields.append('clip')
                if self.diffusion_type == 'flux':
                    required_fields.append('t5')

            # Verify all required fields are present
            if not all(field in class_data for field in required_fields):
                missing = [f for f in required_fields if f not in class_data]
                print(f"[Client {self.id}] Warning: Missing {missing} for class '{class_name}', skipping")
                continue

            audio_embs_list = class_data['audio_embeddings']
            clip_embs_list = class_data.get('clip', []) if self.use_conditioned_vae else []
            t5_embs_list = class_data.get('t5', []) if self.use_conditioned_vae and self.diffusion_type == 'flux' else []

            # Verify non-empty audio data
            if len(audio_embs_list) == 0:
                print(f"[Client {self.id}] Warning: Empty audio data for class '{class_name}', skipping")
                continue

            # For conditioned VAE, verify we have visual embeddings too
            if self.use_conditioned_vae and len(clip_embs_list) == 0:
                print(f"[Client {self.id}] Warning: Empty CLIP data for class '{class_name}', skipping")
                continue

            # Verify that we have the same number of embeddings (only for conditioned VAE)
            if self.use_conditioned_vae:
                if self.diffusion_type == 'flux':
                    if len(audio_embs_list) != len(clip_embs_list) or len(audio_embs_list) != len(t5_embs_list):
                        print(f"[Client {self.id}] Warning: Mismatch in number of samples for class '{class_name}' "
                              f"(audio: {len(audio_embs_list)}, clip: {len(clip_embs_list)}, t5: {len(t5_embs_list)}), skipping")
                        continue
                else:
                    if len(audio_embs_list) != len(clip_embs_list):
                        print(f"[Client {self.id}] Warning: Mismatch in number of samples for class '{class_name}' "
                              f"(audio: {len(audio_embs_list)}, clip: {len(clip_embs_list)}), skipping")
                        continue

            # Store as tensors: (num_samples, ...)
            class_outputs_for_generator[class_name] = {
                'audio_embeddings': torch.stack(audio_embs_list)  # (N, seq_len, 768)
            }

            # Add visual embeddings only for conditioned VAE
            if self.use_conditioned_vae:
                class_outputs_for_generator[class_name]['clip'] = torch.stack(clip_embs_list)  # (N, seq_len, 768)

                # Add T5 embeddings for FLUX
                if self.diffusion_type == 'flux' and t5_embs_list:
                    class_outputs_for_generator[class_name]['t5'] = torch.stack(t5_embs_list)  # (N, seq_len, 4096)

            total_samples += len(audio_embs_list)
            print(f"[Client {self.id}]   Class '{class_name}': {len(audio_embs_list)} samples")

        # Validate we have data to train
        if not class_outputs_for_generator:
            print(f"[Client {self.id}] WARNING: No valid class outputs available for generator training")
            return 0.0

        print(f"[Client {self.id}] Prepared {len(class_outputs_for_generator)} class(es) with {total_samples} total samples")

        # Train the generator(s) based on granularity
        generator_loss = self.train_generator(class_outputs_for_generator)

        # Save checkpoint periodically based on configured frequency
        checkpoint_frequency = getattr(self, 'generator_checkpoint_frequency', 5)
        if self.round % checkpoint_frequency == 0 or self.round == self.global_rounds:
            checkpoint_paths = self.save_generator_checkpoint(round_num=self.round)

            # Log saved checkpoints
            if self.generator_granularity == 'unified':
                print(f"[Client {self.id}] Saved unified generator checkpoint at round {self.round}")
            else:
                print(f"[Client {self.id}] Saved {len(checkpoint_paths)} {self.generator_granularity} generator checkpoint(s) at round {self.round}")

        # Log generator loss
        if self.data_log and not self.no_wandb:
            log_data = {
                f"train/node_{self.id}/generator_loss": generator_loss,
                f"train/node_{self.id}/generator_num_samples": total_samples,
                "round": self.round
            }

            # Add granularity-specific metrics
            if self.generator_granularity != 'unified':
                log_data[f"train/node_{self.id}/num_generators"] = len(self.prompt_generators) if hasattr(self, 'prompt_generators') else 0

            self.data_log(log_data)

        return generator_loss

    def train_generator(self, class_outputs):
        """
        Train generator(s) on adapter outputs based on granularity.

        Args:
            class_outputs: Dict with structure {class_name: {'clip': tensor, 'audio_embeddings': tensor}}

        Returns:
            Average loss across all epochs and classes/groups
        """
        if self.generator_granularity == 'unified':
            # Single generator for all classes
            print(f"[Client {self.id}] Training UNIFIED generator on {len(class_outputs)} class(es)")
            return self._train_single_generator(class_outputs, self.prompt_generator, self.generator_optimizer, 'unified')

        elif self.generator_granularity == 'per_class':
            # Train each class generator separately (one generator per class)
            print(f"[Client {self.id}] Training PER-CLASS generators (one per class)")
            total_losses = []
            trained_count = 0

            for class_name, class_data in class_outputs.items():
                if class_name not in self.class_to_generator_map:
                    print(f"  ⚠ Warning: Class '{class_name}' not in generator mapping, skipping")
                    continue

                gen_key = self.class_to_generator_map[class_name]
                if gen_key not in self.prompt_generators:
                    print(f"  ⚠ Warning: Generator '{gen_key}' not initialized, skipping")
                    continue

                # Train on this class only
                single_class_data = {class_name: class_data}
                generator = self.prompt_generators[gen_key]
                optimizer = self.generator_optimizers[gen_key]

                print(f"\n  Training generator for class '{gen_key}':")
                loss = self._train_single_generator(single_class_data, generator, optimizer, f"class '{gen_key}'")
                total_losses.append(loss)
                trained_count += 1

            avg_loss = sum(total_losses) / len(total_losses) if total_losses else 0.0
            print(f"\n[Client {self.id}] Completed per-class training: {trained_count} generator(s), avg loss={avg_loss:.4f}")
            return avg_loss

        elif self.generator_granularity == 'per_group':
            # Group classes by their generator and train each group
            print(f"[Client {self.id}] Training PER-GROUP generators (one per semantic group)")

            # Group classes by their generator
            groups_data = {}
            for class_name, class_data in class_outputs.items():
                if class_name not in self.class_to_generator_map:
                    print(f"  ⚠ Warning: Class '{class_name}' not in generator mapping, skipping")
                    continue

                gen_key = self.class_to_generator_map[class_name]
                if gen_key not in groups_data:
                    groups_data[gen_key] = {}
                groups_data[gen_key][class_name] = class_data

            # Train each group generator
            total_losses = []
            trained_count = 0

            for gen_key, group_class_data in groups_data.items():
                if gen_key not in self.prompt_generators:
                    print(f"  ⚠ Warning: Generator '{gen_key}' not initialized, skipping")
                    continue

                generator = self.prompt_generators[gen_key]
                optimizer = self.generator_optimizers[gen_key]

                print(f"\n  Training generator for group '{gen_key}' ({len(group_class_data)} classes):")
                loss = self._train_single_generator(group_class_data, generator, optimizer, f"group '{gen_key}'")
                total_losses.append(loss)
                trained_count += 1

            avg_loss = sum(total_losses) / len(total_losses) if total_losses else 0.0
            print(f"\n[Client {self.id}] Completed per-group training: {trained_count} generator(s), avg loss={avg_loss:.4f}")
            return avg_loss

        print(f"[Client {self.id}] ⚠ Warning: Unknown generator granularity '{self.generator_granularity}'")
        return 0.0

    def _train_single_generator(self, class_outputs, generator, optimizer, generator_name):
        """
        Train a single generator on given class outputs.

        Args:
            class_outputs: Dict {class_name: {'clip': tensor(N, seq_len, 768), 'audio_embeddings': tensor(N, seq_len, 512)}}
                          where N is the number of samples for that class
            generator: Generator model to train
            optimizer: Optimizer for this generator
            generator_name: Name for logging

        Returns:
            Average loss
        """
        if generator is None:
            print(f"[Client {self.id}] Generator {generator_name} not initialized, skipping")
            return 0.0

        generator.train()
        total_loss = 0.0
        num_batches = 0

        # Count total samples across all classes
        total_samples = sum(outputs['audio_embeddings'].shape[0] for outputs in class_outputs.values()
                           if 'audio_embeddings' in outputs)

        print(f"[Client {self.id}] Training generator '{generator_name}' ({total_samples} samples, {len(class_outputs)} class(es))")

        # Progress bar for epochs with compact bar
        epoch_pbar = tqdm(range(self.generator_training_epochs),
                         desc=f"  Gen {generator_name[:20]}",
                         leave=True,
                         bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        for epoch in epoch_pbar:
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_kl_loss = 0.0
            epoch_sim_loss = 0.0
            epoch_batches = 0

            for class_name, outputs in class_outputs.items():
                if 'audio_embeddings' not in outputs:
                    raise(f"    ⚠ Warning: Missing audio embeddings for class '{class_name}', skipping")

                if self.use_conditioned_vae:
                    if self.diffusion_type == 'flux':
                        if 'clip' not in outputs or 't5' not in outputs:
                            raise(f"    ⚠ Warning: Missing CLIP or T5 embeddings for class '{class_name}', skipping")
                    clip_embs_all = outputs['clip']               # (N, seq_len, 768 for SD or 2048 for FLUX pooled)

                    # Get T5 embeddings if using FLUX
                    t5_embs_all = outputs.get('t5', None) if self.diffusion_type == 'flux' else None

                    clip_embs_all = clip_embs_all.to(self.device)
                    if t5_embs_all is not None:
                        t5_embs_all = t5_embs_all.to(self.device)

                audio_embs_all = outputs['audio_embeddings']  # (N, seq_len, 768)
                audio_embs_all = audio_embs_all.to(self.device)
                

                num_samples = audio_embs_all.shape[0]

                # Process each sample individually
                for i in range(num_samples):
                    audio_emb = audio_embs_all[i:i+1]  # (1, seq_len, 768) - keep batch dim
                    if self.use_conditioned_vae:
                        clip_emb = clip_embs_all[i]        # (seq_len, 768 for SD or 2048 for FLUX pooled)
                        t5_emb = t5_embs_all[i] if t5_embs_all is not None else None  # (seq_len, 4096) - FLUX T5
                         # Reduce T5 embeddings if using FLUX
                        t5_emb_reduced = None
                        if self.diffusion_type == 'flux' and t5_emb is not None:
                        # Handle case where t5_emb might already have batch dimension
                            if t5_emb.dim() == 2:  # (seq_len, dim)
                                t5_emb_input = t5_emb.unsqueeze(0)  # (1, seq_len, dim)
                            else:  # Already (batch, seq_len, dim)
                                t5_emb_input = t5_emb
                            t5_emb_reduced = self._reduce_sequence_adaptive(t5_emb_input, target_length=self.generator_training_sequence_length).squeeze(0)  # (seq_len, 4096)
                        clip_emb_reduced = self._reduce_sequence_adaptive(clip_emb.unsqueeze(0), target_length=self.generator_training_sequence_length).squeeze(0)  # (seq_len, 768 or 2048)


                    # Always reduce to training sequence length for memory efficiency during training
                    if audio_emb.dim() == 3 and audio_emb.shape[1] == self.generator_training_sequence_length:
                        audio_emb_reduced = audio_emb
                    else:
                        audio_emb_reduced = self._reduce_sequence_adaptive(audio_emb, target_length=self.generator_training_sequence_length)  # (1, seq_len, 768)

                   
                    # Apply augmentation if enabled
                    if self.generator_augmentation:
                        noise = torch.randn_like(audio_emb_reduced) * self.generator_augmentation_noise
                        audio_emb_aug = audio_emb_reduced + noise
                    else:
                        audio_emb_aug = audio_emb_reduced

                    # Forward pass - handle VAEGenerator, ConditionedVAEGenerator, and MultiModalVAEGenerator
                    if isinstance(generator, VAEGenerator) and not isinstance(generator, (ConditionedVAEGenerator, MultiModalVAEGenerator)):
                        # Standard VAEGenerator without conditioning
                        recon_prompts, mu, logvar = generator(audio_emb_aug)
                    elif isinstance(generator, MultiModalVAEGenerator):
                        # MultiModalVAEGenerator expects separate clip_embedding and t5_embedding
                        if self.diffusion_type == 'flux' and t5_emb_reduced is not None:
                            # Pass both CLIP and T5 separately for FLUX
                            recon_prompts, mu, logvar = generator(audio_emb_aug, clip_embedding=clip_emb_reduced, t5_embedding=t5_emb_reduced)
                        else:
                            # Pass only CLIP for SD
                            recon_prompts, mu, logvar = generator(audio_emb_aug, clip_embedding=clip_emb_reduced)
                    else:
                        # ConditionedVAEGenerator expects concatenated visual_condition
                        if self.diffusion_type == 'flux' and t5_emb_reduced is not None:
                            # Concatenate T5 and CLIP embeddings for FLUX
                            visual_condition = torch.cat([t5_emb_reduced, clip_emb_reduced], dim=-1)  # (4, 4096+768)
                        else:
                            # Use only CLIP for SD
                            visual_condition = clip_emb_reduced  # (4, 768)
                        recon_prompts, mu, logvar = generator(audio_emb_aug, visual_condition)

                    # Compute loss (use reduced audio_emb to match recon_prompts dimensions)
                    loss, recon_loss, kl_loss, sim_loss = self.generator_loss_fn(
                        recon_prompts, audio_emb_reduced, mu, logvar, epoch
                    )

                    # Backward pass with provided optimizer
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=5.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_recon_loss += recon_loss.item()
                    epoch_kl_loss += kl_loss.item()
                    epoch_sim_loss += sim_loss.item()
                    epoch_batches += 1

            if epoch_batches > 0:
                avg_epoch_loss = epoch_loss / epoch_batches
                avg_recon_loss = epoch_recon_loss / epoch_batches
                avg_kl_loss = epoch_kl_loss / epoch_batches
                avg_sim_loss = epoch_sim_loss / epoch_batches
                total_loss += avg_epoch_loss
                num_batches += 1

                # Update progress bar with all loss components
                epoch_pbar.set_postfix({
                    'loss': f'{avg_epoch_loss:.4f}',
                    'recon': f'{avg_recon_loss:.4f}',
                    'kl': f'{avg_kl_loss:.4f}',
                    'sim': f'{avg_sim_loss:.4f}'
                })

        # Close progress bar
        epoch_pbar.close()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"  ✓ {generator_name} completed - Avg loss: {avg_loss:.4f}\n")

        return avg_loss

    def generate_synthetic_samples(self, class_outputs):
        """
        Generate synthetic samples using the pretrained generator.

        Args:
            class_outputs: Dict with structure {class_name: {'clip': tensor}}

        Returns:
            Dict with synthetic audio embeddings: {class_name: tensor(num_samples, seq_len, 768)}
            where seq_len is either 4 (default, reduced) or 1214 (full AST output length) depending on
            generator_target_sequence_length configuration.
        """
        if self.prompt_generator is None:
            print(f"[Client {self.id}] Generator not initialized, cannot generate samples")
            return {}

        self.prompt_generator.eval()
        synthetic_samples = {}

        # Determine output sequence length for generation (can be upsampled from training length)
        target_seq_len = self.generator_output_sequence_length
        seq_info = f" with shape [batch, {target_seq_len if target_seq_len else self.generator_training_sequence_length}, 768]" if target_seq_len else f" (default shape [batch, {self.generator_training_sequence_length}, 768])"

        with torch.no_grad():
            # Check if generator is unconditioned VAEGenerator
            is_unconditioned = isinstance(self.prompt_generator, VAEGenerator) and \
                             not isinstance(self.prompt_generator, (ConditionedVAEGenerator, MultiModalVAEGenerator))

            if is_unconditioned:
                # For unconditioned VAE, generate samples without class-specific conditioning
                # Generate once and replicate for all classes
                print(f"[Client {self.id}] Using unconditioned generator - generating {self.synthetic_samples_per_class} samples{seq_info}")
                synthetic_audio_embs = self.prompt_generator.sample(
                    num_samples=self.synthetic_samples_per_class,
                    device=self.device,
                    target_sequence_length=target_seq_len
                )
                # Assign the same samples to all classes
                for class_name in class_outputs.keys():
                    synthetic_samples[class_name] = synthetic_audio_embs
            else:
                # For conditioned generators, generate class-specific samples
                for class_name, outputs in class_outputs.items():
                    if 'clip' not in outputs:
                        continue

                    clip_emb = outputs['clip']  # (seq_len, 768)

                    # Generate synthetic samples conditioned on visual embeddings
                    synthetic_audio_embs = self.prompt_generator.sample(
                        num_samples=self.synthetic_samples_per_class,
                        visual_condition=clip_emb,
                        device=self.device
                    )

                    synthetic_samples[class_name] = synthetic_audio_embs

        print(f"[Client {self.id}] Generated {len(synthetic_samples)} synthetic sample sets{seq_info}")
        return synthetic_samples

    def evaluate_generator_quality(self, real_embeddings, generated_embeddings, metrics=['l2_distance', 'cosine_similarity']):
        """
        Evaluate the quality of generated samples comparing them with real embeddings.

        Args:
            real_embeddings: Dict {class_name: tensor(N, seq_len, dim)} - Real audio embeddings
            generated_embeddings: Dict {class_name: tensor(M, seq_len, dim)} - Generated audio embeddings
            metrics: List of metrics to compute. Options: ['l2_distance', 'cosine_similarity', 'mse', 'mae']

        Returns:
            Dict with quality metrics per class and overall statistics
        """
        if not real_embeddings or not generated_embeddings:
            print(f"[Client {self.id}] ⚠ Warning: Empty embeddings provided for quality evaluation")
            return {}

        results = {
            'per_class': {},
            'overall': {},
            'metrics': metrics
        }

        all_metrics = {metric: [] for metric in metrics}

        print(f"\n[Client {self.id}] 📊 Evaluating generator quality with metrics: {metrics}")

        for class_name in real_embeddings.keys():
            if class_name not in generated_embeddings:
                print(f"  ⚠ Skipping class '{class_name}' - no generated samples")
                continue

            real_emb = real_embeddings[class_name]  # (N, seq_len, dim)
            gen_emb = generated_embeddings[class_name]  # (M, seq_len, dim)

            # Ensure tensors are on the same device
            if real_emb.device != gen_emb.device:
                gen_emb = gen_emb.to(real_emb.device)

            class_metrics = self._compute_embedding_metrics(real_emb, gen_emb, metrics)
            results['per_class'][class_name] = class_metrics

            # Accumulate for overall statistics
            for metric in metrics:
                if metric in class_metrics:
                    all_metrics[metric].append(class_metrics[metric])

            # Print per-class results
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in class_metrics.items()])
            print(f"  • {class_name}: {metrics_str}")

        # Compute overall statistics
        for metric in metrics:
            if all_metrics[metric]:
                values = all_metrics[metric]
                results['overall'][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values))
                }

        # Print overall summary
        print(f"\n[Client {self.id}] 📈 Overall Quality Summary:")
        for metric, stats in results['overall'].items():
            print(f"  • {metric}: mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
                  f"min={stats['min']:.4f}, max={stats['max']:.4f}")

        return results

    def _compute_embedding_metrics(self, real_emb, gen_emb, metrics):
        """
        Compute quality metrics between real and generated embeddings.

        Args:
            real_emb: tensor(N, seq_len, dim) - Real embeddings
            gen_emb: tensor(M, seq_len, dim) - Generated embeddings
            metrics: List of metric names to compute

        Returns:
            Dict with computed metrics
        """
        results = {}

        # Flatten embeddings for comparison: (N, seq_len*dim) and (M, seq_len*dim)
        real_flat = real_emb.reshape(real_emb.shape[0], -1)  # (N, seq_len*dim)
        gen_flat = gen_emb.reshape(gen_emb.shape[0], -1)    # (M, seq_len*dim)

        # Compute mean embeddings for class-level comparison
        real_mean = real_flat.mean(dim=0, keepdim=True)  # (1, seq_len*dim)
        gen_mean = gen_flat.mean(dim=0, keepdim=True)    # (1, seq_len*dim)

        for metric in metrics:
            if metric == 'l2_distance':
                # L2 distance between mean embeddings
                l2_dist = torch.norm(real_mean - gen_mean, p=2).item()
                results['l2_distance'] = l2_dist

            elif metric == 'cosine_similarity':
                # Cosine similarity between mean embeddings
                cos_sim = torch.nn.functional.cosine_similarity(real_mean, gen_mean, dim=1).item()
                results['cosine_similarity'] = cos_sim

            elif metric == 'mse':
                # Mean Squared Error between mean embeddings
                mse = torch.nn.functional.mse_loss(real_mean, gen_mean).item()
                results['mse'] = mse

            elif metric == 'mae':
                # Mean Absolute Error between mean embeddings
                mae = torch.nn.functional.l1_loss(real_mean, gen_mean).item()
                results['mae'] = mae

            elif metric == 'frechet_distance':
                # Fréchet distance (simplified version using means and covariances)
                real_cov = torch.cov(real_flat.T)
                gen_cov = torch.cov(gen_flat.T)

                mean_diff = real_mean - gen_mean
                mean_diff_sq = torch.sum(mean_diff ** 2)

                # Simplified Fréchet distance (without sqrt of covariance product)
                cov_trace = torch.trace(real_cov + gen_cov)
                fid = mean_diff_sq + cov_trace
                results['frechet_distance'] = fid.item()

        return results

    def evaluate_generator_diversity(self, generated_embeddings):
        """
        Evaluate the diversity of generated samples within each class.

        Args:
            generated_embeddings: Dict {class_name: tensor(M, seq_len, dim)}

        Returns:
            Dict with diversity metrics per class and overall
        """
        if not generated_embeddings:
            print(f"[Client {self.id}] ⚠ Warning: Empty embeddings provided for diversity evaluation")
            return {}

        results = {
            'per_class': {},
            'overall': {}
        }

        all_std = []
        all_pairwise_dist = []

        print(f"\n[Client {self.id}] 🎨 Evaluating generator diversity")

        for class_name, gen_emb in generated_embeddings.items():
            # Flatten embeddings
            gen_flat = gen_emb.reshape(gen_emb.shape[0], -1)  # (M, seq_len*dim)

            # Compute standard deviation across samples
            std_per_dim = torch.std(gen_flat, dim=0).mean().item()

            # Compute average pairwise distance
            if gen_flat.shape[0] > 1:
                # Compute pairwise distances efficiently
                dists = torch.cdist(gen_flat, gen_flat, p=2)
                # Get upper triangle (excluding diagonal)
                triu_indices = torch.triu_indices(dists.shape[0], dists.shape[1], offset=1)
                pairwise_dist = dists[triu_indices[0], triu_indices[1]].mean().item()
            else:
                pairwise_dist = 0.0

            results['per_class'][class_name] = {
                'std': std_per_dim,
                'avg_pairwise_distance': pairwise_dist,
                'num_samples': gen_emb.shape[0]
            }

            all_std.append(std_per_dim)
            all_pairwise_dist.append(pairwise_dist)

            print(f"  • {class_name}: std={std_per_dim:.4f}, avg_dist={pairwise_dist:.4f} "
                  f"({gen_emb.shape[0]} samples)")

        # Overall statistics
        if all_std:
            results['overall'] = {
                'mean_std': float(np.mean(all_std)),
                'mean_pairwise_distance': float(np.mean(all_pairwise_dist))
            }

            print(f"\n[Client {self.id}] 📊 Overall Diversity:")
            print(f"  • Mean std: {results['overall']['mean_std']:.4f}")
            print(f"  • Mean pairwise distance: {results['overall']['mean_pairwise_distance']:.4f}")

        return results

    def evaluate_generator_coverage(self, real_embeddings, generated_embeddings, threshold=0.5):
        """
        Evaluate how well generated samples cover the real data distribution.

        Args:
            real_embeddings: Dict {class_name: tensor(N, seq_len, dim)}
            generated_embeddings: Dict {class_name: tensor(M, seq_len, dim)}
            threshold: Distance threshold for considering a real sample as "covered"

        Returns:
            Dict with coverage metrics per class and overall
        """
        if not real_embeddings or not generated_embeddings:
            print(f"[Client {self.id}] ⚠ Warning: Empty embeddings provided for coverage evaluation")
            return {}

        results = {
            'per_class': {},
            'overall': {},
            'threshold': threshold
        }

        all_coverage = []
        all_precision = []

        print(f"\n[Client {self.id}] 🎯 Evaluating generator coverage (threshold={threshold})")

        for class_name in real_embeddings.keys():
            if class_name not in generated_embeddings:
                continue

            real_emb = real_embeddings[class_name]
            gen_emb = generated_embeddings[class_name]

            # Ensure same device
            if real_emb.device != gen_emb.device:
                gen_emb = gen_emb.to(real_emb.device)

            # Flatten embeddings
            real_flat = real_emb.reshape(real_emb.shape[0], -1)
            gen_flat = gen_emb.reshape(gen_emb.shape[0], -1)

            # For each real sample, find minimum distance to generated samples
            # Coverage: % of real samples within threshold of at least one generated sample
            dists_to_gen = torch.cdist(real_flat, gen_flat, p=2)  # (N, M)
            min_dists_real = dists_to_gen.min(dim=1)[0]  # (N,)
            coverage = (min_dists_real < threshold).float().mean().item()

            # Precision: % of generated samples within threshold of at least one real sample
            min_dists_gen = dists_to_gen.min(dim=0)[0]  # (M,)
            precision = (min_dists_gen < threshold).float().mean().item()

            results['per_class'][class_name] = {
                'coverage': coverage,
                'precision': precision,
                'num_real': real_emb.shape[0],
                'num_generated': gen_emb.shape[0]
            }

            all_coverage.append(coverage)
            all_precision.append(precision)

            print(f"  • {class_name}: coverage={coverage:.2%}, precision={precision:.2%} "
                  f"(real={real_emb.shape[0]}, gen={gen_emb.shape[0]})")

        # Overall statistics
        if all_coverage:
            results['overall'] = {
                'mean_coverage': float(np.mean(all_coverage)),
                'mean_precision': float(np.mean(all_precision))
            }

            print(f"\n[Client {self.id}] 📊 Overall Coverage:")
            print(f"  • Mean coverage: {results['overall']['mean_coverage']:.2%}")
            print(f"  • Mean precision: {results['overall']['mean_precision']:.2%}")

        return results

    def set_server(self, server):
        """Set reference to the server for accessing diffusion model."""
        self.server = server

    def generate_images_from_diffusion(self, text_embeddings, base_embeddings=None):
        """
        Generate images using the server's diffusion model.

        Args:
            text_embeddings: Dict containing 't5' and 'clip' embeddings
            base_embeddings: Optional base embeddings for text generation

        Returns:
            Generated images as tensors
        """
        if self.server is None:
            logger.error(f"Client {self.id}: Server reference not set. Cannot generate images.")
            return []

        if 't5' not in text_embeddings or 'clip' not in text_embeddings:
            logger.error(f"Client {self.id}: Text embeddings is missing t5 or clip")
            return []

        # Access server's configuration
        generate_from_t5 = self.server.generate_from_t5_text_embeddings
        generate_from_clip = self.server.generate_from_clip_text_embeddings
        diffusion_device = self.server.diffusion_device
        diffusion_dtype = self.server.global_model.diffusion_dtype
        generate_low_memory = self.server.generate_low_memomy_footprint

        if base_embeddings is not None:
            orginal_prompt_embeds = []
            orginal_pooled_prompt_embeds = []
            for class_name in text_embeddings['class_name']:
                if class_name in base_embeddings:
                    orginal_prompt_embeds.append(base_embeddings[class_name]['flux']['prompt_embeds'])
                    orginal_pooled_prompt_embeds.append(base_embeddings[class_name]['flux']['pooled_prompt_embeds'])
                else:
                    logger.warning(f"Client {self.id}: Class name {class_name} not found in base embeddings")
                    continue

        if generate_from_t5 and base_embeddings:
            prompt_embeds = []
            for class_name in text_embeddings['class_name']:
                if class_name in base_embeddings:
                    prompt_embeds.append(base_embeddings[class_name]['flux']['prompt_embeds'])
                else:
                    logger.warning(f"Client {self.id}: Class name {class_name} not found in base embeddings")
                    continue
            prompt_embeds = torch.stack(prompt_embeds).squeeze(dim=1).to(diffusion_dtype).to(diffusion_device)
        else:
            prompt_embeds = text_embeddings['t5'].to(diffusion_dtype).to(diffusion_device)

        if generate_from_clip and base_embeddings:
            pooled_prompt_embeds = []
            for class_name in text_embeddings['class_name']:
                if class_name in base_embeddings:
                    pooled_prompt_embeds.append(base_embeddings[class_name]['flux']['pooled_prompt_embeds'])
                else:
                    logger.warning(f"Client {self.id}: Class name {class_name} not found in base embeddings")
                    continue
            pooled_prompt_embeds = torch.stack(pooled_prompt_embeds).squeeze(dim=1).to(diffusion_dtype).to(diffusion_device)
        else:
            pooled_prompt_embeds = text_embeddings['clip'].to(diffusion_dtype).to(diffusion_device)

        # Use server's diffusion model
        if not generate_low_memory:
            imgs = self.server.global_model.diffusion_model(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                num_inference_steps=1,
                output_type="pt",
            ).images
        else:
            imgs = self.generate_single_images_from_diffusion(prompt_embeds, pooled_prompt_embeds)

        return imgs

    def generate_single_images_from_diffusion(self, prompt_embeds, pooled_prompt_embeds):
        """Generate images one at a time for low memory footprint."""
        if self.server is None:
            logger.error(f"Client {self.id}: Server reference not set.")
            return []

        imgs = []
        for pe, ppe in zip(prompt_embeds, pooled_prompt_embeds):
            pe = pe.unsqueeze(0)
            ppe = ppe.unsqueeze(0)
            img = self.server.global_model.diffusion_model(
                prompt_embeds=pe,
                pooled_prompt_embeds=ppe,
                num_inference_steps=1,
                output_type="pt",
            ).images
            imgs.append(img)

        imgs = torch.cat(imgs, dim=0)
        return imgs

    def save_generated_images(self, imgs, embeddings, suffix="", round_num=None):
        """
        Save generated images to disk.

        Args:
            imgs: Generated image tensors
            embeddings: Dict containing class_name information
            suffix: Optional suffix for filenames
            round_num: Optional round number (uses server's round if not provided)

        Returns:
            Dict mapping image paths to class names
        """
        if self.server is None:
            logger.error(f"Client {self.id}: Server reference not set.")
            return {}

        images_output_dir = self.server.images_output_dir
        current_round = round_num if round_num is not None else self.server.round

        if not os.path.exists(images_output_dir):
            try:
                os.makedirs(images_output_dir)
            except FileExistsError:
                pass

        saved_images = {}
        for idx, img in enumerate(imgs):
            if type(embeddings['class_name']) == list:
                class_name = embeddings['class_name'][idx]
            else:
                class_name = embeddings['class_name']

            img_save_path = os.path.join(images_output_dir,
                                        f"round_{current_round}_node_{self.id}_img_{class_name}_{idx}{suffix}.png")
            saved_images[img_save_path] = class_name
            img = img.squeeze(0)
            converted_img = transforms.ToPILImage()(img.to(torch.float32).cpu())
            converted_img.save(img_save_path)
            print(f"Client {self.id}: Saved generated image to {img_save_path}")

        return saved_images

    def cleanup_temporary_images(self, generated_images_dict):
        """
        Delete temporary images that were generated for metrics computation
        but are not in save_generated_images_splits.

        Args:
            generated_images_dict: Dict with keys 'on_train', 'on_test', 'on_val'
                                  containing generated image paths
        """
        deleted_count = 0

        for split_key, image_paths_dict in generated_images_dict.items():
            if image_paths_dict is None:
                continue

            # Determine the split name from the key
            split_name = None
            if split_key == 'on_train':
                split_name = 'train'
            elif split_key == 'on_test':
                split_name = 'test'
            elif split_key == 'on_val':
                split_name = 'val'

            # If this split should not be saved, delete the images
            if split_name and split_name not in self.save_generated_images_splits:
                for img_path in image_paths_dict.keys():
                    if os.path.exists(img_path):
                        try:
                            os.remove(img_path)
                            deleted_count += 1
                            logger.debug(f"Client {self.id}: Deleted temporary image {img_path}")
                        except Exception as e:
                            logger.warning(f"Client {self.id}: Failed to delete {img_path}: {e}")

        if deleted_count > 0:
            print(f"Client {self.id}: Cleaned up {deleted_count} temporary images")

    def generate_images(self, split='test', round_num=None):
        """
        Generate images for this client using local data splits.

        Args:
            split: Which split to use - 'test', 'val', 'train', or 'all' (default: 'test')
            round_num: Optional round number for naming

        Returns:
            Dict with keys 'on_train', 'on_test', 'on_val' containing generated image paths
        """
        if self.server is None:
            logger.error(f"Client {self.id}: Server reference not set. Cannot generate images.")
            return {}

        print(f"\nClient {self.id}: Generating images from {split} split using server's diffusion model")

        # Ensure output directory exists
        images_output_dir = self.server.images_output_dir
        if not os.path.exists(images_output_dir):
            try:
                os.makedirs(images_output_dir)
            except FileExistsError:
                pass

        # Move diffusion model to device if needed
        if self.server.optimize_memory_usage or round_num <= 1:
            self.server.global_model.diffusion_model.to(self.server.diffusion_device)

        result = {
            'on_train': None,
            'on_test': None,
            'on_val': None
        }

        # Get datasets based on split parameter
        splits_to_process = []
        if split == 'all':
            splits_to_process = ['val', 'test', 'train']
        else:
            splits_to_process = [split]

        for split_name in splits_to_process:
            dataset = None
            suffix = ''

            if split_name == 'val':
                dataset = self.node_data.get_val_dataset()
                suffix = '_val'
                result_key = 'on_val'
            elif split_name == 'test':
                dataset = self.node_data.get_test_dataset()
                suffix = '_test'
                result_key = 'on_test'
            elif split_name == 'train':
                dataset = self.node_data.get_train_dataset()
                suffix = '_train'
                result_key = 'on_train'
            else:
                logger.warning(f"Client {self.id}: Unknown split '{split_name}', skipping")
                continue

            if dataset is not None and len(dataset) > 0:
                text_embs = dataset.text_embs if hasattr(dataset, 'text_embs') else None

                # Get audio embeddings from the dataset
                embeddings = self.get_audio_embeddings_from_dataset(dataset)

                imgs = self.generate_images_from_diffusion(embeddings, base_embeddings=text_embs)

                # Save images (always save for metrics computation)
                saved_files = self.save_generated_images(imgs, embeddings, suffix=suffix, round_num=round_num)
                result[result_key] = saved_files

                print(f"Client {self.id}: Generated {len(imgs)} images from {split_name} split")
            else:
                logger.info(f"Client {self.id}: No {split_name} dataset available or dataset is empty")

        # Move diffusion model back to CPU if optimizing memory
        if self.server.optimize_memory_usage:
            self.server.global_model.diffusion_model.to(torch.device('cpu'))

        return result

    def test_node_metrics_from_images(self, generated_images):
        """
        Compute test metrics from generated images.
        This function is copied from serverA2V.test_node_metrics_from_images
        """
        test_images = generated_images['on_test'] if 'on_test' in generated_images else {}
        train_images = generated_images['on_train'] if 'on_train' in generated_images else {}
        val_images = generated_images['on_val'] if 'on_val' in generated_images else {}

        found_test_classes = list(test_images.values()) if type(test_images) == dict else []
        found_train_classes = list(train_images.values()) if type(train_images) == dict else []
        found_val_classes = list(val_images.values()) if type(val_images) == dict else []

        found_classes = found_train_classes + found_val_classes + found_test_classes
        filenames = []
        if type(test_images) == dict:
            filenames.extend(list(test_images.keys()))
        if type(train_images) == dict:
            filenames.extend(list(train_images.keys()))
        if type(val_images) == dict:
            filenames.extend(list(val_images.keys()))

        candidate_labels = {}
        enriched_candidate_labels = {}
        unique_labels = list(set(found_classes))

        # Get federation classes from server
        if self.server is None:
            logger.error(f"Client {self.id}: Server reference not set. Cannot compute metrics.")
            return None

        federation_available_classes = self.server.federation_available_classes
        federation_active_classes = self.server.federation_active_classes

        for label_id, label in enumerate(unique_labels):
            candidate_labels[label] = label_id
            text_label = label.replace('_', ' ')
            enriched_candidate_labels[label] = f'This is a photo of {text_label}.'

        ground_truth_classes = []

        for ground_truth_class in found_classes:
            ground_truth_classes.append(federation_available_classes[ground_truth_class])

        ground_truth_classes = torch.tensor(ground_truth_classes)

        # Use server's global model for zero-shot computation
        predictions = self.server.global_model.compute_zero_shot(filenames, federation_available_classes)
        labels = torch.tensor(list(candidate_labels.values()))
        metrics = self.server.global_model._compute_classification_metrics(predictions, ground_truth_classes)

        node_metrics = NodeMetric(phase=NodeMetric.Phase.TEST, task_count=1)
        node_metrics.define_metrics(self.server.global_model.defined_test_metrics, task_count=1)
        for metric in node_metrics.defined_metrics:
            node_metrics[0][metric] = metrics[metric]
        node_metrics['samples'] = len(generated_images)
        node_metrics['steps'] = 1
        print(node_metrics)
        return node_metrics

    def client_test_metrics(self, generated_images, round_num=None):
        """
        Generate and log test metrics for this client.
        Similar to server's federation_test_metric but executed on client side.
        """
        if len(generated_images) == 0:
            logger.warning(f"Client {self.id}: No generated images provided for metrics computation")
            return

        if self.server is None:
            logger.error(f"Client {self.id}: Server reference not set. Cannot compute metrics.")
            return

        # Move zero-shot model to device
        self.server.global_model.zero_shot_model.model.to(self.device)

        # Compute metrics for this client
        node_metrics = self.test_node_metrics_from_images(generated_images)

        if node_metrics is not None:
            print(f"Client {self.id}\n{node_metrics}")

            # Log to wandb if enabled
            if not self.no_wandb:
                wandb_metrics = self.log_metrics(node_metrics, round=round_num if round_num else 0)
                wandb.log(wandb_metrics)

        # Move zero-shot model back to CPU
        self.server.global_model.zero_shot_model.model.to("cpu")


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