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
import torch.nn as nn
from torchvision import transforms

from flcore.routing.scoredrouting import ScoredRouting
from flcore.routing.randomrouting import RandomRouting
from flcore.routing.staticrouting import StaticRouting
from flcore.trainmodel.downstreamsinestesiaadapters import DownstreamSinestesiaAdapters
from flcore.trainmodel.downstreamsinestesia import DownstreamSinestesia
from flcore.trainmodel.Audio2Visual_NoData.src.models.sinestesia import SinestesiaWithClassifier
from datautils.node_dataset import NodeData
from torch.utils.data import ConcatDataset
from flcore.trainmodel.generators import ConditionedVAEGenerator, VAEGenerator

from torchinfo import summary

from utils.node_metric import NodeMetric
from utils.ballooning import GPUMemoryBalloon
from datautils.dataset_vegas import VEGASDataset
from datautils.dataset_esc50 import ESC50Dataset
from datautils.dataset_vggsound import VGGSoundDataset

import logging, sys

# Import Audio2Visual specific modules
# sys.path.append('/home/lpala/fedgfe/system/flcore/trainmodel/Audio2Visual_NoData')
from flcore.trainmodel.Audio2Visual_NoData.src.models.audio2image import Audio2Image, SDImageModel, ImageDiffusion

# Import generators
from flcore.trainmodel.generators import ConditionedVAEGenerator, VAEGenerator, VAELoss, GANGenerator, GANDiscriminator
import torch.nn.functional as F

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
        self.bolooning_reserve_mb = 1024
        if self.config.experiment.use_balooning:
            self.balooning_size = 34*1024
        self.baloon_gpu_id = 1 if torch.cuda.device_count() > 1 else 0
        self.baloon = GPUMemoryBalloon(gpu_id=self.baloon_gpu_id, chunk_size_mb=256,reserve_mb=self.bolooning_reserve_mb)
        self.baloon.allocate_memory(self.balooning_size)

        self.device = torch.device("cuda:0" if torch.cuda.is_available()else 'cpu')

        self.device1 = torch.device("cuda:0" if torch.cuda.device_count() > 1 else self.device)
        self.device2 = torch.device("cuda:1" if torch.cuda.device_count() > 1 else self.device)

        # Audio2Visual specific configuration
        self.diffusion_type = getattr(self.config.federation, 'diffusion_type', 'sd')
        self.diffusion_device = torch.device("cuda:1" if torch.cuda.device_count() > 1 else self.device)
        self.use_act_loss = getattr(args, 'use_act_loss', True)
        self.audio_model_name = getattr(args, 'audio_model_name', "MIT/ast-finetuned-audioset-10-10-0.4593")
        self.img_pipe_name = getattr(args, 'img_pipe_name', "runwayml/stable-diffusion-v1-5")
        self.img_lcm_lora_id = getattr(args, 'img_lcm_lora_id', "latent-consistency/lcm-lora-sdv1-5")
        
        self.optimize_memory_usage = getattr(self.config.experiment, 'optimize_memory_usage', False)

        self.global_model_train = getattr(self.config.feda2v, 'global_model_train', False)
        self.global_model_train_epochs = getattr(self.config.feda2v, 'global_model_train_epochs', 1)
        self.generate_nodes_images_frequency = getattr(self.config.feda2v, 'generate_nodes_images_frequency', 0)
        self.generate_global_images_frequency = getattr(self.config.feda2v, 'generate_global_images_frequency', 0)
        self.generate_low_memomy_footprint = getattr(self.config.feda2v, 'generate_low_memomy_footprint', False)
        self.generate_global_images_average_text_embeddings = getattr(self.config.feda2v, 'generate_global_images_average_text_embeddings', False)
        self.images_output_dir = getattr(self.config.feda2v, 'images_output_dir', 'output_images')
        self.generate_images_frequency = self.generate_nodes_images_frequency
        # Embeddings checkpoint settings
        self.generate_embeddings_only = getattr(self.config.feda2v, 'generate_embeddings_only', False)
        self.embeddings_checkpoint_dir = getattr(self.config.feda2v, 'embeddings_checkpoint_dir', 'checkpoints/embeddings')
        self.generate_from_clip_text_embeddings = getattr(self.config.feda2v, 'generate_from_clip_text_embeddings', False)
        self.generate_from_t5_text_embeddings = getattr(self.config.feda2v, 'generate_from_t5_text_embeddings', False)

        # Image generation splits configuration
        self.save_generated_images_splits = getattr(self.config.feda2v, 'save_generated_images_splits', ['val', 'test'])
        self.output_image_base_name = getattr(self.config.feda2v, 'output_image_base_name', 'img')
        self.generation_split_for_metrics = getattr(self.config.feda2v, 'generation_split_for_metrics', ['train'])

        # Nodes metrics splits
        self.nodes_test_metrics_splits = getattr(self.config.feda2v, 'nodes_test_metrics_splits', ['val', 'test'])
        self.nodes_train_metrics_splits = getattr(self.config.feda2v, 'nodes_train_metrics_splits', ['train'])

        # Server metrics splits
        self.server_test_metrics_splits = getattr(self.config.feda2v, 'server_test_metrics_splits', ['val', 'test'])
        self.server_train_metrics_splits = getattr(self.config.feda2v, 'server_train_metrics_splits', ['train'])

        self.adapter_aggregation_mode = self.config.feda2v.adapter_aggregation_mode
        self.global_model_train_from_nodes_adapters = self.config.feda2v.global_model_train_from_nodes_adapters
        self.global_model_train_from_generator = self.config.feda2v.global_model_train_from_generator

        # Global mean computation configuration
        self.compute_global_mean_from_class_means = getattr(self.config.feda2v, 'compute_global_mean_from_class_means', True)

        # Generator configuration
        self.generators = {}
        self.generator_type = getattr(self.config.feda2v, 'generator_type', 'vae')
        self.use_generator = getattr(self.config.feda2v, 'use_generator', False)
        self.generator_only_mode = getattr(self.config.feda2v, 'generator_only_mode', False)
        self.use_conditioned_vae = getattr(self.config.feda2v, 'use_conditioned_vae', True)
        self.generator_training_mode = getattr(self.config.feda2v, 'generator_training_mode', False)
        self.shared_generator_in_only_mode = getattr(self.config.feda2v, 'shared_generator_in_only_mode', True)

        self.generator_training_sequence_length = getattr(self.config.feda2v,'generator_training_sequence_length', 4)
        self.generator_output_sequence_length = getattr(self.config.feda2v,'generator_output_sequence_length', None)

        # Training mode configuration
        # train_adapters: whether to train adapters (if False, only generator training)
        self.train_adapters = getattr(self.config.feda2v, 'train_adapters', True)
        self.prompt_generator = None
        self.prompt_generator_clip = None
        self.prompt_generator_t5 = None
        self.discriminator_clip = None
        self.discriminator_t5 = None
        self.generator_optimizer = None
        self.generator_loss_fn = None

        # Generator checkpoint configuration with flexible naming
        self.generator_save_checkpoint = getattr(self.config.feda2v, 'generator_save_checkpoint', False)
        self.generator_load_checkpoint = getattr(self.config.feda2v, 'generator_load_checkpoint', False)
        self.generator_checkpoint_frequency = getattr(self.config.feda2v, 'generator_checkpoint_frequency', 10)

        # Support both legacy single path and new flexible directory + base name system
        self.generator_checkpoint_path = getattr(self.config.feda2v, 'generator_checkpoint_path', None)
        self.generator_checkpoint_dir = getattr(self.config.feda2v, 'generator_checkpoint_dir', 'checkpoints/generators')
        self.generator_checkpoint_base_name = getattr(self.config.feda2v, 'generator_checkpoint_base_name', 'server_generator')

        # If legacy path is provided, extract dir and base name from it
        if self.generator_checkpoint_path:
            import os
            self.generator_checkpoint_dir = os.path.dirname(self.generator_checkpoint_path) or 'checkpoints/generators'
            base_with_ext = os.path.basename(self.generator_checkpoint_path)
            self.generator_checkpoint_base_name = os.path.splitext(base_with_ext)[0]

        # Generator training configuration
        self.generator_training_epochs = getattr(self.config.feda2v, 'generator_training_epochs', 5)
        self.generator_augmentation = getattr(self.config.feda2v, 'generator_augmentation', True)
        self.generator_augmentation_noise = getattr(self.config.feda2v, 'generator_augmentation_noise', 0.1)
        self.synthetic_samples_per_class = getattr(self.config.feda2v, 'synthetic_samples_per_class', 'auto')
        self.generator_validation_frequency = getattr(self.config.feda2v, 'generator_validation_frequency', 5)

        # Adapter checkpoint configuration
        self.adapter_checkpoint_dir = getattr(self.config.feda2v, 'adapter_checkpoint_dir', 'checkpoints/adapters')
        self.adapter_checkpoint_base_name = getattr(self.config.feda2v, 'adapter_checkpoint_base_name', 'adapter')
        self.adapter_save_checkpoint = getattr(self.config.feda2v, 'adapter_save_checkpoint', False)
        self.adapter_load_checkpoint = getattr(self.config.feda2v, 'adapter_load_checkpoint', False)
        self.adapter_checkpoint_frequency = getattr(self.config.feda2v, 'adapter_checkpoint_frequency', 5)
        self.adapter_checkpoint_per_type = getattr(self.config.feda2v, 'adapter_checkpoint_per_type', True)

        # Create Encoders models Audio2Visual model
        self.create_global_model(args)

        # #FIXME for testing only
        # self.test_metrics_from_images()

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

        for client in self.clients:
            print(f"\n*** Client {client.id} dataset {client.dataset}")
            if isinstance(client.node_data.dataset, VEGASDataset):
                logger.warn ( "Skipping dataset details for VEGASDataset to avoid long logs." )
                continue
            client.node_data.stats_dump()
            if hasattr(client, "server_synthetic_samples"):
                print(f"  Server synthetic samples available for classes: {list(client.server_synthetic_samples.keys())}")

        if self.no_wandb == False:
            self.define_metrics()

        for client in self.clients:
            client.federation_clients = self.clients

        # Set server reference in clients for accessing diffusion_model
        for client in self.clients:
            client.set_server(self)

        self.nodes_adapters = {}
        self.nodes_adapters_modules = {}

        logger.info("Finished creating Audio2Visual server and clients.")

        self.Budget = []
        self.num_classes = args.num_classes
        self.statistics_dataframe = None

    def test_node_metrics_from_images ( self, node, generated_images ):
        images = {}
        images |= generated_images['test'] if 'test' in generated_images else {}
        images |= generated_images['train'] if 'train' in generated_images else {}
        images |= generated_images['val'] if 'val' in generated_images else {}

        found_classes = list(images.values())
        filenames = list(images.keys())
        # found_classes.append(list(train_images.values()))

        candidate_labels = {}
        enriched_candidate_labels = {}
        unique_labels = list(set(found_classes))
        federation_available_classes = self.federation_available_classes
        federation_active_classes = self.federation_active_classes

        for label_id, label in enumerate(unique_labels):
            candidate_labels[label] = label_id
            text_label = label.replace('_', ' ')
            enriched_candidate_labels[label] = f'This is a photo of {text_label}.'

        ground_truth_classes = []

        for ground_truth_class in found_classes:
            ground_truth_classes.append(federation_available_classes[ground_truth_class])

        ground_truth_classes = torch.tensor(ground_truth_classes) 

        # candidate_labels = [f'This is a photo of {label}.' for label in classes]

        node_splits_metrics = {}
        for split,generated_filenames in generated_images.items():

            found_classes = list(generated_filenames.values())
            ground_truth_classes = []

            for ground_truth_class in found_classes:
                ground_truth_classes.append(federation_available_classes[ground_truth_class])
            ground_truth_classes = torch.tensor(ground_truth_classes, device=self.global_model.zero_shot_model.device) 

            print ( f"Generated images for split {split}: {len(generated_images[split])} samples." )
            filenames = list(generated_filenames.keys())
            predictions = self.global_model.compute_zero_shot( filenames, federation_available_classes )
            labels = torch.tensor(list(candidate_labels.values()))
            metrics = self.global_model._compute_classification_metrics (predictions, ground_truth_classes)

            node_metrics = NodeMetric(phase=NodeMetric.Phase.TEST, task_count=1)
            node_metrics.define_metrics(self.global_model.defined_test_metrics, task_count=1)
            for metric in node_metrics.defined_metrics:
                node_metrics[0][metric] = metrics[metric]
            node_metrics['samples'] = len(generated_images)
            node_metrics['steps'] = 1
            node_metrics.phase = NodeMetric.Phase.TEST
            # if split == 'test':
            #     node_metrics.phase = NodeMetric.Phase.TEST
            # elif split == 'train':
            #     node_metrics.phase = NodeMetric.Phase.TRAIN
            # elif split == 'val':
            #     node_metrics.phase = NodeMetric.Phase.VALIDATE

            node_splits_metrics[split] = node_metrics

            print ( node_metrics )
        return node_splits_metrics

    def test_metrics_from_images ( self ):
        images_path = '/home/lpala/fedgfe/esc50-6n-global-train'
        images = load_images_with_class_from_path(images_path=images_path)
        images, classes, filenames = prepare_for_metrics (images)

        candidate_labels = {}
        enriched_candidate_labels = {}
        unique_labels = list(set(classes))
        for label_id, label in enumerate(unique_labels):
            candidate_labels[label] = label_id
            text_label = label.replace('_', ' ')
            enriched_candidate_labels[label] = f'This is a photo of {text_label}.'

        ground_truth_classes = []

        for ground_truth_class in classes:
            ground_truth_classes.append(candidate_labels[ground_truth_class])

        ground_truth_classes = torch.tensor(ground_truth_classes) 

        # candidate_labels = [f'This is a photo of {label}.' for label in classes]

        predictions = self.global_model.compute_zero_shot( filenames, candidate_labels )
        labels = torch.tensor(list(candidate_labels.values()))
        metrics = self.global_model._compute_classification_metrics (predictions, ground_truth_classes)
        print ( metrics )
                                                 
    def create_global_model(self, args):

        # Get use_cls_token_only configuration from feda2v config
        use_cls_token_only = getattr(self.config.feda2v, 'use_cls_token_only', False) if hasattr(self, 'config') and self.config else False

        # Get use_pretrained_generators flag - if True, AST will NOT be initialized
        use_pretrained_generators = getattr(self.config.feda2v, 'use_pretrained_generators', False) if hasattr(self, 'config') and self.config else False

        self.global_model = DownstreamSinestesiaAdapters(
            args,
            diffusion_type=self.diffusion_type,
            use_cls_token_only=use_cls_token_only,
        )

        # Freeze AST model parameters only if AST was initialized
        if self.global_model.ast_model is not None:
            for param in self.global_model.ast_model.parameters():
                param.requires_grad = False

        self.encoder_audio = None
        self.encoder_image = None
        self.encoder_text = None

        # NOTE: audio_embeddings_dataset_cache is handled by the client, not by the model

        if self.diffusion_type == 'flux':
            img_pipe_name = 'MIT/ast-finetuned-audioset-10-10-0.4593'
        elif self.diffusion_type == 'sd':
            img_pipe_name = 'runwayml/stable-diffusion-v1-5'


        generation_splits = self.server_test_metrics_splits + self.server_train_metrics_splits + self.save_generated_images_splits

        generation_splits = list(set(generation_splits))
        generation_splits_len = len(generation_splits)
        # Only initialize diffusion model if NOT in generator_only_mode
        # In generator_only_mode, we only train generators and don't need diffusion for image generation
        if not self.generator_only_mode and (self.generate_global_images_frequency or self.generate_nodes_images_frequency or generation_splits_len):
            self.global_model.enable_diffusion = True
            self.global_model.image_generation_frequency = self.generate_global_images_frequency
            self.global_model.generate_low_memomy_footprint = self.generate_low_memomy_footprint
            self.global_model.start_diffusion( low_memory_footprint = self.global_model.generate_low_memomy_footprint)

        self.global_adapters = self.global_model.adapters
        self.global_adapters_modules = self.global_model.adapters_modules

        self.global_optimizers = self.create_global_model_optimizers()

        # Initialize generators if enabled (but don't load checkpoints yet - will be done after clients are created)
        if self.use_generator:
            # Only initialize fresh generators if NOT loading from checkpoint
            # Checkpoint loading will happen after clients are created and federation classes are collected
            if not self.generator_load_checkpoint:
                self.initialize_generators()
            else:
                logger.info("Generator checkpoint loading deferred until after client creation")

        # Assign generators to global_model
        self.global_model.generators_dict = self.prompt_generators if hasattr(self, 'prompt_generators') else None

        # If shared_generator_in_only_mode is active, also set prompt_generator references in global_model
        # so that clients can access them directly
        if self.shared_generator_in_only_mode and self.generator_only_mode:
            if hasattr(self, 'prompt_generator') and self.prompt_generator is not None:
                self.global_model.prompt_generator = self.prompt_generator
                logger.info("Set unified prompt_generator in global_model for shared access")
            if hasattr(self, 'prompt_generator_clip') and self.prompt_generator_clip is not None:
                self.global_model.prompt_generator_clip = self.prompt_generator_clip
                logger.info("Set prompt_generator_clip in global_model for shared access")
            if hasattr(self, 'prompt_generator_t5') and self.prompt_generator_t5 is not None:
                self.global_model.prompt_generator_t5 = self.prompt_generator_t5
                logger.info("Set prompt_generator_t5 in global_model for shared access")
            if hasattr(self, 'generator_optimizer') and self.generator_optimizer is not None:
                self.global_model.generator_optimizer = self.generator_optimizer
                logger.info("Set generator_optimizer in global_model for shared access")

        # Load adapter checkpoint if configured
        if self.adapter_load_checkpoint:
            logger.info("Loading adapter checkpoint from configuration")
            success = self.load_adapter_checkpoint()
            if success:
                logger.info("Successfully loaded adapter checkpoint")
            else:
                logger.warning("Could not load adapter checkpoint, starting from scratch")

        return self.global_model, self.generator_model
    
    def create_global_model_optimizers(self):
        optimizers = {}
        for module_name, module in self.global_adapters.items():
            if self.config.training.optimizer == "AdamW":
                optimizers[module_name] = AdamW(params=module.parameters(), lr=self.config.training.learning_rate)

        return optimizers

    def initialize_generators(self):
        """Initialize prompt generators (VAE or GAN) for server-side training."""
        logger.info(f"Initializing {self.generator_type} generator on server (conditioned={self.use_conditioned_vae})")

        if self.generator_type == 'vae':
            # Choose between conditioned and unconditioned VAE
            if self.use_conditioned_vae:
                # VAE for conditioned prompt generation
                self.prompt_generator = ConditionedVAEGenerator(
                    input_dim=768,      # Audio embeddings dimension from AST
                    visual_dim=768,     # CLIP embeddings dimension
                    hidden_dim=512,
                    latent_dim=256,
                    sequence_length=self.generator_training_sequence_length
                )
                logger.info("Conditioned VAE generator initialized")
            else:
                # VAE for unconditioned prompt generation
                self.prompt_generator = VAEGenerator(
                    input_dim=768,      # Audio embeddings dimension from AST
                    hidden_dim=1024,
                    latent_dim=256,
                    sequence_length=self.generator_training_sequence_length
                )
                logger.info("Unconditioned VAE generator initialized")

            # Initialize loss with adaptive beta scheduling based on configured training epochs
            # Beta warmup ratio of 0.5 allows beta to reach 1.0 at 50% of training
            self.generator_loss_fn = VAELoss(
                total_epochs=self.generator_training_epochs,
                beta_warmup_ratio=0.5  # Beta reaches 1.0 at 50% of total epochs (10 epochs if total=20)
            )

            # Optimizer for VAE with balanced learning rate
            # Increased from 0.01x to 0.1x for better convergence
            self.generator_optimizer = torch.optim.AdamW(
                self.prompt_generator.parameters(),
                lr=self.config.training.learning_rate * 0.1,  # 10x faster than before for better convergence
                weight_decay=1e-4  # Strong weight decay to prevent explosion
            )

            # IMPORTANT: Also store in generator_optimizers dict with key 'unified'
            # This ensures consistent access pattern for both unified and per_class granularities
            if not hasattr(self, 'generator_optimizers'):
                self.generator_optimizers = {}
            self.generator_optimizers['unified'] = self.generator_optimizer

        elif self.generator_type == 'gan':
            # GAN generators for CLIP and T5
            self.prompt_generator_clip = GANGenerator(
                latent_dim=256,
                hidden_dim=512,
                output_dim=768  # CLIP dimension
            ).to(self.device)

            if self.diffusion_type == 'flux':
                self.prompt_generator_t5 = GANGenerator(
                    latent_dim=256,
                    hidden_dim=512,
                    output_dim=4096  # T5 dimension
                ).to(self.device)

            # Discriminators
            self.discriminator_clip = GANDiscriminator(
                input_dim=768,
                hidden_dim=512
            ).to(self.device)

            if self.diffusion_type == 'flux':
                self.discriminator_t5 = GANDiscriminator(
                    input_dim=4096,
                    hidden_dim=512
                ).to(self.device)

            # Optimizers for GAN
            gen_params = list(self.prompt_generator_clip.parameters())
            if self.prompt_generator_t5:
                gen_params += list(self.prompt_generator_t5.parameters())

            disc_params = list(self.discriminator_clip.parameters())
            if self.discriminator_t5:
                disc_params += list(self.discriminator_t5.parameters())

            self.generator_optimizer = torch.optim.AdamW(gen_params, lr=1e-4)
            self.discriminator_optimizer = torch.optim.AdamW(disc_params, lr=1e-4)

            # IMPORTANT: Also store in generator_optimizers dict with key 'unified'
            # This ensures consistent access pattern for both unified and per_class granularities
            if not hasattr(self, 'generator_optimizers'):
                self.generator_optimizers = {}
            self.generator_optimizers['unified'] = self.generator_optimizer

            logger.info("GAN generators and discriminators initialized")

    def create_nodes_model(self, model_string=None, global_model=None):
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
            # if not self.generator_only_mode:
            #     self._move_to_gpu(self.diffusion_device, models=['diffusion', 'zeroshot'])
            #     node._move_to_gpu(node.device)
            #     self.client_round_starting_hook(node)
            #     self._move_to_cpu( models=['diffusion', 'zeroshot'] )
            node.device = self.device
            node.round = round
            node.thread = None
            node.federation_size = len(self.clients)
            node._move_to_gpu(node.device)
            if not self.adapter_load_checkpoint:
                node.train()

            if not self.generator_only_mode:
                self.client_round_ending_hook(node)

            if self.generate_images_frequency > 0 and self.round % self.generate_images_frequency == 0:
                if self.generate_embeddings_only:
                    # Save embeddings only (no image generation)
                    self._move_to_gpu(self.diffusion_device, models=['ast','adapter'])
                    node._move_to_gpu(node.device)
                    logger.info(f"Saving embeddings for Node {node.id} to checkpoint (embeddings-only mode)")
                    for split in self.save_generated_images_splits:
                        node.save_embeddings_to_checkpoint(
                            split=split,
                            checkpoint_dir=self.embeddings_checkpoint_dir,
                            round_num=self.round
                        )
                    self._move_to_cpu(models=['ast','adapter'])
                else:
                    # Generate images as normal
                    self._move_to_gpu(self.diffusion_device, models=['diffusion', 'zeroshot'])
                    self.generate_images(node)
                    self._move_to_cpu( models=['diffusion', 'zeroshot'] )
            node_index += 1
            node._move_to_cpu()

    def train(self):
        """Main training loop for Audio2Visual federated learning."""
        training_task = "both"

        if self.generator_training_mode:
            print("\n" + "="*80)
            print("GENERATOR TRAINING MODE ENABLED")
            print("Adapter distribution and aggregation will be DISABLED")
            print("Model evaluation (train/test metrics) will be DISABLED")
            print("="*80 + "\n")

        if self.global_model != None:
            self.global_model = self.global_model.to(self.device)

        for i in range(1, self.global_rounds + 1):
            self.round = i

            if self.no_wandb == False:
                self.data_log({"round": self.round})

            s_t = time.time()
            self.selected_clients = self.clients

            if self.eval_gap > 0 and self.round % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate Audio2Visual models")
                self.evaluate()


            self.round_starting_hook()

            # Skip sending models/adapters in generator-only training mode
            if not self.generator_training_mode:
                if self.aggregation_method != 'none' or self.adapter_aggregation_mode != 'none':
                    self.send_models()
            else:
                print("[Generator Training Mode] Skipping adapter distribution")

            self.train_nodes(i, training_task=training_task)

            if self.store_audio_embeddings and self.audio_embedding_file_name is not None and self.round == 1:
                self.save_audio_embeddings(file_name=self.audio_embedding_file_name)
                print(f"Saved audio embeddings to {self.audio_embedding_file_name}")

            print(self.uploaded_ids)

            self.Budget.append(time.time() - s_t)
            print('-' * 50 + f"Round {self.round} time: ", self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

            # Skip receiving/aggregating models/adapters in generator-only training mode
            if not self.generator_training_mode:
                if self.model_aggregation != "none" or self.adapter_aggregation_mode != 'none':
                    self.receive_models()
                    self.aggregate_parameters()
            else:
                print("[Generator Training Mode] Skipping adapter aggregation")

            if self.model_backbone_save_checkpoint:
                self.save_checkpoint()

            self.round_ending_hook()

        self.save_results()
        self.evaluate()
        wandb.finish()

    def node_metrics_log( self, node, suffix="", train_splits = [], test_splits = [] ):
        node_metrics = self.node_metrics(node, train_splits=train_splits, test_splits=test_splits)
        # round_train_metrics = node_metrics['train']['train'] if 'train' in node_metrics['train'] else None
        # round_test_metrics = node_metrics['test']['val'] if 'val' in node_metrics['test'] else None
        # round_test_metrics_on_train = node_metrics['test']['train'] if 'train' in node_metrics['test'] else None

        if not self.no_wandb:
            wandb_metrics = {}

            for metric_type, metric_splits in node_metrics.items():
                for split, metric in metric_splits.items():
                    wandb_metrics.update(node.log_metrics( metric, round = self.round, suffix=f"_on_{split}{suffix}"))

            for wandb_metric in wandb_metrics:
                metrics = {wandb_metric: wandb_metrics[wandb_metric], "round": self.round}
                self.data_log(metrics)

         # Display only text_loss or accuracy if present
        display_metrics = []
        for metric_name in ['text_loss', 'accuracy']:
            for metric_type, metric_splits in node_metrics.items():
                for split, metrics in metric_splits.items():
                    if metric_name in metrics:
                        display_metrics.append(f"{metric_type} on {split} {metric_name} {metrics[metric_name]['mean']:.4f}")

        if display_metrics:
            print(F"Node {node.id} metrics: " + ", ".join(display_metrics))

    def round_starting_hook(self):
        # FIXME
        logger.warning(f"=== Round {self.round} starting hook temporally disabled ===")
        return

        if self.eval_gap > 0 and self.round % self.eval_gap != 0:
            return
        
        self.baloon.deflate_if_inflated()
        self._move_to_gpu(self.diffusion_device, models=['diffusion', 'zeroshot'])
        for node in self.clients:
            node._move_to_gpu(node.device)
            self.node_metrics_log(node, suffix="_pre", train_splits = self.nodes_train_metrics_splits )
            node._move_to_cpu()

        self._move_to_cpu( models=['diffusion', 'zeroshot'] )
        

    def round_ending_hook(self):

        self.baloon.deflate_if_inflated()
        generated_images = {}
        if self.generate_nodes_images_frequency > 0 and self.round % self.generate_nodes_images_frequency == 0:
            if self.generate_embeddings_only:
                # Save embeddings only (no image generation)
                logger.info(f"Saving embeddings for all nodes to checkpoints (embeddings-only mode)")
                for node in self.clients:
                    node._move_to_gpu(node.device)
                    ### Disabilitata la metrica non potendo generare immagini
                    # self.node_metrics_log(node, suffix="_post", train_splits = self.nodes_train_metrics_splits)
                    # self.node_metrics_log(node,  test_splits = self.nodes_test_metrics_splits)
                    checkpoint_path = node.save_embeddings_to_checkpoint(
                        split='all',
                        checkpoint_dir=self.embeddings_checkpoint_dir,
                        round_num=self.round
                    )
                    logger.info(f"Node {node.id} embeddings saved to: {checkpoint_path}")
                    node._move_to_cpu()
            else:
                # Generate images as normal
                self._move_to_gpu(self.diffusion_device, models=['diffusion', 'zeroshot'])

                for node in self.clients:
                    node._move_to_gpu(node.device)
                    self.node_metrics_log(node, suffix="_post", train_splits = self.nodes_train_metrics_splits)
                    self.node_metrics_log(node,  test_splits = self.nodes_test_metrics_splits)

                    # Generate images using client's method (includes internal caching check)
                    client_result = node.generate_images(split='all', round_num=self.round)

                    # Convert from client format {'on_train': ..., 'on_val': ..., 'on_test': ...}
                    # to server format {'train': ..., 'val': ..., 'test': ...}
                    generated_images[node.id] = {}
                    if client_result.get('on_train') is not None:
                        generated_images[node.id]['train'] = client_result['on_train']
                    if client_result.get('on_val') is not None:
                        generated_images[node.id]['val'] = client_result['on_val']
                    if client_result.get('on_test') is not None:
                        generated_images[node.id]['test'] = client_result['on_test']

                    node._move_to_cpu()


        use_pretrained_generators = getattr(self.config.feda2v, 'use_pretrained_generators', False)
        if use_pretrained_generators and hasattr(self, 'client_synthetic_samples') and len(self.client_synthetic_samples) > 0:
            self.aggregate_synthetic_samples()

        if self.global_model_train and (self.config.feda2v.global_model_train_from_nodes_audio_embeddings or self.config.feda2v.global_model_train_from_nodes_adapters):
            self._move_to_gpu(self.device)

            if self.config.feda2v.global_model_train_from_nodes_audio_embeddings:
                loss = self.global_model_train_from_nodes_text_embeddings()
                self._move_to_cpu()
                print(f"\nGlobal Audio2Visual model trained from nodes embeddings with loss {loss:.4f}")

                # Save global adapter checkpoint after training from embeddings
                if self.adapter_save_checkpoint and self.round % self.adapter_checkpoint_frequency == 0:
                    logger.info(f"Saving global adapter checkpoint at round {self.round}")
                    self.save_adapter_checkpoint(round_num=self.round)

            if self.global_model_train_from_generator:
                self.train_generator_from_class_prompts()

            if self.global_model_train_from_nodes_adapters:
                logger.info(f"\n=== Round {self.round}: Training Generator and Global Adapters ===")
                result = self.global_model_train_from_nodes_adapters_output()

                # Handle return values
                if isinstance(result, tuple) and len(result) == 2:
                    generator_loss, adapter_loss = result
                    print(f"\nGenerator Loss: {generator_loss:.4f}, Adapter Fine-tuning Loss: {adapter_loss:.4f}")

                    # Validate generator if configured
                    validation_metrics = {}
                    if self.use_generator and self.round % self.generator_validation_frequency == 0:
                        logger.info(f"Validating generator at round {self.round}")
                        validation_metrics = self.validate_generator(self.nodes_per_class_adapters_outputs_means)

                    # Log to wandb
                    if not self.no_wandb:
                        log_dict = {
                            "server/generator_loss": generator_loss,
                            "server/adapter_finetuning_loss": adapter_loss,
                            "round": self.round
                        }

                        # Add validation metrics if available
                        if validation_metrics:
                            log_dict.update({
                                "server/generator_validation_similarity": validation_metrics.get('avg_cosine_similarity', 0),
                                "server/generator_validation_mse": validation_metrics.get('avg_mse_loss', 0),
                                "server/generator_validation_l1": validation_metrics.get('avg_l1_loss', 0),
                            })

                        self.data_log(log_dict)

                    # Save generator checkpoint if configured
                    if self.generator_save_checkpoint and self.round % self.generator_checkpoint_frequency == 0:
                        logger.info(f"Saving generator checkpoint at round {self.round}")
                        self.save_generator_checkpoint(round_num=self.round)

                    # Save global adapter checkpoint after training from nodes adapters
                    if self.adapter_save_checkpoint and self.round % self.adapter_checkpoint_frequency == 0:
                        logger.info(f"Saving global adapter checkpoint at round {self.round}")
                        self.save_adapter_checkpoint(round_num=self.round)

                else:
                    # Fallback for single value return
                    print(f"\nGlobal model trained with loss {result:.4f}")

                self._move_to_cpu()
        
        self.baloon.inflate_if_not_inflated()

        if self.generate_global_images_frequency > 0 and self.round % self.generate_global_images_frequency == 0:
            if self.generate_embeddings_only:
                # Save server embeddings only (no image generation)
                logger.info(f"Saving server embeddings to checkpoint (embeddings-only mode)")
                checkpoint_path = self.save_server_embeddings_to_checkpoint(
                    checkpoint_dir=self.embeddings_checkpoint_dir,
                    round_num=self.round
                )
                logger.info(f"Server embeddings saved to: {checkpoint_path}")
                generated_images['server'] = {}
            else:
                # Generate global images as normal
                generated_images['server'] = {}
                # Generate server images for all configured server splits
                for split_name in self.server_test_metrics_splits:
                    split_images = self.generate_global_images_with_aggregated_adapters(split=split_name)
                    if split_images is not None:
                        generated_images['server'][split_name] = split_images

        if self.optimize_memory_usage and self.global_model != None and self.global_model.diffusion_model != None:
                self.global_model.diffusion_model.to(torch.device('cpu'))

        self.federation_test_metric(generated_images)

        models_to_move = []
        if not self.generator_only_mode:
            models_to_move.append('diffusion')
            models_to_move.append('zeroshot')   

        self._move_to_cpu( models=models_to_move )

    def federation_test_metric ( self, generated_images):

        if len(generated_images) == 0:
            return
        
        for node in self.clients:
            node_id = node.id
            node = self.clients[node_id]
            if node_id not in generated_images:
                logger.warn(f"Node {node_id} not found in generated images")
                continue

            node_metrics = self.test_node_metrics_from_images(node, generated_images[node_id])
            print ( f"Node {node_id}\n{node_metrics}" )

            if not self.no_wandb:
                for split, metric in node_metrics.items():
                    wandb_metrics = node.log_metrics(metric, round=self.round)
                    # Include round for proper x-axis in wandb charts
                    wandb.log({**wandb_metrics, "round": self.round})

        if 'server' in generated_images:
            server_metrics = self.test_node_metrics_from_images(None, generated_images['server'])    
            print ( f"Server\n{server_metrics}" )

            if not self.no_wandb:
                for split, metric in server_metrics.items():
                    wandb_metrics = self.log_metrics(metric, round=self.round, suffix=f'_on_{split}')
                    # Include round for proper x-axis in wandb charts
                    wandb.log({**wandb_metrics, "round": self.round})


    def generate_global_images_average_text_embeddings_from_nodes(self):
        nodes_classes_text_embeddings = {}

        for node in self.clients:
            node_dataset = node.node_data.dataset.dataset if isinstance(node.node_data.dataset, torch.utils.data.Subset) else node.node_data.dataset
            node_classes = node_dataset.active_classes
            node_text_embeddings = node_dataset.text_embs
            print ( f"Node {node.id} classes {node_classes}")
            for node_class in node_classes:
                nodes_classes_text_embeddings.update({node_class: node_text_embeddings[node_class]})

        print ( f"Nodes text embeddings {nodes_classes_text_embeddings.keys()}")
        return

    def save_server_embeddings_to_checkpoint(self, checkpoint_dir='checkpoints/embeddings', round_num=None):
        """
        Save server's global embeddings to checkpoint by processing federation dataset splits
        using global adapters, similar to how nodes process their data.

        Args:
            checkpoint_dir: Directory to save checkpoint files
            round_num: Optional round number for naming

        Returns:
            Path to saved checkpoint file
        """
        import datetime
        from torch.utils.data import DataLoader

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Prepare checkpoint data structure
        checkpoint_data = {
            'node_id': 'server',
            'round': round_num if round_num is not None else -1,
            'timestamp': datetime.datetime.now().isoformat(),
            'embeddings': [],
            'metadata': {
                'diffusion_type': self.diffusion_type,
                'federation_classes': self.federation_available_classes
            }
        }

        logger.info("Collecting embeddings from federation dataset using global adapters")

        # Move models to device
        self._move_to_gpu(self.device, models=['adapter', 'ast'])

        try:
            # Process each split in server_node_data_splits
            if not hasattr(self, 'server_node_data_splits') or not self.server_node_data_splits:
                logger.warning("Server: No federation dataset available for embedding extraction")
                return None

            for split_name, node_data in self.server_node_data_splits.items():
                if node_data is None:
                    logger.warning(f"Server: No dataset available for split '{split_name}'")
                    continue

                # Get dataset for this split
                if split_name == 'train':
                    dataset = node_data.get_train_dataset()
                elif split_name == 'val':
                    dataset = node_data.get_val_dataset()
                elif split_name == 'test':
                    dataset = node_data.get_test_dataset()
                else:
                    logger.warning(f"Server: Unknown split name '{split_name}'")
                    continue

                if dataset is None or len(dataset) == 0:
                    logger.warning(f"Server: Dataset for split '{split_name}' is empty")
                    continue

                logger.info(f"Processing federation split '{split_name}' with {len(dataset)} samples")

                # Create DataLoader
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.args.batch_size if hasattr(self.args, 'batch_size') else 8,
                    shuffle=False,
                    num_workers=0
                )

                # Process batches
                embeddings_by_class = {}
                
                with torch.no_grad():
                    for batch_idx, batch in enumerate(dataloader):
                        # Extract audio data
                        if 'audio' not in batch or not isinstance(batch['audio'], torch.Tensor):
                            logger.warning(f"Server: Audio data not found in {split_name} batch {batch_idx}")
                            continue

                        audio_data = batch['audio'].to(self.device)
                        class_names = batch.get('class_name', [])

                        # Extract audio embeddings using AST model
                        if self.global_model.ast_model is None or self.global_model.ast_feature_extractor is None:
                            logger.warning("Server: AST model not initialized, skipping audio processing")
                            continue

                        # Convert audio to numpy if needed
                        if isinstance(audio_data, torch.Tensor):
                            audio_data_np = audio_data.to('cpu').numpy()
                        else:
                            audio_data_np = audio_data

                        # Extract features
                        audio_inputs = self.global_model.ast_feature_extractor(
                            audio_data_np,
                            sampling_rate=16000,
                            return_tensors="pt",
                            padding=True
                        ).input_values.to(self.device, self.global_model.torch_dtype)

                        self.global_model.ast_model.eval()
                        audio_embeddings = self.global_model.ast_model(audio_inputs).last_hidden_state

                        # Process through global adapters
                        adapter_outputs = {}
                        for adapter_name, adapter_module in self.global_adapters.items():
                            adapter_module.eval()
                            adapter_outputs[adapter_name] = adapter_module(audio_embeddings).detach().cpu()

                        # Group by class
                        for idx, class_name in enumerate(class_names):
                            if class_name not in embeddings_by_class:
                                embeddings_by_class[class_name] = {
                                    'clip': [],
                                    't5': [],
                                    'audio_embeddings': []
                                }

                            # Collect outputs from each adapter
                            if 'clip' in adapter_outputs:
                                embeddings_by_class[class_name]['clip'].append(
                                    adapter_outputs['clip'][idx].unsqueeze(0)
                                )
                            if 't5' in adapter_outputs:
                                embeddings_by_class[class_name]['t5'].append(
                                    adapter_outputs['t5'][idx].unsqueeze(0)
                                )

                            embeddings_by_class[class_name]['audio_embeddings'].append(
                                audio_embeddings[idx].detach().cpu().unsqueeze(0)
                            )

                # Store all embeddings per class (not just the mean)
                for class_name, emb_dict in embeddings_by_class.items():
                    try:
                        # Stack all embeddings for each type (keep all samples, not mean)
                        embedding_entry = {
                            'split': split_name,
                            'class_name': class_name,
                            'node_id': 'server',
                        }

                        for module_name in self.global_adapters.keys():
                            if module_name not in emb_dict:
                                continue
                            module_tensor = torch.cat(emb_dict[module_name], dim=0)
                            embedding_entry[module_name] = module_tensor.cpu()  # Initialize

                        if emb_dict['audio_embeddings']:
                            audio_tensor = torch.cat(emb_dict['audio_embeddings'], dim=0)
                            embedding_entry['audio_embeddings'] = audio_tensor.cpu()  # Keep all samples, not mean

                        checkpoint_data['embeddings'].append(embedding_entry)

                        logger.debug(f"Server: Processed class '{class_name}' for split '{split_name}' "
                                   f"(clip: {embedding_entry['clip'].shape if 'clip' in embedding_entry else 'N/A'}, "
                                   f"t5: {embedding_entry['t5'].shape if 't5' in embedding_entry else 'N/A'}, "
                                   f"audio: {embedding_entry['audio_embeddings'].shape if 'audio_embeddings' in embedding_entry else 'N/A'})")

                    except Exception as e:
                        logger.error(f"Server: Error processing class '{class_name}' for split '{split_name}': {e}")
                        continue

                logger.info(f"Server: Extracted embeddings for {len(embeddings_by_class)} classes from split '{split_name}'")

        except Exception as e:
            logger.error(f"Server: Error during embedding extraction: {e}")
            import traceback
            traceback.print_exc()
            return None

        finally:
            # Move models back to CPU
            self._move_to_cpu(models=['adapter', 'ast'])

        # Save checkpoint
        checkpoint_filename = f"server_embeddings_r{round_num if round_num is not None else 0}.pt"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"Server: Saved {len(checkpoint_data['embeddings'])} embeddings to {checkpoint_path}")

        return checkpoint_path

    def global_model_train_from_nodes_text_embeddings(self):
        self.global_model = self.global_model.to(self.device)

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
        """
        Train generator using adapter outputs from clients, then fine-tune global adapters.
        """
        self.global_model = self.global_model.to(self.device)

        # Step 1: Collect adapter outputs per class from all clients
        all_class_prompts = defaultdict(list)  # {class_name: [prompt_tensors from all clients]}

        for node_id, per_class_adapters_outputs in self.nodes_per_class_adapters_outputs_means.items():
            for class_name, adapters_dict in per_class_adapters_outputs.items():
                # adapters_dict contains {'clip': tensor, 't5': tensor, 'audio_embeddings': tensor}
                all_class_prompts[class_name].append({
                    'clip': adapters_dict.get('clip', None),
                    't5': adapters_dict.get('t5', None),
                    'audio_embeddings': adapters_dict.get('audio_embeddings', None)
                })

        if not all_class_prompts:
            logger.warning("No adapter outputs received from clients")
            return 0.0

        # Step 2: Train generator from class prompts (if enabled)
        generator_loss = 0.0
        if self.use_generator and self.prompt_generator is not None:
            logger.info(f"\nTraining {self.generator_type} generator from {len(all_class_prompts)} classes")
            generator_loss = self.train_generator_from_class_prompts(all_class_prompts)

        # Step 3: Fine-tune global adapters with generated prompts
        logger.info(f"\nFine-tuning global adapters")
        adapter_loss = self.finetune_adapters_with_prompts(all_class_prompts)

        return generator_loss, adapter_loss
    
    def global_model_step_per_node (self):
        pass

    def train_generator_from_class_prompts(self, all_class_prompts):
        """
        Train the VAE/GAN generator to replicate adapter prompts from clients with data augmentation.

        Args:
            all_class_prompts: dict {class_name: [{adapter_outputs}, ...]}

        Returns:
            Average generator loss
        """
        if self.prompt_generator is None:
            logger.warning("No generator initialized")
            return 0.0

        self.prompt_generator.train()
        total_loss = 0.0
        num_batches = 0
        num_epochs = self.generator_training_epochs if hasattr(self, 'generator_training_epochs') else self.global_model_train_epochs

        logger.info(f"Training generator for {num_epochs} epochs with augmentation={self.generator_augmentation}")

        from tqdm import tqdm
        epoch_pbar = tqdm(range(num_epochs), desc="Server Generator",
                         bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

        for epoch in epoch_pbar:
            epoch_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_kl_loss = 0.0
            epoch_sim_loss = 0.0
            epoch_batches = 0

            for class_name, prompts_list in all_class_prompts.items():
                # Filter and stack valid prompts
                clip_prompts = [p['clip'] for p in prompts_list if p['clip'] is not None]
                audio_embs = [p['audio_embeddings'] for p in prompts_list if p['audio_embeddings'] is not None]

                if not clip_prompts or not audio_embs:
                    continue

                # Stack tensors
                clip_prompts = torch.stack(clip_prompts).to(self.device)
                audio_embs = torch.stack(audio_embs).to(self.device)

                # Apply data augmentation (add Gaussian noise) after first epoch
                if epoch > 0 and self.generator_augmentation:
                    noise_scale = self.generator_augmentation_noise
                    audio_noise = torch.randn_like(audio_embs) * noise_scale
                    audio_embs = audio_embs + audio_noise

                    # Optional: also add small noise to CLIP embeddings
                    if self.generator_type == 'vae':
                        clip_noise = torch.randn_like(clip_prompts) * (noise_scale * 0.5)
                        clip_prompts = clip_prompts + clip_noise

                if self.generator_type == 'vae':
                    # VAE Training
                    self.generator_optimizer.zero_grad()

                    # Forward pass through VAE (conditioned or unconditioned)
                    if self.use_conditioned_vae:
                        recon_prompts, mu, logvar = self.prompt_generator(
                            audio_embs,
                            visual_condition=clip_prompts
                        )
                    else:
                        # Unconditioned VAE - no visual condition
                        recon_prompts, mu, logvar = self.prompt_generator(audio_embs)

                    # Compute VAE loss
                    total_vae_loss, recon_loss, kl_loss, sim_loss = self.generator_loss_fn(
                        recon_prompts,
                        audio_embs,
                        mu,
                        logvar,
                        epoch
                    )

                    # Check for NaN in loss before backward
                    if torch.isnan(total_vae_loss) or torch.isinf(total_vae_loss):
                        logger.error(f"NaN/Inf detected in loss at epoch {epoch}, batch {epoch_batches}")
                        logger.error(f"  recon_loss: {recon_loss.item()}, kl_loss: {kl_loss.item()}, sim_loss: {sim_loss.item()}")
                        logger.error(f"  mu stats: min={mu.min().item():.4f}, max={mu.max().item():.4f}, mean={mu.mean().item():.4f}")
                        logger.error(f"  logvar stats: min={logvar.min().item():.4f}, max={logvar.max().item():.4f}, mean={logvar.mean().item():.4f}")
                        continue  # Skip this batch

                    # Backward and optimize
                    total_vae_loss.backward()

                    # Gradient clipping for stability (increased from 1.0 to 5.0)
                    torch.nn.utils.clip_grad_norm_(self.prompt_generator.parameters(), max_norm=5.0)

                    self.generator_optimizer.step()

                    epoch_loss += total_vae_loss.item()
                    epoch_recon_loss += recon_loss.item()
                    epoch_kl_loss += kl_loss.item()
                    epoch_sim_loss += sim_loss.item()
                    epoch_batches += 1

                    # Update progress bar in real-time during training
                    if epoch_batches > 0:
                        avg_loss = epoch_loss / epoch_batches
                        avg_recon = epoch_recon_loss / epoch_batches
                        avg_kl = epoch_kl_loss / epoch_batches
                        avg_sim = epoch_sim_loss / epoch_batches
                        epoch_pbar.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'recon': f'{avg_recon:.4f}',
                            'kl': f'{avg_kl:.4f}',
                            'sim': f'{avg_sim:.4f}',
                            'class': class_name,
                            'batches': epoch_batches
                        })

                    if num_batches % 5 == 0:
                        logger.debug(f"  Epoch {epoch+1} Class '{class_name}': VAE Loss={total_vae_loss.item():.4f} "
                                  f"(Recon={recon_loss.item():.4f}, KL={kl_loss.item():.4f}, Sim={sim_loss.item():.4f})")

                elif self.generator_type == 'gan':
                    # GAN Training
                    gan_loss = self.train_gan_step(clip_prompts, audio_embs, class_name)
                    epoch_loss += gan_loss
                    epoch_batches += 1

                num_batches += 1

            if epoch_batches > 0:
                avg_epoch_loss = epoch_loss / epoch_batches
                avg_recon_loss = epoch_recon_loss / epoch_batches
                avg_kl_loss = epoch_kl_loss / epoch_batches
                avg_sim_loss = epoch_sim_loss / epoch_batches

                # Update progress bar with all loss components
                epoch_pbar.set_postfix({
                    'loss': f'{avg_epoch_loss:.4f}',
                    'recon': f'{avg_recon_loss:.4f}',
                    'kl': f'{avg_kl_loss:.4f}',
                    'sim': f'{avg_sim_loss:.4f}',
                    'batches': epoch_batches
                })

                total_loss += avg_epoch_loss

        epoch_pbar.close()

        avg_loss = total_loss / num_epochs if num_epochs > 0 else 0.0
        logger.info(f"Generator training completed. Average loss: {avg_loss:.4f}")

        return avg_loss

    def train_gan_step(self, clip_prompts, audio_embs, class_name):
        """
        Perform one GAN training step.

        Args:
            clip_prompts: Target CLIP embeddings
            audio_embs: Audio embeddings
            class_name: Name of the class being trained

        Returns:
            Total GAN loss
        """
        batch_size = clip_prompts.size(0)

        # Train Discriminator
        self.discriminator_optimizer.zero_grad()

        # Real samples
        real_output = self.discriminator_clip(clip_prompts)
        real_label = torch.ones_like(real_output)
        d_loss_real = F.binary_cross_entropy_with_logits(real_output, real_label)

        # Fake samples
        z = torch.randn(batch_size, 256).to(self.device)
        fake_prompts = self.prompt_generator_clip(z)
        fake_output = self.discriminator_clip(fake_prompts.detach())
        fake_label = torch.zeros_like(fake_output)
        d_loss_fake = F.binary_cross_entropy_with_logits(fake_output, fake_label)

        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.discriminator_optimizer.step()

        # Train Generator
        self.generator_optimizer.zero_grad()

        z = torch.randn(batch_size, 256).to(self.device)
        fake_prompts = self.prompt_generator_clip(z)
        fake_output = self.discriminator_clip(fake_prompts)
        g_loss = F.binary_cross_entropy_with_logits(fake_output, real_label)

        g_loss.backward()
        self.generator_optimizer.step()

        logger.debug(f"  Class '{class_name}': D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")

        return g_loss.item()

    def finetune_adapters_with_prompts(self, all_class_prompts):
        """
        Fine-tune global adapters using prompts from clients (and optionally generated ones).

        Args:
            all_class_prompts: dict {class_name: [{adapter_outputs}, ...]}

        Returns:
            Average fine-tuning loss
        """
        self.global_model.train()
        total_loss = 0.0
        num_samples = 0

        for class_name, prompts_list in all_class_prompts.items():
            # Get target prompts from clients
            target_clips = [p['clip'] for p in prompts_list if p['clip'] is not None]
            target_audios = [p['audio_embeddings'] for p in prompts_list if p['audio_embeddings'] is not None]

            if not target_clips or not target_audios:
                continue

            target_clip = torch.stack(target_clips).to(self.device)
            target_audio = torch.stack(target_audios).to(self.device)

            # Compute mean target for this class
            mean_target_clip = target_clip.mean(dim=0, keepdim=True)
            mean_target_audio = target_audio.mean(dim=0, keepdim=True)

            # Generate synthetic audio embeddings if generator is available
            if self.use_generator and self.prompt_generator is not None:
                self.prompt_generator.eval()
                with torch.no_grad():
                    num_synthetic = self.synthetic_samples_per_class if hasattr(self, 'synthetic_samples_per_class') else 5

                    if self.generator_type == 'vae':
                        # Use VAE to generate synthetic samples conditioned on visual features
                        synthetic_audio_embs = self.prompt_generator.sample(
                            num_samples=num_synthetic,
                            visual_condition=mean_target_clip,
                            device=self.device
                        )
                    elif self.generator_type == 'gan':
                        # For GAN, generate from random latent vectors
                        z = torch.randn(num_synthetic, 256).to(self.device)
                        if self.prompt_generator_clip is not None:
                            synthetic_audio_embs = self.prompt_generator_clip(z)
                        else:
                            logger.warning("GAN generator not properly initialized")
                            synthetic_audio_embs = mean_target_audio.repeat(num_synthetic, 1, 1)

                    # Combine real and synthetic embeddings
                    combined_audio = torch.cat([mean_target_audio, synthetic_audio_embs], dim=0)

                    logger.debug(f"  Class '{class_name}': Combined {mean_target_audio.shape[0]} real + {num_synthetic} synthetic samples")
            else:
                combined_audio = mean_target_audio

            # Fine-tune adapters
            for optimizer in self.global_optimizers.values():
                optimizer.zero_grad()

            # Forward through global adapters
            outputs = {}
            for adapter_name, adapter_module in self.global_adapters.items():
                outputs[adapter_name] = adapter_module(combined_audio)

            # Compute loss against target prompts
            loss = torch.tensor(0.0, device=self.device)
            if 'clip' in outputs and 'clip' in self.global_adapters:
                # Target should match the shape of the output
                target_expanded = mean_target_clip.expand_as(outputs['clip'][:1])
                loss += F.mse_loss(outputs['clip'][:1], target_expanded)

            # Backward and optimize
            if loss.item() > 0:
                loss.backward()
                for optimizer in self.global_optimizers.values():
                    optimizer.step()

                total_loss += loss.item()
                num_samples += 1

                logger.debug(f"  Class '{class_name}': Adapter fine-tuning loss = {loss.item():.4f}")

        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0
        logger.info(f"Adapter fine-tuning completed. Average loss: {avg_loss:.4f}")

        return avg_loss

    def global_model_create_images(self, audio_embeddings, num_images=1):
        prompt_embeds = audio_embeddings['t5'].to(self.global_model.diffusion_model_device).to(torch.bfloat16)
        pooled_prompt_embeds = audio_embeddings['clip'].to(self.global_model.diffusion_model_device).to(torch.bfloat16)

        imgs = self.global_model.diffusion_model(
                                        prompt_embeds= prompt_embeds,
                                        pooled_prompt_embeds=pooled_prompt_embeds,
                                        num_inference_steps=1,
                                        output_type="pt",
                                        ).images
        
        return imgs

    def save_audio_embeddings(self, file_name="audio_embeddings.pt"):
        """Save audio embeddings from clients to a file and AST cache."""
        all_audio_embeddings = {}

        for client in self.clients:
            if hasattr(client, 'audio_embedding_store') and client.audio_embedding_store is not None:
                all_audio_embeddings.update(client.audio_embedding_store)

        # Save to legacy .pt file for backward compatibility
        if len(all_audio_embeddings) > 0:
            torch.save(all_audio_embeddings, file_name)
            logger.info(f"Saved {len(all_audio_embeddings)} audio embeddings to legacy file: {file_name}")

            # Also save to AST cache for VEGAS datasets
            for client in self.clients:
                if client.dataset == "VEGAS":
                    # Get the dataset (handle both direct and Subset)
                    if isinstance(client.node_data.train_dataset, torch.utils.data.Subset):
                        dataset = client.node_data.train_dataset.dataset
                    else:
                        dataset = client.node_data.train_dataset

                    # Save to AST cache if it's a VEGASDataset with cache enabled
                    if isinstance(dataset, VEGASDataset) and dataset.enable_ast_cache:
                        # AST cache configuration - must match extraction parameters
                        ast_sample_rate = 16000
                        ast_duration = 5.0
                        ast_model_name = "ast-finetuned"

                        logger.info(f"Saving audio embeddings to AST cache for client {client.id}...")

                        # Filter embeddings for this client's classes
                        client_embeddings = {}
                        for emb_key, emb_value in all_audio_embeddings.items():
                            # Check if this embedding belongs to client's classes
                            if ':' in emb_key:
                                _, class_name = emb_key.split(':', 1)
                                if class_name.lower() in [c.lower() for c in dataset.active_classes.keys()]:
                                    client_embeddings[emb_key] = emb_value

                        if len(client_embeddings) > 0:
                            success = dataset.save_ast_embeddings_to_cache(
                                embeddings=client_embeddings,
                                sample_rate=ast_sample_rate,
                                duration=ast_duration,
                                model_name=ast_model_name
                            )

                            if success:
                                logger.info(f"✓ Saved {len(client_embeddings)} embeddings to AST cache for client {client.id}")
                            else:
                                logger.warning(f"Failed to save AST cache for client {client.id}")

                        # Only save once per unique dataset configuration
                        break
        else:
            logger.warning("No audio embeddings to save")
        
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

    def save_generator_checkpoint(self, round_num=None):
        """
        Save generator model checkpoint with comprehensive metadata.
        Supports both unified and multiple generators (per_class/per_group).

        Args:
            round_num: Optional round number to include in checkpoint filename
        """
        import datetime

        # Check if we have generators to save
        has_unified_generator = hasattr(self, 'prompt_generator') and self.prompt_generator is not None
        has_multiple_generators = hasattr(self, 'prompt_generators') and self.prompt_generators

        if not self.use_generator or (not has_unified_generator and not has_multiple_generators):
            logger.warning("Generator not initialized, cannot save checkpoint")
            return

        if not self.generator_save_checkpoint:
            return

        # Create checkpoint directory if needed
        if not os.path.exists(self.generator_checkpoint_dir):
            os.makedirs(self.generator_checkpoint_dir, exist_ok=True)

        # Build checkpoint path with flexible naming: base_name_server[_round_Y].pt
        if round_num is not None:
            checkpoint_path = os.path.join(
                self.generator_checkpoint_dir,
                f'{self.generator_checkpoint_base_name}_round_{round_num}.pt'
            )
        else:
            checkpoint_path = os.path.join(
                self.generator_checkpoint_dir,
                f'{self.generator_checkpoint_base_name}.pt'
            )

        # Collect aggregated metadata from all clients
        num_clients = len(self.clients)
        all_datasets = set()
        all_classes = set()
        client_metadata = []

        for client in self.clients:
            client_info = {
                'client_id': client.id,
                'dataset': getattr(client, 'dataset_name', None),
            }

            # Try to get selected classes from client
            if hasattr(client, 'node_data') and hasattr(client.node_data, 'train_dataset'):
                train_dataset = client.node_data.train_dataset
                if hasattr(train_dataset, 'selected_classes'):
                    client_info['selected_classes'] = train_dataset.selected_classes
                    if train_dataset.selected_classes:
                        all_classes.update(train_dataset.selected_classes)
                elif hasattr(train_dataset, 'dataset') and hasattr(train_dataset.dataset, 'selected_classes'):
                    client_info['selected_classes'] = train_dataset.dataset.selected_classes
                    if train_dataset.dataset.selected_classes:
                        all_classes.update(train_dataset.dataset.selected_classes)

            if client_info['dataset']:
                all_datasets.add(client_info['dataset'])

            client_metadata.append(client_info)

        # Create comprehensive checkpoint metadata
        checkpoint = {
            # Server identification
            'checkpoint_type': 'server',
            'is_global': True,

            # Training state
            'round': round_num if round_num is not None else self.round,
            'timestamp': datetime.datetime.now().isoformat(),

            # Generator configuration
            'generator_type': self.generator_type,
            'diffusion_type': self.diffusion_type,
            'generator_training_epochs': self.generator_training_epochs,
            'synthetic_samples_per_class': self.synthetic_samples_per_class,

            # Federation metadata
            'num_clients': num_clients,
            'client_metadata': client_metadata,
            'federated_datasets': sorted(list(all_datasets)),
            'federated_classes': sorted(list(all_classes)) if all_classes else None,

            # Model architecture info
            'audio_model_name': self.audio_model_name,
            'img_pipe_name': self.img_pipe_name,

            # Training configuration
            'global_model_train': self.global_model_train,
            'global_model_train_from_nodes_adapters': getattr(self.config.feda2v, 'global_model_train_from_nodes_adapters', None),
        }

        # Save generator state based on whether we have unified or multiple generators
        if has_multiple_generators:
            # Save multiple generators (per_class or per_group)
            prompt_generators_state = {}
            generator_optimizers_state = {}
            generator_keys_list = list(self.prompt_generators.keys())

            for gen_key, generator in self.prompt_generators.items():
                prompt_generators_state[gen_key] = generator.state_dict()
                if hasattr(self, 'generator_optimizers') and gen_key in self.generator_optimizers:
                    generator_optimizers_state[gen_key] = self.generator_optimizers[gen_key].state_dict()

            checkpoint.update({
                'prompt_generators': prompt_generators_state,
                'generator_optimizers': generator_optimizers_state if generator_optimizers_state else None,
                'use_conditioned_vae': getattr(self, 'use_conditioned_vae', False),
                'sequence_length': getattr(self, 'generator_training_sequence_length', 4),
                'generator_keys': generator_keys_list,  # List of all generator keys (classes/groups)
                'num_generators': len(prompt_generators_state),
            })
            logger.info(f"Saved {len(prompt_generators_state)} generators to checkpoint")
            logger.info(f"Generator keys: {sorted(generator_keys_list)}")

        elif has_unified_generator:
            # Save single unified generator
            if self.generator_type == 'vae':
                checkpoint.update({
                    'generator_state_dict': self.prompt_generator.state_dict(),
                    'optimizer_state_dict': self.generator_optimizer.state_dict() if self.generator_optimizer else None,
                })
            elif self.generator_type == 'gan':
                checkpoint.update({
                    'generator_clip_state_dict': self.prompt_generator_clip.state_dict() if self.prompt_generator_clip else None,
                    'generator_t5_state_dict': self.prompt_generator_t5.state_dict() if self.prompt_generator_t5 else None,
                    'discriminator_clip_state_dict': self.discriminator_clip.state_dict() if self.discriminator_clip else None,
                    'discriminator_t5_state_dict': self.discriminator_t5.state_dict() if self.discriminator_t5 else None,
                    'generator_optimizer_state_dict': self.generator_optimizer.state_dict() if self.generator_optimizer else None,
                    'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict() if hasattr(self, 'discriminator_optimizer') and self.discriminator_optimizer else None,
                })

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved generator checkpoint to {checkpoint_path}")

    def load_generator_checkpoint(self, checkpoint_path=None, strict_validation=True, warn_only=False):
        """
        Load generator model checkpoint with metadata validation.

        Args:
            checkpoint_path: Optional path to checkpoint. If None, uses flexible naming system
            strict_validation: If True, reject checkpoint on critical metadata mismatch
            warn_only: If True, only print warnings without rejecting checkpoint

        Returns:
            bool: True if checkpoint loaded successfully, False otherwise
        """
        if not self.use_generator:
            logger.warning("Generator not enabled, skipping checkpoint load")
            return False

        if not self.generator_load_checkpoint:
            return False

        # Use provided path or search for checkpoints automatically
        if checkpoint_path is None:
            # Search for checkpoint files matching the pattern
            import glob

            # First, try to find checkpoints with round numbers
            pattern_with_round = os.path.join(
                self.generator_checkpoint_dir,
                f'{self.generator_checkpoint_base_name}*_round_*.pt'
            )
            checkpoint_files = glob.glob(pattern_with_round)

            if checkpoint_files:
                # Extract round numbers and find the latest
                rounds = []
                for f in checkpoint_files:
                    # Extract round number from filename
                    import re
                    match = re.search(r'_round_(\d+)\.pt$', f)
                    if match:
                        rounds.append(int(match.group(1)))

                if rounds:
                    latest_round = max(rounds)
                    logger.info(f"Found checkpoints up to round {latest_round}")

                    # Build pattern for latest round
                    pattern_latest_round = os.path.join(
                        self.generator_checkpoint_dir,
                        f'{self.generator_checkpoint_base_name}*_round_{latest_round}.pt'
                    )
                    checkpoint_files_latest = glob.glob(pattern_latest_round)

                    if len(checkpoint_files_latest) == 1:
                        # Single checkpoint file (unified generator)
                        checkpoint_path = checkpoint_files_latest[0]
                        logger.info(f"Loading unified generator from round {latest_round}")
                    else:
                        # Multiple checkpoint files (per_class or per_group)
                        logger.info(f"Found {len(checkpoint_files_latest)} checkpoint files for round {latest_round}")
                        return self._load_multiple_generator_checkpoints(checkpoint_files_latest, strict_validation, warn_only)
                else:
                    logger.warning("No valid round numbers found in checkpoint filenames")
                    return False
            else:
                # Try without round number
                pattern_no_round = os.path.join(
                    self.generator_checkpoint_dir,
                    f'{self.generator_checkpoint_base_name}*.pt'
                )
                checkpoint_files = glob.glob(pattern_no_round)

                if not checkpoint_files:
                    logger.warning(f"No generator checkpoints found in {self.generator_checkpoint_dir}")
                    return False
                elif len(checkpoint_files) == 1:
                    checkpoint_path = checkpoint_files[0]
                    logger.info(f"Loading generator from {checkpoint_path}")
                else:
                    # Multiple files without round numbers
                    logger.info(f"Found {len(checkpoint_files)} checkpoint files")
                    return self._load_multiple_generator_checkpoints(checkpoint_files, strict_validation, warn_only)

        if checkpoint_path and not os.path.exists(checkpoint_path):
            logger.warning(f"Generator checkpoint not found at {checkpoint_path}")
            return False

        try:
            # Load checkpoint to CPU (generators run on CPU for memory efficiency)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Display checkpoint metadata
            logger.info(f"\n{'='*60}")
            logger.info(f"Loading checkpoint from: {checkpoint_path}")
            logger.info(f"{'='*60}")
            logger.info(f"Checkpoint metadata:")
            logger.info(f"  - Type: {checkpoint.get('checkpoint_type', 'client/legacy')}")
            logger.info(f"  - Round: {checkpoint.get('round', 'N/A')}")
            logger.info(f"  - Timestamp: {checkpoint.get('timestamp', 'N/A')}")
            logger.info(f"  - Generator type: {checkpoint.get('generator_type', 'N/A')}")

            if checkpoint.get('checkpoint_type') == 'server':
                logger.info(f"  - Num clients: {checkpoint.get('num_clients', 'N/A')}")
                logger.info(f"  - Federated datasets: {checkpoint.get('federated_datasets', 'N/A')}")
                logger.info(f"  - Federated classes: {checkpoint.get('federated_classes', 'N/A')}")
            else:
                logger.info(f"  - Node ID: {checkpoint.get('node_id', checkpoint.get('client_id', 'N/A'))}")
                logger.info(f"  - Dataset: {checkpoint.get('dataset_name', 'N/A')}")
                logger.info(f"  - Selected classes: {checkpoint.get('selected_classes', 'N/A')}")

            # Validation
            validation_errors = []
            validation_warnings = []

            # Validate generator type (critical)
            if 'generator_type' in checkpoint:
                if checkpoint['generator_type'] != self.generator_type:
                    msg = f"Generator type mismatch: checkpoint={checkpoint['generator_type']}, current={self.generator_type}"
                    validation_errors.append(msg)

            # Validate diffusion type
            if 'diffusion_type' in checkpoint:
                if checkpoint['diffusion_type'] != self.diffusion_type:
                    msg = f"Diffusion type mismatch: checkpoint={checkpoint['diffusion_type']}, current={self.diffusion_type}"
                    validation_warnings.append(msg)

            # Validate it's a server checkpoint or compatible
            if checkpoint.get('checkpoint_type') == 'client':
                msg = "Loading a client checkpoint into server - may not be fully compatible"
                validation_warnings.append(msg)

            # Print validation results
            if validation_errors:
                logger.error(f"\nValidation ERRORS:")
                for error in validation_errors:
                    logger.error(f"  ✗ {error}")

            if validation_warnings:
                logger.warning(f"\nValidation WARNINGS:")
                for warning in validation_warnings:
                    logger.warning(f"  ⚠ {warning}")

            # Decide whether to reject
            if validation_errors and strict_validation and not warn_only:
                logger.error(f"\nCheckpoint validation failed. Set strict_validation=False to load anyway.")
                return False

            if not validation_errors and not validation_warnings:
                logger.info("✓ Checkpoint validation passed")

            logger.info(f"{'='*60}\n")

            # Check if checkpoint contains multiple generators (per_class or per_group)
            has_multiple_generators = 'prompt_generators' in checkpoint

            if has_multiple_generators:
                # Initialize prompt_generators dictionary if not already present
                if not hasattr(self, 'prompt_generators'):
                    self.prompt_generators = {}
                if not hasattr(self, 'generator_optimizers'):
                    self.generator_optimizers = {}

                # Load multiple generators from checkpoint
                logger.info(f"Loading {len(checkpoint['prompt_generators'])} generators from checkpoint")

                for gen_key, gen_state_dict in checkpoint['prompt_generators'].items():
                    # Create generator instance based on type
                    if self.generator_type == 'vae':
                        from flcore.trainmodel.generators import ConditionedVAEGenerator, VAEGenerator

                        # Determine visual_dim from checkpoint or use default
                        visual_dim = checkpoint.get('visual_dim', 768)

                        if checkpoint.get('use_conditioned_vae', False):
                            generator = ConditionedVAEGenerator(
                                input_dim=768,
                                visual_dim=visual_dim,
                                hidden_dim=512,
                                latent_dim=256,
                                sequence_length=checkpoint.get('sequence_length', 4)
                            )
                        else:
                            generator = VAEGenerator(
                                input_dim=768,
                                hidden_dim=1024,
                                latent_dim=256,
                                sequence_length=checkpoint.get('sequence_length', 4)
                            )

                        # Keep generator in CPU
                        generator = generator.to('cpu')
                        generator.load_state_dict(gen_state_dict)
                        self.prompt_generators[gen_key] = generator
                        logger.info(f"Loaded VAE generator for '{gen_key}'")

                        # Load optimizer for this generator if available
                        if 'generator_optimizers' in checkpoint and gen_key in checkpoint['generator_optimizers']:
                            optimizer = torch.optim.AdamW(
                                generator.parameters(),
                                lr=self.config.training.learning_rate * 0.1
                            )
                            optimizer.load_state_dict(checkpoint['generator_optimizers'][gen_key])
                            self.generator_optimizers[gen_key] = optimizer
                            logger.info(f"Loaded optimizer for generator '{gen_key}'")

                # Initialize loss function if not already done
                if not hasattr(self, 'generator_loss_fn') or self.generator_loss_fn is None:
                    from flcore.trainmodel.generators import VAELoss
                    self.generator_loss_fn = VAELoss(
                        total_epochs=self.generator_training_epochs,
                        beta_warmup_ratio=0.5
                    )

            else:
                # Single unified generator - convert to dictionary format
                # Initialize prompt_generators dictionary if not already present
                if not hasattr(self, 'prompt_generators'):
                    self.prompt_generators = {}
                if not hasattr(self, 'generator_optimizers'):
                    self.generator_optimizers = {}

                # Load state dictionaries
                if self.generator_type == 'vae':
                    if 'generator_state_dict' in checkpoint:
                        from flcore.trainmodel.generators import ConditionedVAEGenerator, VAEGenerator

                        # Determine parameters from checkpoint or config
                        visual_dim = checkpoint.get('visual_dim', 768)
                        use_conditioned = checkpoint.get('use_conditioned_vae', getattr(self, 'use_conditioned_vae', False))
                        sequence_length = checkpoint.get('sequence_length', getattr(self, 'generator_training_sequence_length', 4))

                        # Create generator instance
                        if use_conditioned:
                            generator = ConditionedVAEGenerator(
                                input_dim=768,
                                visual_dim=visual_dim,
                                hidden_dim=512,
                                latent_dim=256,
                                sequence_length=sequence_length
                            )
                        else:
                            generator = VAEGenerator(
                                input_dim=768,
                                hidden_dim=1024,
                                latent_dim=256,
                                sequence_length=sequence_length
                            )

                        # Keep in CPU
                        generator = generator.to('cpu')
                        generator.load_state_dict(checkpoint['generator_state_dict'])

                        # Get classes from checkpoint metadata
                        generator_classes = checkpoint.get('generator_classes', None)
                        selected_classes = checkpoint.get('selected_classes', None)
                        generator_key = checkpoint.get('generator_key', 'unified')

                        # Determine which classes to register
                        classes_to_serve = []
                        if generator_classes:
                            classes_to_serve = generator_classes
                        elif selected_classes:
                            classes_to_serve = selected_classes
                        else:
                            # No class info - use a generic key
                            classes_to_serve = [generator_key]

                        logger.info(f"Loaded unified VAE generator serving {len(classes_to_serve)} classes: {classes_to_serve}")

                        # Register generator for each class it serves (only if class is active)
                        for class_key in classes_to_serve:
                            # Check if this class is in active classes
                            if hasattr(self, 'federation_active_classes') and self.federation_active_classes:
                                if class_key not in self.federation_active_classes:
                                    logger.info(f"  ✗ Skipping registration for class '{class_key}' (not in active classes)")
                                    continue

                            self.prompt_generators[class_key] = generator
                            logger.info(f"  → Registered generator for class '{class_key}'")

                    if self.generator_training_mode and 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
                        generator_key = checkpoint.get('generator_key', 'unified')
                        optimizer = torch.optim.AdamW(
                            generator.parameters(),
                            lr=self.config.training.learning_rate * 0.1
                        )
                        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        self.generator_optimizers[generator_key] = optimizer

                        # Also set legacy reference if this is the unified generator
                        if generator_key == 'unified':
                            self.generator_optimizer = optimizer

                        logger.info(f"  → Loaded optimizer for '{generator_key}'")

                elif self.generator_type == 'gan':
                    if 'generator_clip_state_dict' in checkpoint and self.prompt_generator_clip is not None:
                        self.prompt_generator_clip.load_state_dict(checkpoint['generator_clip_state_dict'])
                        logger.info("Loaded GAN CLIP generator state")

                    if 'generator_t5_state_dict' in checkpoint and self.prompt_generator_t5 is not None:
                        self.prompt_generator_t5.load_state_dict(checkpoint['generator_t5_state_dict'])
                        logger.info("Loaded GAN T5 generator state")

                    if 'discriminator_clip_state_dict' in checkpoint and self.discriminator_clip is not None:
                        self.discriminator_clip.load_state_dict(checkpoint['discriminator_clip_state_dict'])
                        logger.info("Loaded GAN CLIP discriminator state")

                    if 'discriminator_t5_state_dict' in checkpoint and self.discriminator_t5 is not None:
                        self.discriminator_t5.load_state_dict(checkpoint['discriminator_t5_state_dict'])
                        logger.info("Loaded GAN T5 discriminator state")

                    if 'generator_optimizer_state_dict' in checkpoint and checkpoint['generator_optimizer_state_dict'] is not None:
                        if self.generator_optimizer is not None:
                            self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])

                            # Also store in generator_optimizers dict with key 'unified'
                            if not hasattr(self, 'generator_optimizers'):
                                self.generator_optimizers = {}
                            self.generator_optimizers['unified'] = self.generator_optimizer

                            logger.info("Loaded GAN generator optimizer state")

                    if 'discriminator_optimizer_state_dict' in checkpoint and checkpoint['discriminator_optimizer_state_dict'] is not None:
                        if hasattr(self, 'discriminator_optimizer') and self.discriminator_optimizer is not None:
                            self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
                            logger.info("Loaded GAN discriminator optimizer state")

            # Freeze generators if training is not enabled
            if hasattr(self, 'prompt_generators') and self.prompt_generators:
                if self.generator_training_mode:
                    logger.info("Generator training is disabled - freezing all generators")
                    for class_key, generator in self.prompt_generators.items():
                        for param in generator.parameters():
                            param.requires_grad = False
                        logger.info(f"  → Frozen generator for class '{class_key}'")

            logger.info(f"Successfully loaded generator checkpoint from round {checkpoint.get('round', 'unknown')}")
            return True

        except Exception as e:
            logger.error(f"Error loading generator checkpoint: {e}")
            return False

    def _load_multiple_generator_checkpoints(self, checkpoint_files, strict_validation=True, warn_only=False):
        """
        Load multiple generator checkpoints (per_class or per_group granularity).

        Args:
            checkpoint_files: List of checkpoint file paths
            strict_validation: If True, reject checkpoint on critical metadata mismatch
            warn_only: If True, only print warnings without rejecting checkpoint

        Returns:
            bool: True if all checkpoints loaded successfully, False otherwise
        """
        import re

        logger.info(f"Found {len(checkpoint_files)} generator checkpoint files")

        # Filter checkpoint files to only load generators for classes in federation_active_classes
        if hasattr(self, 'federation_active_classes') and self.federation_active_classes:
            # Get list of active class names (keys from the dictionary)
            active_class_names = list(self.federation_active_classes.keys())
            logger.info(f"Filtering checkpoints for {len(active_class_names)} active classes: {active_class_names}")

            filtered_files = []
            for checkpoint_file in checkpoint_files:
                filename = os.path.basename(checkpoint_file)
                # Try to extract class name from filename
                class_match = re.search(r'_class_(.+?)(?:_round_\d+)?\.pt$', filename)
                if class_match:
                    class_name = class_match.group(1)
                    if class_name in active_class_names:
                        filtered_files.append(checkpoint_file)
                        logger.info(f"  ✓ Including checkpoint for class '{class_name}'")
                    else:
                        logger.info(f"  ✗ Skipping checkpoint for class '{class_name}' (not in active classes)")
                else:
                    # If we can't extract class name, include it (might be a group or unified generator)
                    filtered_files.append(checkpoint_file)
                    logger.info(f"  ? Including checkpoint {filename} (cannot determine class)")

            checkpoint_files = filtered_files
            logger.info(f"After filtering: {len(checkpoint_files)} checkpoint files to load")

        if not checkpoint_files:
            logger.warning("No checkpoint files to load after filtering")
            return False

        logger.info(f"Loading {len(checkpoint_files)} generator checkpoints")

        # Initialize dictionaries for multiple generators
        if not hasattr(self, 'prompt_generators'):
            self.prompt_generators = {}
        if not hasattr(self, 'generator_optimizers'):
            self.generator_optimizers = {}

        loaded_count = 0
        failed_count = 0

        for checkpoint_file in checkpoint_files:
            try:
                # Extract generator key from filename
                # Pattern: {base_name}_node{id}_class_{class_name}_round_{round}.pt
                # or: {base_name}_node{id}_group_{group_name}_round_{round}.pt
                filename = os.path.basename(checkpoint_file)

                # Try to extract class or group name from filename
                # Patterns supported:
                # - {base_name}_node{id}_class_{class_name}_round_{round}.pt
                # - {base_name}_node{id}_class_{class_name}.pt
                # - {base_name}_node{id}_group_{group_name}_round_{round}.pt
                # - {base_name}_node{id}_group_{group_name}.pt
                gen_key = None
                granularity_type = None

                # Try class pattern first (matches everything after "_class_" until "_round_" or ".pt")
                class_match = re.search(r'_class_(.+?)(?:_round_\d+)?\.pt$', filename)
                if class_match:
                    # gen_key = class_match.group(1).replace('_', ' ')
                    gen_key = class_match.group(1)
                    granularity_type = 'class'
                else:
                    # Try group pattern
                    group_match = re.search(r'_group_([^_\.]+?)(?:_round_\d+)?\.pt$', filename)
                    if group_match:
                        gen_key = group_match.group(1)
                        granularity_type = 'group'

                if gen_key:
                    logger.info(f"Loading generator for {granularity_type} '{gen_key}' from {filename}")
                else:
                    logger.warning(f"Could not extract generator key from filename: {filename}")
                    logger.warning(f"Expected pattern: *_class_{{name}}_round_{{N}}.pt or *_group_{{name}}_round_{{N}}.pt")
                    continue

                # Load checkpoint
                checkpoint = torch.load(checkpoint_file, map_location='cpu')

                # Verify/extract generator key from checkpoint metadata
                # Priority: metadata > filename
                metadata_gen_key = checkpoint.get('generator_key', None)
                generator_classes = checkpoint.get('generator_classes', None)  # Classes handled by THIS generator
                selected_classes = checkpoint.get('selected_classes', None)    # All classes of the node
                granularity = checkpoint.get('generator_granularity', None)
                class_name = checkpoint.get('class_name', None)
                group_name = checkpoint.get('group_name', None)

                # Determine generator key based on granularity and metadata
                final_gen_key = gen_key  # Start with filename-based key

                if metadata_gen_key:
                    # Explicit generator_key in metadata (most reliable)
                    final_gen_key = metadata_gen_key
                    logger.info(f"  Using generator_key from metadata: '{metadata_gen_key}'")
                elif granularity == 'per_class' and class_name:
                    # Per-class granularity: use class_name field
                    final_gen_key = class_name
                    logger.info(f"  Using class_name from metadata (per_class): '{class_name}'")
                elif granularity == 'per_group' and group_name:
                    # Per-group granularity: use group_name field
                    final_gen_key = group_name
                    logger.info(f"  Using group_name from metadata (per_group): '{group_name}'")
                elif generator_classes and len(generator_classes) == 1:
                    # Single class in generator_classes - use it as key
                    final_gen_key = generator_classes[0]
                    logger.info(f"  Using single class from generator_classes: '{final_gen_key}'")
                elif gen_key:
                    # Fallback to filename-based key
                    logger.info(f"  Using generator key from filename: '{gen_key}'")
                else:
                    logger.warning(f"Could not determine generator key from metadata or filename")
                    logger.warning(f"Metadata: generator_key={metadata_gen_key}, class_name={class_name}, group_name={group_name}, generator_classes={generator_classes}")
                    failed_count += 1
                    continue

                # Update gen_key with final decision
                gen_key = final_gen_key

                # Display checkpoint metadata
                logger.info(f"  - Type: {checkpoint.get('checkpoint_type', 'N/A')}")
                logger.info(f"  - Round: {checkpoint.get('round', 'N/A')}")
                logger.info(f"  - Generator type: {checkpoint.get('generator_type', 'N/A')}")
                logger.info(f"  - Granularity: {granularity if granularity else 'N/A'}")
                if granularity == 'per_class':
                    logger.info(f"  - Class name: {class_name}")
                elif granularity == 'per_group':
                    logger.info(f"  - Group name: {group_name}")
                    logger.info(f"  - Generator classes: {generator_classes}")
                logger.info(f"  - Node classes: {selected_classes if selected_classes else 'N/A'}")

                # Validation
                validation_errors = []

                # Validate generator type (critical)
                if 'generator_type' in checkpoint:
                    if checkpoint['generator_type'] != self.generator_type:
                        msg = f"Generator type mismatch: checkpoint={checkpoint['generator_type']}, current={self.generator_type}"
                        validation_errors.append(msg)

                if validation_errors and strict_validation and not warn_only:
                    logger.error(f"Checkpoint validation failed for {gen_key}: {validation_errors}")
                    failed_count += 1
                    continue

                # Create generator instance based on type
                if self.generator_type == 'vae':

                    # Get parameters from checkpoint
                    visual_dim = checkpoint.get('visual_dim', 768)
                    use_conditioned = checkpoint.get('use_conditioned_vae', False)
                    sequence_length = checkpoint.get('sequence_length', 1214)

                    if use_conditioned:
                        generator = ConditionedVAEGenerator(
                            input_dim=768,
                            visual_dim=visual_dim,
                            hidden_dim=512,
                            latent_dim=256,
                            sequence_length=sequence_length
                        )
                    else:
                        generator = VAEGenerator(
                            input_dim=768,
                            hidden_dim=1024,
                            latent_dim=256,
                            sequence_length=sequence_length
                        )

                    # Keep in CPU
                    generator = generator.to('cpu')

                    # #FIX ME
                    # logger.warn( "SKIPPING GENRATOR LOAD STATE DICT FOR DEBUGGING PURPOSES")
                    # loaded_count += 1
                    # continue
                    # Load state dict
                    if 'generator_state_dict' in checkpoint:
                        generator.load_state_dict(checkpoint['generator_state_dict'])

                        # Determine which classes this generator should serve
                        # If generator_classes is specified, create a reference for EACH class
                        # Otherwise, use only the gen_key
                        classes_to_serve = []

                        if generator_classes and len(generator_classes) > 0:
                            # This generator serves multiple classes (e.g., per_group)
                            # Create a reference for each class
                            classes_to_serve = generator_classes
                            logger.info(f"✓ Loaded VAE generator for key '{gen_key}' serving {len(generator_classes)} classes: {generator_classes}")
                        else:
                            # This generator serves a single class (e.g., per_class)
                            classes_to_serve = [gen_key]
                            logger.info(f"✓ Loaded VAE generator for '{gen_key}'")

                        # Register generator for each class it serves (only if class is active)
                        for class_key in classes_to_serve:
                            # Check if this class is in active classes
                            if hasattr(self, 'federation_active_classes') and self.federation_active_classes:
                                if class_key not in self.federation_active_classes:
                                    logger.info(f"  ✗ Skipping registration for class '{class_key}' (not in active classes)")
                                    continue

                            self.prompt_generators[class_key] = generator
                            logger.info(f"  → Registered generator for class '{class_key}'")

                        # Load optimizer if available (store under the primary gen_key)
                        if 'optimizer_state_dict' in checkpoint and checkpoint['optimizer_state_dict'] is not None:
                            optimizer = torch.optim.AdamW(
                                generator.parameters(),
                                lr=self.config.training.learning_rate * 0.1
                            )
                            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                            self.generator_optimizers[gen_key] = optimizer
                            logger.info(f"  → Loaded optimizer for '{gen_key}'")

                        loaded_count += 1
                    else:
                        logger.warning(f"No generator_state_dict found in checkpoint for {gen_key}")
                        failed_count += 1

                elif self.generator_type == 'gan':
                    logger.warning("Multiple GAN generators not yet supported in server")
                    failed_count += 1

            except Exception as e:
                logger.error(f"Error loading checkpoint {checkpoint_file}: {e}")
                failed_count += 1

        # Initialize loss function if not already done
        if not hasattr(self, 'generator_loss_fn') or self.generator_loss_fn is None:
            from flcore.trainmodel.generators import VAELoss
            self.generator_loss_fn = VAELoss(
                total_epochs=self.generator_training_epochs,
                beta_warmup_ratio=0.5
            )

        logger.info(f"\n{'='*60}")
        logger.info(f"Generator checkpoint loading summary:")
        logger.info(f"  - Checkpoint files loaded: {loaded_count}")
        logger.info(f"  - Failed: {failed_count}")

        if self.prompt_generators:
            # Count unique generator instances
            unique_generators = len(set(id(gen) for gen in self.prompt_generators.values()))
            logger.info(f"  - Unique generator instances: {unique_generators}")
            logger.info(f"  - Total class keys served: {len(self.prompt_generators)}")
            logger.info(f"  - Classes with generators: {sorted(list(self.prompt_generators.keys()))}")

            # Freeze generators if training is not enabled
            generator_training_enabled = getattr(self, 'generator_training_mode', False) or getattr(self, 'generator_only_mode', False)
            if not generator_training_enabled:
                logger.info(f"\nGenerator training is disabled - freezing all generators")
                # Track unique generators to avoid freezing the same instance multiple times
                frozen_generators = set()
                for class_key, generator in self.prompt_generators.items():
                    gen_id = id(generator)
                    if gen_id not in frozen_generators:
                        for param in generator.parameters():
                            param.requires_grad = False
                        frozen_generators.add(gen_id)
                logger.info(f"  → Frozen {len(frozen_generators)} unique generator instances")
        else:
            logger.info(f"  - No generators available")

        logger.info(f"{'='*60}\n")

        return loaded_count > 0

    def save_adapter_checkpoint(self, round_num=None):
        """
        Save adapter checkpoint(s) for the global model with comprehensive metadata.
        Saves clip_adapter, t5_adapter, clip_projection, and t5_projection from the global Audio2Image model.

        Args:
            round_num: Optional round number to include in checkpoint filename

        Returns:
            list: List of saved checkpoint paths
        """
        import os
        import datetime

        if not self.adapter_save_checkpoint:
            return []

        if self.global_model is None:
            logger.warning("Global model not initialized, cannot save adapter checkpoint")
            return []

        # Create checkpoint directory if it doesn't exist
        os.makedirs(self.adapter_checkpoint_dir, exist_ok=True)

        saved_paths = []

        # Get the global audio2image model
        global_audio2image = self.global_model.get_audio2image_model() if hasattr(self.global_model, 'get_audio2image_model') else None

        if global_audio2image is None:
            logger.warning("Cannot get global audio2image model, cannot save adapter checkpoint")
            return []

        if self.adapter_checkpoint_per_type:
            # Save each adapter type separately
            adapter_types = []

            if hasattr(global_audio2image, 'clip_adapter') and global_audio2image.clip_adapter is not None:
                adapter_types.append(('clip_adapter', global_audio2image.clip_adapter))

            if hasattr(global_audio2image, 't5_adapter') and global_audio2image.t5_adapter is not None:
                adapter_types.append(('t5_adapter', global_audio2image.t5_adapter))

            if hasattr(global_audio2image, 'clip_projection') and global_audio2image.clip_projection is not None:
                adapter_types.append(('clip_projection', global_audio2image.clip_projection))

            if hasattr(global_audio2image, 't5_projection') and global_audio2image.t5_projection is not None:
                adapter_types.append(('t5_projection', global_audio2image.t5_projection))

            for adapter_name, adapter in adapter_types:
                if round_num is not None:
                    checkpoint_path = os.path.join(
                        self.adapter_checkpoint_dir,
                        f'{self.adapter_checkpoint_base_name}_server_{adapter_name}_round_{round_num}.pt'
                    )
                else:
                    checkpoint_path = os.path.join(
                        self.adapter_checkpoint_dir,
                        f'{self.adapter_checkpoint_base_name}_server_{adapter_name}.pt'
                    )

                saved_paths.append(self._save_single_adapter_checkpoint(
                    checkpoint_path, round_num, adapter_name, adapter
                ))
        else:
            # Save all adapters in a single checkpoint
            if round_num is not None:
                checkpoint_path = os.path.join(
                    self.adapter_checkpoint_dir,
                    f'{self.adapter_checkpoint_base_name}_server_round_{round_num}.pt'
                )
            else:
                checkpoint_path = os.path.join(
                    self.adapter_checkpoint_dir,
                    f'{self.adapter_checkpoint_base_name}_server.pt'
                )

            saved_paths.append(self._save_single_adapter_checkpoint(
                checkpoint_path, round_num, 'all', global_audio2image
            ))

        return saved_paths

    def _save_single_adapter_checkpoint(self, checkpoint_path, round_num, adapter_name, adapter_or_model):
        """
        Save a single adapter checkpoint with metadata.

        Args:
            checkpoint_path: Path where to save the checkpoint
            round_num: Training round number
            adapter_name: Name of the adapter ('clip_adapter', 't5_adapter', 'clip_projection', 't5_projection', or 'all')
            adapter_or_model: Adapter instance or global audio2image model

        Returns:
            str: Path where checkpoint was saved
        """
        import datetime

        # Prepare comprehensive checkpoint metadata
        checkpoint = {
            # Server identification
            'server': True,
            'round': round_num if round_num is not None else 0,
            'timestamp': datetime.datetime.now().isoformat(),
            'adapter_name': adapter_name,
        }

        # Save adapter state(s)
        if adapter_name == 'all':
            # Save all adapters in one checkpoint from the global model
            if hasattr(adapter_or_model, 'clip_adapter') and adapter_or_model.clip_adapter is not None:
                checkpoint['clip_adapter_state_dict'] = adapter_or_model.clip_adapter.state_dict()

            if hasattr(adapter_or_model, 't5_adapter') and adapter_or_model.t5_adapter is not None:
                checkpoint['t5_adapter_state_dict'] = adapter_or_model.t5_adapter.state_dict()

            if hasattr(adapter_or_model, 'clip_projection') and adapter_or_model.clip_projection is not None:
                checkpoint['clip_projection_state_dict'] = adapter_or_model.clip_projection.state_dict()

            if hasattr(adapter_or_model, 't5_projection') and adapter_or_model.t5_projection is not None:
                checkpoint['t5_projection_state_dict'] = adapter_or_model.t5_projection.state_dict()
        else:
            # Save specific adapter
            if adapter_or_model is not None:
                checkpoint['adapter_state_dict'] = adapter_or_model.state_dict()

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Create concise log message
        if adapter_name == 'all':
            print(f"[Server] Saved all adapters to {os.path.basename(checkpoint_path)}")
        else:
            print(f"[Server] Saved {adapter_name} to {os.path.basename(checkpoint_path)}")

        return checkpoint_path

    def load_adapter_checkpoint(self, checkpoint_path=None, strict_validation=True, warn_only=False):
        """
        Load adapter checkpoint(s) for the global model with metadata validation.

        Args:
            checkpoint_path: Optional path to checkpoint file (for single file mode)
            strict_validation: If True, reject checkpoint on metadata mismatch
            warn_only: If True, only print warnings without rejecting checkpoint

        Returns:
            bool: True if all checkpoints loaded successfully, False otherwise
        """
        import os
        import glob

        if not self.adapter_load_checkpoint:
            return False

        if self.global_model is None:
            logger.warning("Global model not initialized, cannot load adapter checkpoint")
            return False

        success = True

        if self.adapter_checkpoint_per_type:
            # Load each adapter type separately
            adapter_types = ['clip_adapter', 't5_adapter', 'clip_projection', 't5_projection']

            # First, discover all available adapter sets (by node)
            # Pattern: {base_name}_node{X}_{adapter_type}*.pt
            node_adapter_sets = {}  # {node_id: {adapter_type: [checkpoint_files]}}

            for adapter_name in adapter_types:
                # Search for any node's checkpoint (not just server)
                pattern = os.path.join(
                    self.adapter_checkpoint_dir,
                    f'{self.adapter_checkpoint_base_name}_node*_{adapter_name}*.pt'
                )

                checkpoint_files = glob.glob(pattern)

                # Group by node_id
                for checkpoint_file in checkpoint_files:
                    # Extract node_id from filename: base_name_node{X}_{adapter_name}_...
                    filename = os.path.basename(checkpoint_file)
                    parts = filename.split('_')

                    # Find the part that starts with 'node'
                    node_id = None
                    for part in parts:
                        if part.startswith('node'):
                            node_id = part.replace('node', '')
                            break

                    if node_id is None:
                        continue

                    if node_id not in node_adapter_sets:
                        node_adapter_sets[node_id] = {}

                    if adapter_name not in node_adapter_sets[node_id]:
                        node_adapter_sets[node_id][adapter_name] = []

                    node_adapter_sets[node_id][adapter_name].append(checkpoint_file)

            # Check if we have complete sets (all 4 adapter types)
            complete_sets = {}
            for node_id, adapters in node_adapter_sets.items():
                if len(adapters) == len(adapter_types):
                    # All adapter types present for this node
                    complete_sets[node_id] = adapters

            if not complete_sets:
                print(f"[Server] ⚠ No complete adapter checkpoint set found")
                print(f"[Server] ⚠ Available partial sets: {list(node_adapter_sets.keys())}")
                return False

            # Verify we have exactly one complete set
            if len(complete_sets) > 1:
                print(f"[Server] ⚠ WARNING: Found {len(complete_sets)} complete adapter sets from nodes: {list(complete_sets.keys())}")
                print(f"[Server] ⚠ Will use round-robin distribution to clients")
                # Store all sets for round-robin distribution
                self._multiple_adapter_sets = complete_sets
            else:
                print(f"[Server] ✓ Found exactly one complete adapter set from node {list(complete_sets.keys())[0]}")
                self._multiple_adapter_sets = None

            # Load the first complete set into global model (if using single set)
            if len(complete_sets) == 1:
                node_id = list(complete_sets.keys())[0]
                for adapter_name in adapter_types:
                    # Use the most recent checkpoint for this adapter type
                    checkpoint_files = sorted(complete_sets[node_id][adapter_name])
                    checkpoint_file = checkpoint_files[-1]

                    print(f"[Server] Loading {adapter_name} from node {node_id}: {os.path.basename(checkpoint_file)}")

                    loaded = self._load_single_adapter_checkpoint(
                        checkpoint_file, strict_validation, warn_only, adapter_name, self.global_model
                    )

                    if not loaded:
                        success = False
            else:
                # Multiple sets: will be distributed round-robin in set_clients
                print(f"[Server] Skipping global model adapter loading (will distribute to clients in round-robin)")
                success = True
        else:
            # Load all adapters from a single checkpoint
            if checkpoint_path is None:
                # Auto-detect checkpoint file (search for node checkpoints)
                pattern = os.path.join(
                    self.adapter_checkpoint_dir,
                    f'{self.adapter_checkpoint_base_name}_node*.pt'
                )

                checkpoint_files = sorted(glob.glob(pattern))

                if not checkpoint_files:
                    print(f"[Server] ⚠ No adapter checkpoint found")
                    return False

                if len(checkpoint_files) > 1:
                    print(f"[Server] ⚠ WARNING: Found {len(checkpoint_files)} adapter checkpoint files")
                    print(f"[Server] ⚠ Using the most recent: {os.path.basename(checkpoint_files[-1])}")

                checkpoint_path = checkpoint_files[-1]

            success = self._load_single_adapter_checkpoint(
                checkpoint_path, strict_validation, warn_only, 'all', self.global_model
            )

        return success

    def _load_single_adapter_checkpoint(self, checkpoint_path, strict_validation=True, warn_only=False, adapter_name='all', global_audio2image=None):
        """
        Load a single adapter checkpoint with validation.

        Args:
            checkpoint_path: Path to checkpoint file
            strict_validation: If True, reject checkpoint on metadata mismatch
            warn_only: If True, only print warnings without rejecting checkpoint
            adapter_name: Name of adapter to load ('clip_adapter', 't5_adapter', etc., or 'all')
            global_audio2image: Global audio2image model instance

        Returns:
            bool: True if loaded successfully, False otherwise
        """
        import os

        if not os.path.exists(checkpoint_path):
            print(f"[Server] ⚠ Checkpoint not found: {checkpoint_path}")
            return False

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Load adapter state(s)
            if adapter_name == 'all':
                # Load all adapters from checkpoint
                if 'clip_adapter_state_dict' in checkpoint and hasattr(global_audio2image, 'clip_adapter') and global_audio2image.clip_adapter is not None:
                    try:
                        global_audio2image.clip_adapter.load_state_dict(checkpoint['clip_adapter_state_dict'])
                        print(f"  ✓ Loaded clip_adapter from round {checkpoint.get('round', 'N/A')}")
                    except Exception as e:
                        print(f"  ⚠ Warning: Could not load clip_adapter: {e}")

                if 't5_adapter_state_dict' in checkpoint and hasattr(global_audio2image, 't5_adapter') and global_audio2image.t5_adapter is not None:
                    try:
                        global_audio2image.t5_adapter.load_state_dict(checkpoint['t5_adapter_state_dict'])
                        print(f"  ✓ Loaded t5_adapter from round {checkpoint.get('round', 'N/A')}")
                    except Exception as e:
                        print(f"  ⚠ Warning: Could not load t5_adapter: {e}")

                if 'clip_projection_state_dict' in checkpoint and hasattr(global_audio2image, 'clip_projection') and global_audio2image.clip_projection is not None:
                    try:
                        global_audio2image.clip_projection.load_state_dict(checkpoint['clip_projection_state_dict'])
                        print(f"  ✓ Loaded clip_projection from round {checkpoint.get('round', 'N/A')}")
                    except Exception as e:
                        print(f"  ⚠ Warning: Could not load clip_projection: {e}")

                if 't5_projection_state_dict' in checkpoint and hasattr(global_audio2image, 't5_projection') and global_audio2image.t5_projection is not None:
                    try:
                        global_audio2image.t5_projection.load_state_dict(checkpoint['t5_projection_state_dict'])
                        print(f"  ✓ Loaded t5_projection from round {checkpoint.get('round', 'N/A')}")
                    except Exception as e:
                        print(f"  ⚠ Warning: Could not load t5_projection: {e}")
            else:
                # Load specific adapter
                if 'adapter_state_dict' in checkpoint:
                    adapter = None
                    model_adapters_model = global_audio2image.adapters
                    # get the appropriate adapter from the global model
                    if adapter_name == 'clip_adapter' and hasattr(model_adapters_model['clip'], 'adapter_clip'):
                        adapter = model_adapters_model['clip'].adapter_clip
                    elif adapter_name == 't5_adapter' and hasattr(model_adapters_model['t5'], 'adapter_t5'):
                        adapter = model_adapters_model['t5'].adapter_t5
                    elif adapter_name == 'clip_projection' and hasattr(model_adapters_model['clip'], 'projection_clip'):
                        adapter = model_adapters_model['clip'].projection_clip
                    elif adapter_name == 't5_projection' and hasattr(model_adapters_model['t5'], 'projection_t5'):
                        adapter = model_adapters_model['t5'].projection_t5

                    if adapter is not None:
                        try:
                            adapter.load_state_dict(checkpoint['adapter_state_dict'])
                            print(f"[Server] ✓ Loaded {adapter_name} from round {checkpoint.get('round', 'N/A')}")
                        except Exception as e:
                            print(f"[Server] ⚠ Warning: Could not load {adapter_name}: {e}")
                            return False
                    else:
                        print(f"[Server] ⚠ Warning: Adapter {adapter_name} not found in global model")
                        return False

            return True

        except Exception as e:
            logger.error(f"Error loading adapter checkpoint: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_single_adapter_checkpoint_to_client(self, checkpoint_path, client, adapter_name):
        """
        Load a specific adapter checkpoint into a client's model.

        Args:
            checkpoint_path: Path to checkpoint file
            client: Client object to load adapter into
            adapter_name: Name of adapter to load ('clip_adapter', 't5_adapter', 'clip_projection', 't5_projection')

        Returns:
            bool: True if loaded successfully, False otherwise
        """
        import os

        if not os.path.exists(checkpoint_path):
            print(f"[Server] ⚠ Checkpoint not found: {checkpoint_path}")
            return False

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            if 'adapter_state_dict' not in checkpoint:
                print(f"[Server] ⚠ No adapter_state_dict in checkpoint: {checkpoint_path}")
                return False

            # Get the appropriate adapter from the client's model
            adapter = None

            if adapter_name == 'clip_adapter' and 'clip' in client.adapters:
                clip_seq = client.adapters['clip']
                adapter = clip_seq.adapter_clip if hasattr(clip_seq, 'adapter_clip') else None
            elif adapter_name == 'clip_projection' and 'clip' in client.adapters:
                clip_seq = client.adapters['clip']
                adapter = clip_seq.projection_clip if hasattr(clip_seq, 'projection_clip') else None
            elif adapter_name == 't5_adapter' and client.diffusion_type == 'flux' and 't5' in client.adapters:
                t5_seq = client.adapters['t5']
                adapter = t5_seq.adapter_t5 if hasattr(t5_seq, 'adapter_t5') else None
            elif adapter_name == 't5_projection' and client.diffusion_type == 'flux' and 't5' in client.adapters:
                t5_seq = client.adapters['t5']
                adapter = t5_seq.projection_t5 if hasattr(t5_seq, 'projection_t5') else None

            if adapter is not None:
                try:
                    adapter.load_state_dict(checkpoint['adapter_state_dict'])
                    print(f"[Server]   ✓ Client {client.id}: Loaded {adapter_name} from round {checkpoint.get('round', 'N/A')}")
                    return True
                except Exception as e:
                    print(f"[Server]   ⚠ Client {client.id}: Could not load {adapter_name}: {e}")
                    return False
            else:
                print(f"[Server]   ⚠ Client {client.id}: Adapter {adapter_name} not found in client model")
                return False

        except Exception as e:
            print(f"[Server] Error loading adapter checkpoint for client {client.id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def validate_generator(self, test_class_prompts):
        """
        Validate generator quality by comparing generated vs real prompts.

        Args:
            test_class_prompts: dict {class_name: [{'clip': tensor, 't5': tensor, 'audio_embeddings': tensor}, ...]}

        Returns:
            dict: Validation metrics including cosine similarity and reconstruction error
        """
        if not self.use_generator or self.prompt_generator is None:
            logger.warning("Generator not initialized, cannot validate")
            return {}

        self.prompt_generator.eval()

        metrics = {
            'cosine_similarity': [],
            'mse_loss': [],
            'l1_loss': [],
            'per_class_similarity': {}
        }

        with torch.no_grad():
            for class_name, prompts_list in test_class_prompts.items():
                if not prompts_list:
                    continue

                class_similarities = []
                class_mse = []
                class_l1 = []

                for prompt_dict in prompts_list:
                    # Extract embeddings
                    audio_emb = prompt_dict.get('audio_embeddings')
                    if audio_emb is None:
                        continue

                    # Move to device and ensure correct shape
                    audio_emb = audio_emb.to(self.device)
                    if audio_emb.dim() == 2:
                        audio_emb = audio_emb.unsqueeze(0)  # Add batch dimension

                    if self.generator_type == 'vae':
                        # Get visual condition
                        clip_emb = prompt_dict.get('clip')
                        if clip_emb is not None:
                            clip_emb = clip_emb.to(self.device)
                            if clip_emb.dim() == 2:
                                clip_emb = clip_emb.unsqueeze(0)

                            # Generate prompt
                            generated, mu, logvar = self.prompt_generator(audio_emb, visual_condition=clip_emb)
                        else:
                            # Generate without visual condition
                            generated, mu, logvar = self.prompt_generator(audio_emb)

                        # Compute metrics
                        cos_sim = F.cosine_similarity(generated, audio_emb, dim=-1).mean().item()
                        mse = F.mse_loss(generated, audio_emb).item()
                        l1 = F.l1_loss(generated, audio_emb).item()

                        class_similarities.append(cos_sim)
                        class_mse.append(mse)
                        class_l1.append(l1)

                    elif self.generator_type == 'gan':
                        # For GAN, generate from random noise and compare distribution
                        batch_size = audio_emb.size(0)
                        z = torch.randn(batch_size, 256).to(self.device)

                        if self.prompt_generator_clip is not None:
                            generated_clip = self.prompt_generator_clip(z)
                            clip_emb = prompt_dict.get('clip')
                            if clip_emb is not None:
                                clip_emb = clip_emb.to(self.device)
                                if clip_emb.dim() == 2:
                                    clip_emb = clip_emb.unsqueeze(0)

                                cos_sim = F.cosine_similarity(generated_clip, clip_emb, dim=-1).mean().item()
                                class_similarities.append(cos_sim)

                # Store per-class metrics
                if class_similarities:
                    metrics['per_class_similarity'][class_name] = np.mean(class_similarities)
                    metrics['cosine_similarity'].extend(class_similarities)
                    metrics['mse_loss'].extend(class_mse)
                    metrics['l1_loss'].extend(class_l1)

        # Compute overall metrics
        if metrics['cosine_similarity']:
            avg_similarity = np.mean(metrics['cosine_similarity'])
            avg_mse = np.mean(metrics['mse_loss'])
            avg_l1 = np.mean(metrics['l1_loss'])

            logger.info(f"\n=== Generator Validation Metrics ===")
            logger.info(f"Average Cosine Similarity: {avg_similarity:.4f}")
            logger.info(f"Average MSE Loss: {avg_mse:.6f}")
            logger.info(f"Average L1 Loss: {avg_l1:.6f}")
            logger.info(f"Classes validated: {len(metrics['per_class_similarity'])}")

            # Log per-class similarities
            for class_name, sim in sorted(metrics['per_class_similarity'].items()):
                logger.info(f"  - {class_name}: {sim:.4f}")

            return {
                'avg_cosine_similarity': avg_similarity,
                'avg_mse_loss': avg_mse,
                'avg_l1_loss': avg_l1,
                'per_class_similarity': metrics['per_class_similarity'],
                'num_classes': len(metrics['per_class_similarity'])
            }
        else:
            logger.warning("No valid prompts for generator validation")
            return {}

    def aggregate_synthetic_samples(self):
        """
        Aggregate synthetic samples from all clients.
        Store aggregated samples for use in global training or adapter fine-tuning.
        """
        if not hasattr(self, 'client_synthetic_samples') or len(self.client_synthetic_samples) == 0:
            print("[Server] No synthetic samples to aggregate")
            return

        print(f"\n[Server] Aggregating synthetic samples from {len(self.client_synthetic_samples)} clients")

        # Structure: {class_name: {client_id: tensor}}
        aggregated_by_class = defaultdict(dict)

        for client_id, synthetic_samples in self.client_synthetic_samples.items():
            for class_name, samples in synthetic_samples.items():
                aggregated_by_class[class_name][client_id] = samples

        # Memory tracking - before clearing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated(self.device) / (1024**2)
            logger.debug(f"[MemTrack] Server - Before aggregate synthetic samples clear: {mem_before:.2f} MB")

        # Clear previous aggregated samples to prevent memory leaks
        if hasattr(self, 'aggregated_synthetic_samples'):
            del self.aggregated_synthetic_samples
        torch.cuda.empty_cache()

        # Memory tracking - after clearing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_after_clear = torch.cuda.memory_allocated(self.device) / (1024**2)
            mem_freed = mem_before - mem_after_clear
            logger.debug(f"[MemTrack] Server - After clear synthetic samples: {mem_after_clear:.2f} MB (freed {mem_freed:.2f} MB)")

        # Store aggregated samples
        self.aggregated_synthetic_samples = {}

        total_samples = 0
        for class_name, client_samples in aggregated_by_class.items():
            # Collect all samples for this class
            all_samples = []
            for client_id, samples in client_samples.items():
                all_samples.append(samples)
                total_samples += samples.size(0) if hasattr(samples, 'size') else len(samples)

            # Stack or concatenate samples
            if len(all_samples) > 0:
                self.aggregated_synthetic_samples[class_name] = torch.cat(all_samples, dim=0)

        # Memory tracking - after aggregation
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_after_agg = torch.cuda.memory_allocated(self.device) / (1024**2)
            mem_added = mem_after_agg - mem_after_clear
            logger.debug(f"[MemTrack] Server - After aggregating {total_samples} samples: {mem_after_agg:.2f} MB (added {mem_added:.2f} MB)")

        print(f"[Server] Aggregated {total_samples} synthetic samples across {len(self.aggregated_synthetic_samples)} classes")

        # Log summary
        for class_name, samples in self.aggregated_synthetic_samples.items():
            num_samples = samples.size(0) if hasattr(samples, 'size') else len(samples)
            print(f"  - {class_name}: {num_samples} samples")

        # Clear client synthetic samples to free memory
        self.client_synthetic_samples = {}

        # Log to wandb if enabled
        if not self.no_wandb:
            self.data_log({
                "server/synthetic_samples_total": total_samples,
                "server/synthetic_samples_classes": len(self.aggregated_synthetic_samples),
                "round": self.round
            })

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        self.gathered_nodes = 0

        if self.adapter_aggregation_mode != 'none':
            received_model = self.receive_nodes_adapters()

        if self.aggregation_method == 'per_class_average':
            received_model = self.receive_models_per_class_average()
        
        self.gathered_nodes = received_model
        return received_model
    
    def receive_nodes_adapters(self):

        active_clients = self.selected_clients

        # Memory tracking - before clearing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated(self.device) / (1024**2)
            logger.debug(f"[MemTrack] Server - Before receive_nodes_adapters clear: {mem_before:.2f} MB")

        # Clear previous round's adapter data to prevent memory leaks
        if hasattr(self, 'nodes_adapters'):
            del self.nodes_adapters
        if hasattr(self, 'nodes_adapters_modules'):
            del self.nodes_adapters_modules
        torch.cuda.empty_cache()

        # Memory tracking - after clearing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_after_clear = torch.cuda.memory_allocated(self.device) / (1024**2)
            mem_freed = mem_before - mem_after_clear
            logger.debug(f"[MemTrack] Server - After clear: {mem_after_clear:.2f} MB (freed {mem_freed:.2f} MB)")

        self.nodes_adapters = {}
        self.nodes_adapters_modules = {}
        gathered_nodes = 0

        for node in active_clients:
            self.nodes_adapters[node.id] = node.adapters
            self.nodes_adapters_modules[node.id] = node.adapters_modules

            gathered_nodes += 1

        # Memory tracking - after receiving
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_after_receive = torch.cuda.memory_allocated(self.device) / (1024**2)
            mem_added = mem_after_receive - mem_after_clear
            logger.debug(f"[MemTrack] Server - After receiving {gathered_nodes} adapters: {mem_after_receive:.2f} MB (added {mem_added:.2f} MB)")

        return gathered_nodes

    def receive_models_per_class_average(self):

        active_clients = self.selected_clients

        # Memory tracking - before clearing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated(self.device) / (1024**2)
            logger.debug(f"[MemTrack] Server - Before receive_models_per_class_average clear: {mem_before:.2f} MB")

        # Clear previous round's data explicitly to prevent memory leaks
        if hasattr(self, 'nodes_per_class_outputs'):
            del self.nodes_per_class_outputs
        if hasattr(self, 'nodes_per_class_outputs_means'):
            del self.nodes_per_class_outputs_means
        if hasattr(self, 'nodes_per_class_adapters_outputs'):
            del self.nodes_per_class_adapters_outputs
        if hasattr(self, 'nodes_per_class_adapters_outputs_means'):
            del self.nodes_per_class_adapters_outputs_means
        torch.cuda.empty_cache()

        # Memory tracking - after clearing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            mem_after_clear = torch.cuda.memory_allocated(self.device) / (1024**2)
            mem_freed = mem_before - mem_after_clear
            logger.debug(f"[MemTrack] Server - After clear per_class data: {mem_after_clear:.2f} MB (freed {mem_freed:.2f} MB)")

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

            node.send_time_cost['num_rounds'] += 1
            node.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
            node._move_to_cpu()

            if self.model_aggregation == "per_class_average":
                self.send_models_per_class_average(node)

            if self.adapter_aggregation_mode == 'avg':
                self.send_adapters(node)
            elif self.adapter_aggregation_mode == 'kd':
                self.send_adapters(node)
        return
    
    def send_adapters(self, node):
        logger.info(f"Sending global adapters to node {node.id}")
        self.global_model = self.global_model.to(self.device)
        node.update_local_adapters(self.global_adapters)

        self.global_model = self.global_model.to("cpu")

    def send_models_per_class_average(self,node):
        return
        self.global_model = self.global_model.to(self.device)
        for client in self.clients:
            start_time = time.time()
            client.update_local_adapters(self.global_adapters)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
            client._move_to_cpu()

        self.global_model = self.global_model.to("cpu")

    def aggregate_parameters(self):
        """Aggregate Audio2Visual model parameters from clients."""
        assert (self.gathered_nodes > 0)

        if self.aggregation_method == 'per_class_average':
            self.aggregate_audio_encoder_parameters_per_class_avarage()

        if self.adapter_aggregation_mode == 'avg':
            self.aggregate_adapters_parameters_fedavg()
        elif self.adapter_aggregation_mode == 'kd':
            self.aggregate_adapters_knowledge_distillation()
        elif self.adapter_aggregation_mode != 'none':
            print(f"Adapter aggregation mode {self.adapter_aggregation_mode} not recognized.")
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

        # Compute global mean from per-class means if configured
        if self.compute_global_mean_from_class_means:
            self._compute_global_mean_from_class_means(global_per_class_output_means)

    def _compute_global_mean_from_class_means(self, global_per_class_output_means):
        """
        Compute a unified global mean by averaging all per-class means.

        This method takes the per-class output means and computes a single global mean
        by stacking all class means and averaging them across the class dimension.

        Args:
            global_per_class_output_means (dict): Dictionary mapping class names to their mean tensors

        Sets:
            self.global_output_means: The computed global mean tensor
        """
        if not global_per_class_output_means:
            print("[Server] Warning: No per-class means available to compute global mean")
            self.global_output_means = None
            return

        global_means = []
        for class_name, per_class_output_mean in global_per_class_output_means.items():
            global_means.append(per_class_output_mean)

        if not global_means:
            print("[Server] Warning: No valid means to stack for global mean computation")
            self.global_output_means = None
            return

        global_means = torch.stack(global_means)
        global_mean = torch.mean(global_means, dim=0)

        self.global_output_means = global_mean

        print(f"[Server] Computed global mean from {len(global_per_class_output_means)} class means")
        print(f"[Server] Global mean shape: {global_mean.shape}")

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
        logger.info(f"Aggregated adapters from nodes {list(nodes_adapters_states.keys())} using FedAvg.")

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

    def aggregate_adapters_knowledge_distillation(self):
        """
        Aggregate adapters using Knowledge Distillation approach:
        1. Freeze node adapters (eval mode)
        2. Create ONE single adapter on server
        3. Train server adapter on ALL classes:
           - For each class: generate synthetic samples with frozen generators
           - Compute MSE loss between server adapter outputs and frozen node adapters outputs
           - Train with accumulated loss from all classes
        4. Load trained adapter into global_adapters_modules
        5. Send trained adapter to all client nodes
        """
        logger.info("[KD Aggregation] Starting Knowledge Distillation adapter aggregation")

        # Check prerequisites
        if not hasattr(self, 'prompt_generators') or not self.prompt_generators:
            logger.error("[KD Aggregation] No generators available, cannot perform KD aggregation")
            return

        if not self.nodes_adapters_modules:
            logger.error("[KD Aggregation] No adapter modules from nodes, cannot perform KD aggregation")
            return

        # Get configuration parameters
        kd_epochs = getattr(self.config.feda2v, 'kd_training_epochs', 10)
        kd_batch_size = getattr(self.config.training, 'batch_size', 8)
        kd_lr = getattr(self.config.feda2v, 'kd_learning_rate', 0.001)
        kd_samples_per_class = getattr(self.config.feda2v, 'kd_samples_per_class', 100)

        logger.info(f"[KD Aggregation] Config: epochs={kd_epochs}, batch_size={kd_batch_size}, lr={kd_lr}, samples_per_class={kd_samples_per_class}")

        # Step 1: Freeze node adapters and collect target generator function
        logger.info("[KD Aggregation] Step 1: Freezing node adapters")
        self._move_to_gpu(self.device, models=['global_adapters'])
        self._freeze_nodes_adapters()

        # Step 2: Collect classes present in nodes
        classes_to_train = self._get_classes_from_nodes()
        if not classes_to_train:
            logger.error("[KD Aggregation] No classes found in nodes")
            return
        logger.info(f"[KD Aggregation] Classes to train on: {classes_to_train}")

        # Step 3: Create single server adapter
        # logger.info("[KD Aggregation] Step 2: Creating single server adapter")
        # server_adapter = self._create_single_server_adapter()

        # Step 4: Train server adapter with KD on all classes
        logger.info("[KD Aggregation] Step 3: Training server adapter with Knowledge Distillation")
        self._train_single_adapter_kd(
            server_adapter=self.global_adapters,
            classes_to_train=classes_to_train,
            epochs=kd_epochs,
            batch_size=kd_batch_size,
            learning_rate=kd_lr,
            samples_per_class=kd_samples_per_class
        )

        # Step 5: Load trained adapter into global_adapters_modules
        # logger.info("[KD Aggregation] Step 4: Loading trained adapter into global_adapters_modules")
        # self.global_adapters_modules.load_state_dict(server_adapter.state_dict())

        # Step 6: Generate and distribute synthetic samples to nodes (for image generation)
        logger.info("[KD Aggregation] Step 5: Generating synthetic samples for nodes")
        if hasattr(self, 'generate_synthetic_samples_for_all_nodes') and hasattr(self, 'distribute_synthetic_samples_to_nodes'):
            try:
                self.generate_synthetic_samples_for_all_nodes()
                self.distribute_synthetic_samples_to_nodes()
                logger.info("[KD Aggregation] Synthetic samples generated and distributed to nodes")
            except Exception as e:
                logger.error(f"[KD Aggregation] Failed to generate/distribute synthetic samples: {e}")
        else:
            logger.warning("[KD Aggregation] Synthetic sample generation/distribution methods not available")

        # Step 7: Cleanup
        logger.info("[KD Aggregation] Step 6: Cleaning up intermediate data")
        # del server_adapter
        # torch.cuda.empty_cache()
        self._move_to_cpu(models=['global_adapters'])
        self._unfreeze_nodes_adapters()

        logger.info("[KD Aggregation] Knowledge Distillation aggregation completed successfully")

    def _freeze_nodes_adapters(self):
        """
        Set all node adapters to eval mode and freeze their parameters.
        """
        for node in self.clients:
            node_id = node.id
            self._freeze_node_adapter(node_id)
        logger.info(f"[KD] Frozen {len(self.nodes_adapters_modules)} node adapters")
    
    def _freeze_node_adapter(self, node_id):
        """
        Set a specific node's adapter to eval mode and freeze its parameters.

        Args:
            node_id: ID of the node whose adapter to freeze
        """
        node_adapter = self.nodes_adapters_modules.get(node_id)
        if node_adapter is not None:
            node_adapter.eval()
            for param in node_adapter.parameters():
                param.requires_grad = False
            logger.info(f"[KD] Frozen adapter for node {node_id}")
        else:
            logger.warning(f"[KD] No adapter found for node {node_id} to freeze")

    def _unfreeze_nodes_adapters(self):
        """
        Unfreeze all node adapters' parameters.
        """
        for node in self.clients:
            node_id = node.id
            self._unfreeze_node_adapter(node_id)
        logger.info(f"[KD] Unfrozen {len(self.nodes_adapters_modules)} node adapters")

    def _unfreeze_node_adapter(self, node_id):
        """
        Unfreeze a specific node's adapter parameters.

        Args:
            node_id: ID of the node whose adapter to unfreeze
        """
        node_adapter = self.nodes_adapters_modules.get(node_id)
        if node_adapter is not None:
            node_adapter.train()
            for param in node_adapter.parameters():
                param.requires_grad = True
            logger.info(f"[KD] Unfrozen adapter for node {node_id}")
        else:
            logger.warning(f"[KD] No adapter found for node {node_id} to unfreeze")

    def _get_classes_from_nodes(self):
        """
        Collect all unique classes present in the nodes.

        Returns:
            list: List of unique class names
        """
        all_classes = set()
        for node in self.clients:
            if hasattr(node.node_data.dataset, 'active_classes') and node.node_data.dataset.active_classes:
                all_classes.update(node.node_data.dataset.active_classes)
          
        # Filter classes that have generators
        classes_with_generators = [c for c in all_classes if c in self.prompt_generators]

        logger.info(f"[KD] Found {len(all_classes)} classes in nodes, {len(classes_with_generators)} have generators")
        return classes_with_generators

    def _create_single_server_adapter(self):
        """
        Create a single adapter module on server.

        Returns:
            DownstreamSinestesiaAdapters: Single adapter instance
        """
        from system.flcore.trainmodel.downstreamsinestesiaadapters import DownstreamSinestesiaAdapters

        device = self.device if hasattr(self, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'

        adapter = DownstreamSinestesiaAdapters(
            diffusion_type=self.global_model.diffusion_type,
            init_ast_model=False  # Don't need AST model, only adapters
        )
        adapter.setup_adapters()
        adapter.to(device)

        logger.info(f"[KD] Created single server adapter on device {device}")
        return adapter

    def _train_single_adapter_kd(self, server_adapter, classes_to_train,
                                  epochs, batch_size, learning_rate, samples_per_class):
        """
        Train single server adapter using Knowledge Distillation with MSE loss.
        The adapter is trained on synthetic samples from ALL classes, with loss
        accumulated across all classes.

        Args:
            server_adapter: Single adapter module to train
            classes_to_train: List of class names to train on
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            samples_per_class: Number of synthetic samples to generate per class
        """
        device = self.device if hasattr(self, 'device') else 'cuda' if torch.cuda.is_available() else 'cpu'
        mse_loss = nn.MSELoss()


        optimizer = self.global_optimizers
        # Create single optimizer for the server adapter
        # optimizer = torch.optim.AdamW(
        #     server_adapter.parameters(),
        #     lr=learning_rate,
        #     weight_decay=getattr(self.config.feda2v, 'adapters_weight_decay', 0.0001)
        # )

        # Collect which nodes have which classes
        node_classes = {}
        for node in self.clients:
            if hasattr(node.node_data.dataset, 'selected_classes') and node.node_data.dataset.selected_classes:
                node_classes[node.id] = node.node_data.dataset.selected_classes

        logger.info(f"[KD Training] Node classes mapping: {node_classes}")
        logger.info(f"[KD Training] Training single adapter on {len(classes_to_train)} classes")

        # Training loop
        for epoch in range(epochs):
            epoch_total_loss = 0.0
            epoch_class_losses = {}
            total_batches = 0

            # Iterate over all classes
            for node_id, node_active_classes in node_classes.items():
                logger.info(f"[KD Training] Node {node_id} has classes: {node_classes}")
                for class_name in node_active_classes:
                    if class_name not in self.prompt_generators:
                        logger.warning(f"[KD Training] No generator for class '{class_name}', skipping")
                        continue

                    generator = self.prompt_generators[class_name]
                    generator.eval()  # Keep generator frozen

                    # Generate synthetic samples for this class
                    with torch.no_grad():
                        try:
                            if self.synthetic_samples_per_class is not None:
                                synthetic_samples = self.synthetic_samples_per_node[node_id][class_name].to(self.device)
                                # if synthetic_samples is None or synthetic_samples.size(0) < samples_per_class:
                                #     logger.warning(f"[KD Training] Not enough pre-generated samples for class '{class_name}', generating new samples")
                                #     synthetic_samples = generator.sample(num_samples=self.synthetic_samples_per_class, device=device)
                                
                                # synthetic_samples = synthetic_samples[:samples_per_class].to(device)
                        except Exception as e:
                            logger.error(f"[KD Training] Failed to generate samples for class '{class_name}': {e}")
                            continue

                    # Get target outputs from nodes that have this class (frozen adapters)
                    class_targets = {}

                    with torch.no_grad():
                        for node_id, node_class in node_classes.items():
                            
                            self.nodes_adapters[node_id] = {k: v.to(self.device) for k, v in self.nodes_adapters[node_id].items()}
                            node_adapter = self.nodes_adapters[node_id]
                            if node_adapter is None:
                                continue
                            for adapter_name in node_adapter.keys():
                                if adapter_name not in class_targets:
                                    class_targets[adapter_name] = []
                            
                            try:
                                for adapter_name, adapter_module in node_adapter.items():
                                    class_targets[adapter_name].append(adapter_module(synthetic_samples.to(device)).detach())

                            except Exception as e:
                                logger.error(f"[KD Training] Failed to get target outputs for node {node_id} adapter {adapter_name}, class '{class_name}': {e}")
                                continue

                    # Average targets across nodes for this class
                    # if not class_clip_targets and not class_t5_targets:
                    #     logger.warning(f"[KD Training] No target outputs for class '{class_name}', skipping")
                    #     continue
                    targets = {}
                    for adapter_name, outputs in class_targets.items():
                        if outputs:
                            targets[adapter_name] = torch.stack(class_targets[adapter_name]).mean(dim=0)
                        else:
                            targets[adapter_name] = None
                    # clip_target = torch.stack(class_targets['clip']).mean(dim=0) if 'clip' in class_targets else None
                    # t5_target = torch.stack(class_targets['t5']).mean(dim=0) if 't5' in class_targets else None

                    # Train on batches for this class
                    num_batches = (samples_per_class + batch_size - 1) // batch_size
                    class_loss = {adapter_name: 0.0 for adapter_name in server_adapter.keys() }

                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, samples_per_class)

                        batch_samples = synthetic_samples[start_idx:end_idx]

                        for optimizer_name, optimizer in self.global_optimizers.items():
                            optimizer.zero_grad()

                        # Forward pass through server adapter
                        batch_loss = { adapter_name: 0.0 for adapter_name in server_adapter.keys() }

                        try:
                            for adapter_name, adapter_module in server_adapter.items():
                                if adapter_name in targets and targets[adapter_name] is not None:
                                    adapter_output = adapter_module(batch_samples)
                                    batch_target = targets[adapter_name][start_idx:end_idx]
                                    batch_loss[adapter_name] += mse_loss(adapter_output, batch_target)
                          
                                    if batch_loss[adapter_name] > 0:
                                        batch_loss[adapter_name].backward()
                                        if adapter_name not in self.global_optimizers:
                                            logger.error(f"[KD Training] No optimizer found for adapter '{adapter_name}'")
                                            continue
                                        self.global_optimizers[adapter_name].step()
                                        
                                        class_loss[adapter_name] += batch_loss[adapter_name].item()
                                        epoch_total_loss += batch_loss[adapter_name].item()
                                        total_batches += 1
                        except Exception as e:
                            logger.error(f"[KD Training] Error during training for class '{class_name}', batch {batch_idx}: {e}")
                            continue

                    # Track per-class loss
                    avg_class_loss = {}
                    avg_class_loss = {adapter_name: class_loss[adapter_name] / num_batches if num_batches > 0 else 0.0 for adapter_name in class_loss}
                    epoch_class_losses[class_name] = avg_class_loss

                self.nodes_adapters[node_id] = {k: v.to('cpu') for k, v in self.nodes_adapters[node_id].items()}

                # Log epoch progress
                avg_epoch_loss = epoch_total_loss / total_batches if total_batches > 0 else 0.0

                if epoch % max(1, epochs // 5) == 0 or epoch == epochs - 1:
                    for class_name, class_losses in epoch_class_losses.items():
                        loss_details = ", ".join([f"{adapter_name}: {loss:.6f}" for adapter_name, loss in class_losses.items()])
                        logger.info(f"[KD Training] Epoch {epoch+1}/{epochs} - Class '{class_name}' Losses: {loss_details}")
                    logger.info(f"[KD Training] Epoch {epoch+1}/{epochs} - Total Loss: {avg_epoch_loss:.6f}")
                    logger.info(f"[KD Training] Epoch {epoch+1}/{epochs} - Per-class Losses: {loss_details}")

                    # Memory tracking
                    if torch.cuda.is_available():
                        mem_allocated = torch.cuda.memory_allocated() / 1024**2
                        logger.debug(f"[KD Training] GPU Memory: {mem_allocated:.2f} MB")

        logger.info(f"[KD Training] Completed training single adapter over {epochs} epochs")

    def add_parameters(self, w, client_model):
        """Add weighted client parameters to global Audio2Visual model."""
        if hasattr(self.global_model, 'audio2image') and hasattr(client_model, 'audio2image'):
            for server_param, client_param in zip(
                    self.global_model.audio2image.parameters(),
                    client_model.audio2image.parameters()):
                if server_param.requires_grad and client_param.requires_grad:
                    server_param.data += client_param.data.clone() * w

    def client_round_starting_hook(self, node):
        # Skip metrics in generator-only training mode
        if self.generator_training_mode:
            return

        if self.eval_gap > 0 and self.round % self.eval_gap == 0:
            return

        print(f"\nStarting training round {self.round} for Node {node.id}.")

        node_metrics = self.node_metrics(node, train_splits=self.nodes_train_metrics_splits, test_splits=[])

        if not self.no_wandb:
            wandb_metrics = {}

            for metric_type, metric_splits in node_metrics.items():
                for split, metric in metric_splits.items():
                    wandb_metrics.update(node.log_metrics( metric, round = self.round, suffix=f"_on_{split}_pre"))

            for wandb_metric in wandb_metrics:
                metrics = {wandb_metric: wandb_metrics[wandb_metric], "round": self.round}
                self.data_log(metrics)

        # Display only text_loss or accuracy if present
        display_metrics = []
        for metric_name in ['text_loss', 'accuracy']:
            for metric_type, metric_splits in node_metrics.items():
                for split, metrics in metric_splits.items():
                    if metric_name in metrics:
                        display_metrics.append(f"{metric_type} on {split} {metric_name} {metrics[metric_name]['mean']:.4f}")

        if display_metrics:
            print(F"Node {node.id} pre-round metrics: " + ", ".join(display_metrics))

    def client_round_ending_hook(self, client):
        # Skip metrics in generator-only training mode (but still allow synthetic sample generation)
        if self.generator_training_mode:
            # Generate synthetic samples if using pretrained generators
            use_pretrained_generators = getattr(self.config.feda2v, 'use_pretrained_generators', False)
            if use_pretrained_generators and hasattr(client, 'prompt_generator') and client.prompt_generator is not None:
                # Only run synthetic sample generation, skip metrics
                if self.round % 10 == 0:  # Generate samples every 10 rounds
                    from system.flcore.trainmodel.generators import SyntheticSample

                    print(f"\n[Server] Generating synthetic samples from Client {client.id}...")

                    # Generate prompts using the trained generator
                    num_samples = 10  # Generate 10 samples
                    synthetic_samples = []

                    for i in range(num_samples):
                        # Sample from the generator
                        audio_embedding = client.prompt_generator.sample(1, client.device)

                        # Generate the actual image using diffusion
                        # This would require the diffusion model and proper text embeddings
                        # For now, just store the audio embedding
                        synthetic_samples.append(SyntheticSample(
                            audio_embedding=audio_embedding.cpu(),
                            generated_image=None,  # Would contain the actual generated image
                            metadata={'client_id': client.id, 'round': self.round}
                        ))

                    # Store synthetic samples
                    if not hasattr(self, 'client_synthetic_samples'):
                        self.client_synthetic_samples = {}

                    self.client_synthetic_samples[client.id] = synthetic_samples

                    print(f"[Server] Collected {len(synthetic_samples)} synthetic sample sets from Client {client.id}")
            return

        # if self.eval_gap % self.round:
        #     return

        # client._move_to_gpu(client.device)
        # node_metrics = self.node_metrics(client, train_splits=self.nodes_train_metrics_splits, test_splits=self.nodes_test_metrics_splits)
        # round_train_metrics = node_metrics['train_metrics']['train'] if 'train' in node_metrics['train_metrics'] else None
        # round_test_metrics = node_metrics['test_metrics']['val'] if 'val' in node_metrics['test_metrics'] else None
        # round_test_metrics_on_train = node_metrics['test_metrics']['train'] if 'train' in node_metrics['test_metrics'] else None

        # if not self.no_wandb:
        #     wandb_metrics = {}

        #     if isinstance(round_train_metrics, NodeMetric):
        #         wandb_metrics.update(client.log_metrics(round_train_metrics, round=self.round, suffix="_end"))
        #     if isinstance(round_test_metrics, NodeMetric):
        #         wandb_metrics.update(client.log_metrics(round_test_metrics, round=self.round, suffix="_end"))
        #     if isinstance(round_test_metrics_on_train, NodeMetric):
        #         wandb_metrics.update(client.log_metrics(round_test_metrics_on_train, round=self.round, suffix="_end"))

        #     for wandb_metric in wandb_metrics:
        #         metrics = {wandb_metric: wandb_metrics[wandb_metric], "round": self.round}
        #         self.data_log(metrics)

        #  # Display only text_loss or accuracy if present
        # display_metrics = []
        # for metric_name in ['text_loss', 'accuracy']:
        #     if metric_name in round_train_metrics:
        #         display_metrics.append(f"train {metric_name} {round_train_metrics[metric_name]['mean']:.4f}")
        #     if metric_name in round_test_metrics:
        #         display_metrics.append(f"test {metric_name} {round_test_metrics[metric_name]['mean']:.4f}")
        #     if round_test_metrics_on_train is not None and metric_name in round_test_metrics_on_train:
        #         display_metrics.append(f"test_on_train {metric_name} {round_test_metrics_on_train[metric_name]['mean']:.4f}")

        # if display_metrics:
        #     print(F"Node {client.id} post-round metrics: " + ", ".join(display_metrics))

        # Generate synthetic samples if using pretrained generators
        use_pretrained_generators = getattr(self.config.feda2v, 'use_pretrained_generators', False)
        if use_pretrained_generators and hasattr(client, 'prompt_generator') and client.prompt_generator is not None:
            # Get class outputs from training
            class_outputs = client.get_training_adapter_outputs_mean()

            if class_outputs is not None and len(class_outputs) > 0:
                # Generate synthetic samples
                synthetic_samples = client.generate_synthetic_samples(class_outputs)

                # Store synthetic samples for aggregation
                if not hasattr(self, 'client_synthetic_samples'):
                    self.client_synthetic_samples = {}

                self.client_synthetic_samples[client.id] = synthetic_samples

                print(f"[Server] Collected {len(synthetic_samples)} synthetic sample sets from Client {client.id}")

        # Save client adapter checkpoint if configured
        if client.adapter_save_checkpoint and self.round % client.adapter_checkpoint_frequency == 0:
            logger.info(f"Saving adapter checkpoint for node {client.id} at round {self.round}")
            client.save_adapter_checkpoint(round_num=self.round)

    def node_metrics(self, client, train_splits=['train'], test_splits=['val'] ):
        node_metrics = {'train': {}, 'test': {}}

        for split in train_splits:
            metrics = self.round_train_metrics(client, split=split)
            node_metrics['train'][f'{split}'] = metrics
       
        for split in test_splits:
            metrics = self.round_test_metrics(client, split=split)
            node_metrics['test'][f'{split}'] = metrics

        return node_metrics

    def generate_images_from_diffusion(self, text_embeddings, base_embeddings = None):
        if 't5' not in text_embeddings or 'clip' not in text_embeddings:
            print ('Text embeddings is missing something')
            return[]

        prompt_embeds = None
        pooled_prompt_embeds = None
        imgs = None

        try:
            if base_embeddings is not None:
                orginal_prompt_embeds = []
                orginal_pooled_prompt_embeds = []
                for class_name in text_embeddings['class_name']:
                    if class_name in base_embeddings:
                        orginal_prompt_embeds.append( base_embeddings[class_name]['flux']['prompt_embeds'] )
                        orginal_pooled_prompt_embeds.append( base_embeddings[class_name]['flux']['pooled_prompt_embeds'] )
                    else:
                        print ( f"Class name {class_name} not found in base embeddings")
                        continue

            if self.generate_from_t5_text_embeddings and base_embeddings:
                prompt_embeds = []
                for class_name in text_embeddings['class_name']:
                    if class_name in base_embeddings:
                        prompt_embeds.append( base_embeddings[class_name]['flux']['prompt_embeds'] )
                    else:
                        print ( f"Class name {class_name} not found in base embeddings")
                        continue
                prompt_embeds = torch.stack(prompt_embeds).squeeze(dim=1).to(self.global_model.diffusion_dtype).to(self.diffusion_device)
            else:
                prompt_embeds = text_embeddings['t5'].to(self.global_model.diffusion_dtype).to(self.diffusion_device)

            if self.generate_from_clip_text_embeddings and base_embeddings:
                pooled_prompt_embeds = []
                for class_name in text_embeddings['class_name']:
                    if class_name in base_embeddings:
                        pooled_prompt_embeds.append( base_embeddings[class_name]['flux']['pooled_prompt_embeds'] )
                    else:
                        print ( f"Class name {class_name} not found in base embeddings")
                        continue
                pooled_prompt_embeds = torch.stack(pooled_prompt_embeds).squeeze(dim=1).to(self.global_model.diffusion_dtype).to(self.diffusion_device)
            else:
                pooled_prompt_embeds = text_embeddings['clip'].to(self.global_model.diffusion_dtype).to(self.diffusion_device)


            if not self.generate_low_memomy_footprint:
                imgs = self.global_model.diffusion_model(
                                            prompt_embeds=prompt_embeds,
                                            pooled_prompt_embeds=pooled_prompt_embeds,
                                            num_inference_steps=1,
                                            output_type="pt",
                                            ).images
            else:
                imgs = self.generate_single_images_from_diffusion(prompt_embeds, pooled_prompt_embeds)

            return imgs

        finally:
            # CRITICAL: Cleanup intermediate tensors
            if prompt_embeds is not None:
                del prompt_embeds
            if pooled_prompt_embeds is not None:
                del pooled_prompt_embeds
            torch.cuda.empty_cache()
    
    def generate_single_images_from_diffusion(self, prompt_embeds, pooled_prompt_embeds):
        imgs = []
        for pe, ppe in zip(prompt_embeds, pooled_prompt_embeds):
            pe_unsqueezed = None
            ppe_unsqueezed = None
            img = None

            try:
                pe_unsqueezed = pe.unsqueeze(0)
                ppe_unsqueezed = ppe.unsqueeze(0)
                img = self.global_model.diffusion_model(
                                            prompt_embeds=pe_unsqueezed,
                                            pooled_prompt_embeds=ppe_unsqueezed,
                                            num_inference_steps=1,
                                            output_type="pt",
                                            ).images
                imgs.append(img)

            finally:
                # Cleanup intermediate tensors
                if pe_unsqueezed is not None:
                    del pe_unsqueezed
                if ppe_unsqueezed is not None:
                    del ppe_unsqueezed
                # Note: don't delete img as it's appended to imgs list

        imgs = torch.cat(imgs,dim=0)
        return imgs
    
    def save_generated_images(self, imgs, client_id, embeddings, suffix="", output_image_base_name=None):
        saved_images = {}
        base_name = output_image_base_name if output_image_base_name is not None else self.output_image_base_name

        for idx, img in enumerate(imgs):
            if type(embeddings['class_name']) == list:
                class_name = embeddings['class_name'][idx]
            else:
                class_name = embeddings['class_name']

            img_save_path = os.path.join(self.images_output_dir, f"round_{self.round}_node_{client_id}_{base_name}_{class_name}_{suffix}_{idx}.png")
            saved_images[img_save_path] = class_name
            img = img.squeeze(0)
            converted_img = transforms.ToPILImage()(img.to(torch.float32).cpu())
            converted_img.save(img_save_path)
            print(f"Saved generated image to {img_save_path}")
        return saved_images

    def generate_images(self, client):
        """Generate images using the client's Audio2Visual model."""
        print(f"\nGenerating images for Node {client.id} using Audio2Visual model.")

        if not os.path.exists(self.images_output_dir):
            try:
                os.makedirs(self.images_output_dir)
            except FileExistsError:
                pass

        # Get all available datasets
        node_val_dataset = client.node_data.get_val_dataset()
        node_test_dataset = client.node_data.get_test_dataset()
        node_train_dataset = client.node_data.get_train_dataset()

        # Map split names to datasets
        split_datasets = {
            'val': node_val_dataset,
            'test': node_test_dataset,
            'train': node_train_dataset
        }

        # Initialize result dictionary
        generated_images_files = {}

        # Use nodes splits for per-node generation
        generation_splits = self.nodes_test_metrics_splits

        # Generate images for configured splits
        for split_name in generation_splits:
            dataset = split_datasets.get(split_name)

            if dataset is None or len(dataset) == 0:
                print(f"Unable to get {split_name} split from node {client.id}")
                continue

            # Variables for cleanup
            text_embs = None
            embeddings = None
            generated_imgs = None

            try:
                print(f"Generating images for split: {split_name}")
                text_embs = dataset.text_embs

                # Get audio embeddings from dataset (FIXED function)
                embeddings = client.get_audio_embeddings_from_dataset(dataset)

                if embeddings is None:
                    print(f"Failed to get embeddings for {split_name} split")
                    continue

                # Generate images from diffusion model
                generated_imgs = self.generate_images_from_diffusion(embeddings, base_embeddings=text_embs)

                # Save generated images
                saved_files = self.save_generated_images(
                    generated_imgs,
                    client.id,
                    embeddings,
                    suffix=f'{split_name}'
                )

                generated_images_files[split_name] = saved_files

            except Exception as e:
                logger.error(f"Error generating images for split {split_name}: {e}")
                import traceback
                traceback.print_exc()

            finally:
                # CRITICAL: Cleanup split data to prevent accumulation
                if embeddings is not None:
                    for key in list(embeddings.keys()):
                        if isinstance(embeddings[key], torch.Tensor):
                            del embeddings[key]
                    del embeddings

                if generated_imgs is not None:
                    del generated_imgs

                # Force GPU cleanup after each split
                torch.cuda.empty_cache()

                print(f"✓ Cleaned up memory after {split_name} split")

        # embeddings = client.get_audio_embeddings_for_generation(num_embeddings=2, from_train=True )

        # on_train_imgs = self.generate_images_from_diffusion(embeddings)
        # on_train_images_files = self.save_generated_images(on_train_imgs, client, embeddings, "_train")

        # node_dataset = client.node_data.train_dataset.dataset if isinstance( client.node_data.train_dataset, torch.utils.data.Subset) else client.node_data.train_dataset
        # text_embs = node_dataset.text_embs

        # for node_class in node_dataset.active_classes.keys():
        #     embeddings = { 'clip': text_embs[node_class][self.diffusion_type]['pooled_prompt_embeds'],
        #                   't5': text_embs[node_class][self.diffusion_type]['prompt_embeds'],
        #                   'class_name': node_class }
                          
        #     from_text_imgs = self.generate_images_from_diffusion(embeddings)
        #     from_text_images_files = self.save_generated_images(from_text_imgs, client, embeddings, "_from_embs")
        # for node_class in node_dataset.active_classes.keys():
        #     prompt_embeds = text_embs[node_class][self.diffusion_type]['prompt_embeds']
        #     pooled_prompt_embeds = text_embs[node_class][self.diffusion_type]['pooled_prompt_embeds']
        #     imgs = self.global_model.diffusion_model(
        #                                 prompt_embeds= prompt_embeds,
        #                                 pooled_prompt_embeds=pooled_prompt_embeds,
        #                                 num_inference_steps=1,
        #                                 output_type="pt",
        #                                 ).images
        #     img_save_path = os.path.join(self.images_output_dir, f"round_{self.round}_node_{client.id}_img_{node_class}_from_textembs.png")
        #     converted_img = transforms.ToPILImage()(imgs[0].cpu().detach())
        #     converted_img.save(img_save_path)
        #     print(f"Saved generated image from text embeddings to {img_save_path}")
        return generated_images_files

    def generate_global_images_with_aggregated_adapters(self, split='val'):
        """Generate images using aggregated adapters and federation dataset for specified split."""
        if split not in self.server_node_data_splits:
            print(f"Warning: No federation dataset available for global image generation on {split}")
            return None

        print(f"\nGenerating global images using aggregated adapters at round {self.round} on {split} split")

        saved_splits_images = {} 
        for split,node_data in self.server_node_data_splits.items():
            if split == 'val':
                fed_dataset = self.federation_val_data
            if split == 'test':
                fed_dataset = self.federation_test_data
            if split == 'train':
                logger.warning(f"Global image generation on 'train' split is not supported, skipping")
                # fed_dataset = node_data.train_dataset

            if fed_dataset is None or len(fed_dataset) == 0:
                print(f"Warning: Federation {split} dataset is empty")
                return None
            
            if isinstance(fed_dataset, torch.utils.data.ConcatDataset):
                print(f"Federation {split} dataset is a ConcatDataset with {len(fed_dataset.datasets)} datasets")
                for ds_idx, ds in enumerate(fed_dataset.datasets):
                    ds = ds.to(self.device)
            elif isinstance(fed_dataset, VEGASDataset):
                print(f"Federation {split} dataset is a VEGASDataset with length {len(fed_dataset.dataset)}")
                fed_dataset = fed_dataset.to(self.device)
            else:
                print(f"Federation {split} dataset is a dataset with length {len(fed_dataset)}")

            # Ensure output directory exists
            if not os.path.exists(self.images_output_dir):
                try:
                    os.makedirs(self.images_output_dir)
                except FileExistsError:
                    pass
            
            self._move_to_cpu( models=['diffusion'])
            self._move_to_gpu(self.device, models=['adapter', 'ast'])

            # Create dataloader for the federation dataset
            from torch.utils.data import DataLoader
            dataloader = DataLoader(fed_dataset, batch_size=len(fed_dataset), shuffle=False)

            generated_images = []
            embeddings = {}
            embeddings['class_name'] = []
            for module_name in self.global_adapters.keys():
                embeddings[module_name] = []

            text_embs = []


            with torch.no_grad():
                for batch_idx, samples in enumerate(dataloader):
                    # Move audio data to device
                    if 'audio' in samples and isinstance(samples['audio'], torch.Tensor):
                        audio_data = samples['audio'].to(self.device)
                    else:
                        print(f"Warning: Audio data not found in federation {split} batch")
                        continue

                    # Extract audio embeddings using the global model
                    # Skip if AST not initialized (using pretrained generators)
                    if self.global_model.ast_model is None or self.global_model.ast_feature_extractor is None:
                        print(f"Warning: AST model not initialized (using pretrained generators), skipping audio processing")
                        continue

                    if isinstance(audio_data, torch.Tensor):
                        audio_data_np = audio_data.to('cpu').numpy()

                    audio_inputs = self.global_model.ast_feature_extractor(
                        audio_data_np,
                        sampling_rate=16000,
                        return_tensors="pt",
                        padding=True
                    ).input_values.to(self.device, self.global_model.torch_dtype)

                    self.global_model.ast_model.eval()
                    audio_embeddings = self.global_model.ast_model(audio_inputs).last_hidden_state

                    for module_name, adapter_module in self.global_adapters.items():
                        adapter_module.eval()
                        adapter_module = adapter_module.to(self.device)
                        adapted_embeddings = adapter_module(audio_embeddings)
                        embeddings[module_name].extend(adapted_embeddings)

                    embeddings['class_name'].extend(samples['class_name'])
                    if hasattr(fed_dataset, 'text_embs'):
                        text_embs.extend(fed_dataset.text_embs)

                # Generate images from adapted embeddings
                for module_name in self.global_adapters.keys():
                    embeddings[module_name] = torch.stack(embeddings[module_name], dim=0)
                self._move_to_cpu(models=['adapter', 'ast'])
                self._move_to_gpu(self.diffusion_device, models=['diffusion']) 

                if len(text_embs):
                    generated_images = self.generate_images_from_diffusion(embeddings, base_embeddings=text_embs)
                else:
                    generated_images = self.generate_images_from_diffusion(embeddings)

                saved_splits_images |= self.save_generated_images(generated_images, "server", embeddings, suffix=split)

                self._move_to_cpu(models=['diffusion', 'adapter', 'ast'])

                print(f"Generated and saved {len(generated_images)} global images using aggregated adapters")

            if isinstance(fed_dataset, torch.utils.data.ConcatDataset):
                print(f"Federation {split} dataset is a ConcatDataset with {len(fed_dataset.datasets)} datasets")
                for ds_idx, ds in enumerate(fed_dataset.datasets):
                    ds = ds.to('cpu')
            elif isinstance(fed_dataset, VEGASDataset):
                print(f"Federation {split} dataset is a VEGASDataset with length {len(fed_dataset.dataset)}")
                fed_dataset = fed_dataset.to('cpu')
            else:
                print(f"Federation {split} dataset is a dataset with length {len(fed_dataset)}")

        return saved_splits_images

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
                # Support both new name (num_samples_per_class) and old name (num_samples) for backward compatibility
                num_samples_per_class = node_config.get('num_samples_per_class', node_config.get('num_samples', 0))

                # Get federated split parameters
                node_split_id = getattr(node_config, 'node_split_id', None)
                samples_per_node = getattr(node_config, 'samples_per_node', None)
                node_split_seed = getattr(node_config, 'node_split_seed', 42)

                train_ratio = 0.8 if getattr(node_config, 'train_ratio', 0.8) is None else getattr(node_config, 'train_ratio', 0.8)
                val_ratio = 0.1 if getattr(node_config, 'val_ratio', 0.1) == None else getattr(node_config, 'val_ratio', 0.1)
                test_ratio = 1.0 - train_ratio - val_ratio

                node_dataset = VEGASDataset(
                    selected_classes=selected_classes,
                    excluded_classes=excluded_classes,
                    num_samples_per_class=num_samples_per_class,
                    test_ratio=test_ratio,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    node_split_id=node_split_id,
                    samples_per_node=samples_per_node,
                    node_split_seed=node_split_seed,
                    use_saved_audio_embeddings=self.use_saved_audio_embeddings,
                    enable_ast_cache=True,  # Enable AST cache
                    ast_cache_dir="/home/lpala/fedgfe/dataset/VEGAS.cache"  # Centralized cache directory
                )

                # Try to load AST embeddings from cache first
                ast_cache_loaded = False
                if self.use_saved_audio_embeddings:
                    # AST cache configuration - should match how embeddings are extracted
                    ast_sample_rate = 16000
                    ast_duration = 5.0
                    ast_model_name = "ast-finetuned"  # MIT/ast-finetuned-audioset-10-10-0.4593

                    logger.info(f"Attempting to load AST embeddings from cache for node {node_id}...")
                    ast_cache_loaded = node_dataset.load_ast_embeddings_from_cache(
                        sample_rate=ast_sample_rate,
                        duration=ast_duration,
                        model_name=ast_model_name
                    )

                    if ast_cache_loaded:
                        logger.info(f"✓ AST embeddings loaded from cache for node {node_id}")
                        node_dataset.filter_audio_embeddings_from_file()
                        audio_embedding_file_loaded = True
                    else:
                        logger.info(f"AST cache not found or incompatible for node {node_id}, will try alternative methods")

                # Fallback 1: Try to copy embeddings from existing clients
                if not audio_embedding_file_loaded:
                    for client in self.clients:
                        if client.dataset == "VEGAS":
                            if isinstance(client.node_data.train_dataset, torch.utils.data.Subset):
                                dataset = client.node_data.train_dataset.dataset
                            else:
                                dataset = client.node_data.train_dataset
                            if isinstance(dataset, VEGASDataset) and dataset.audio_embs_from_file is not None:
                                audio_embedding_file_loaded = True
                                node_dataset.audio_embs_from_file = dataset.audio_embs_from_file
                                logger.info(f"Copied audio embeddings from existing client for node {node_id}")
                                break

                # Fallback 2: Load from legacy .pt file
                if self.use_saved_audio_embeddings and self.audio_embedding_file_name is not None and not audio_embedding_file_loaded:
                    node_dataset.load_audio_embeddings_from_file(self.audio_embedding_file_name)
                    logger.info(f"Loaded audio embeddings for VEGAS dataset from legacy file {self.audio_embedding_file_name}")

            elif node_config.dataset == "ESC50":
                selected_classes = getattr(node_config, 'selected_classes', None)
                excluded_classes = getattr(node_config, 'excluded_classes', None)
                # split = getattr(node_config, 'dataset_split', 'train')
                # use_folds = getattr(node_config, 'use_folds', False)
                train_folds = getattr(node_config, 'train_folds', [0, 1, 2, 3])
                test_folds = getattr(node_config, 'test_folds', [4])

                train_ratio = 0.8 if getattr(node_config, 'train_ratio', 0.8) is None else getattr(node_config, 'train_ratio', 0.8)
                val_ratio = 0.1 if getattr(node_config, 'val_ratio', 0.1) == None else getattr(node_config, 'val_ratio', 0.1)
                test_ratio = 1.0 - train_ratio - val_ratio


                node_dataset = ESC50Dataset(
                    selected_classes=selected_classes,
                    excluded_classes=excluded_classes,
                    node_id=int(node_id),
                    test_ratio=test_ratio,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio
                )

            elif node_config.dataset == "VGGSound":
                audio_embedding_file = "/home/lpala/fedgfe/dataset/Audio/vggsound_audio_embs_dict.pt"
                audio_embedding_file_loaded = False
                selected_classes = getattr(node_config, 'selected_classes', None)
                excluded_classes = getattr(node_config, 'excluded_classes', None)
                # Support both new name (num_samples_per_class) and old name (num_samples) for backward compatibility
                num_samples_per_class = node_config.get('num_samples_per_class', node_config.get('num_samples', 0))

                train_ratio = 0.7 if getattr(node_config, 'train_ratio', 0.7) is None else getattr(node_config, 'train_ratio', 0.7)
                val_ratio = 0.1 if getattr(node_config, 'val_ratio', 0.1) is None else getattr(node_config, 'val_ratio', 0.1)
                test_ratio = 0.2 if getattr(node_config, 'test_ratio', 0.2) is None else getattr(node_config, 'test_ratio', 0.2)
                use_official_split = getattr(node_config, 'use_official_split', True)

                node_dataset = VGGSoundDataset(
                    root_dir="/home/lpala/fedgfe/dataset/Audio/vggsound",
                    text_embedding_file="/home/lpala/fedgfe/dataset/Audio/vggsound_text_embs_dict.pt",
                    audio_embedding_file=audio_embedding_file if self.use_saved_audio_embeddings else None,
                    selected_classes=selected_classes,
                    excluded_classes=excluded_classes,
                    num_samples_per_class=num_samples_per_class if num_samples_per_class > 0 else None,
                    node_id=int(node_id),
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    use_official_split=use_official_split,
                    stratify=True,
                    load_audio=True,
                    load_video=False,
                    load_image=False
                )

                # Load audio embeddings from existing clients if available
                for client in self.clients:
                    if client.dataset == "VGGSound":
                        if isinstance(client.node_data.train_dataset, torch.utils.data.Subset):
                            dataset = client.node_data.train_dataset.dataset
                        else:
                            dataset = client.node_data.train_dataset
                        if isinstance(dataset, VGGSoundDataset) and hasattr(dataset, 'audio_embs') and dataset.audio_embs is not None:
                            audio_embedding_file_loaded = True
                            if hasattr(node_dataset, 'audio_embs'):
                                node_dataset.audio_embs = dataset.audio_embs
                            break

                if self.use_saved_audio_embeddings and self.audio_embedding_file_name is not None and not audio_embedding_file_loaded:
                    if hasattr(node_dataset, 'load_audio_embeddings_from_file'):
                        node_dataset.load_audio_embeddings_from_file(self.audio_embedding_file_name)
                        logger.info(f"Loaded audio embeddings for VGGSound dataset from {self.audio_embedding_file_name}")

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

        # If we have multiple adapter sets, distribute them round-robin to clients
        if hasattr(self, '_multiple_adapter_sets') and self._multiple_adapter_sets is not None:
            print(f"\n[Server] Distributing {len(self._multiple_adapter_sets)} adapter sets to {len(self.clients)} clients in round-robin fashion")

            node_list = list(self._multiple_adapter_sets.keys())
            adapter_types = ['clip_adapter', 't5_adapter', 'clip_projection', 't5_projection']

            for client_idx, client in enumerate(self.clients):
                # Assign adapter set in round-robin
                assigned_node_id = node_list[client_idx % len(node_list)]
                print(f"[Server] Client {client.id} -> loading adapters from node {assigned_node_id}")

                # Load each adapter type for this client
                for adapter_name in adapter_types:
                    checkpoint_files = sorted(self._multiple_adapter_sets[assigned_node_id][adapter_name])
                    checkpoint_file = checkpoint_files[-1]  # Use most recent

                    # Load adapter checkpoint into this specific client
                    loaded = self._load_single_adapter_checkpoint_to_client(
                        checkpoint_file, client, adapter_name
                    )

                    if not loaded:
                        print(f"[Server] ⚠ Failed to load {adapter_name} for client {client.id}")

     

        self.federation_available_classes = []
        self.federation_active_classes = []
        for node in self.clients:
            node_dataset = node.node_data.dataset if not isinstance(node.node_data.dataset, torch.utils.data.Subset) else node.node_data.dataset.dataset
            self.federation_available_classes.extend(node_dataset.available_classes)
            self.federation_available_classes.extend(node_dataset.available_classes)
            self.federation_active_classes.extend(node_dataset.active_classes)
            self.federation_active_classes.extend(node_dataset.active_classes)


        self.federation_available_classes = list(set(self.federation_available_classes))
        self.federation_active_classes = list(set(self.federation_active_classes))

        # Dictionary to store datasets for each split
        split_datasets = {split_name: [] for split_name in self.server_test_metrics_splits}

        # Mapping from split names to NodeData getter methods
        split_getters = {
            'train': 'get_train_dataset',
            'val': 'get_val_dataset',
            'test': 'get_test_dataset'
        }

        # Collect datasets from all nodes for each split
        for node in self.clients:
            for split_name in self.server_test_metrics_splits:
                if split_name in split_getters:
                    getter_method = split_getters[split_name]
                    if hasattr(node.node_data, getter_method):
                        dataset = getattr(node.node_data, getter_method)()
                        if dataset is not None:
                            split_datasets[split_name].append(dataset)
                else:
                    print(f"Warning: Unknown split name '{split_name}', skipping")

        # Create NodeData instances for each split with merged datasets
        self.server_node_data_splits = {}

        for split_name, datasets in split_datasets.items():
            if datasets:
                merged_dataset = ConcatDataset(datasets)

                # Determine which parameter to use based on split name
                kwargs = {
                    'node_id': -1,  # Use -1 to indicate this is the server-level data
                    'dataset_split_id': -1
                }

                if split_name == 'train':
                    kwargs['custom_train_dataset'] = merged_dataset
                elif split_name == 'val':
                    kwargs['custom_val_dataset'] = merged_dataset
                elif split_name == 'test':
                    kwargs['custom_test_dataset'] = merged_dataset

                # Get collate_fn from first client's node_data
                collate_fn = None
                if len(self.clients) > 0:
                    collate_fn = self.clients[0].node_data.collate_fn

                # Create NodeData instance
                self.server_node_data_splits[split_name] = NodeData(
                    self.args,
                    collate_fn=collate_fn,
                    **kwargs
                )
                print(f"Created server {split_name} dataset with {len(merged_dataset)} samples from {len(datasets)} nodes")
            else:
                self.server_node_data_splits[split_name] = None
                print(f"Warning: No {split_name} datasets found in any node")

        # Maintain backward compatibility with existing code
        self.federation_val_data = self.server_node_data_splits.get('val', None)
        self.federation_test_data = self.server_node_data_splits.get('test', None)

        self.federation_available_classes = list((set(self.federation_available_classes)))
        self.federation_active_classes = list((set(self.federation_active_classes)))

        self.federation_available_classes.sort()
        self.federation_active_classes.sort()

        self.federation_available_classes = {v: i for i, v in enumerate(self.federation_available_classes)}
        self.federation_active_classes = {v: i for i, v in enumerate(self.federation_active_classes)}

        # Setup synthetic_samples_per_class based on federation dataset size
        self._setup_synthetic_samples_count_for_federation()

        # Load generator checkpoints after clients are created and federation classes are collected
        if self.use_generator and self.generator_load_checkpoint:
            logger.info("Loading generator checkpoints now that federation classes are available **TEMPORARY DISABLED**")
            success = self.load_generator_checkpoint()
            if success:
                logger.info("Successfully loaded generator checkpoint")
                # Assign generators to global_model after loading
                self.global_model.generators_dict = self.prompt_generators if hasattr(self, 'prompt_generators') else None

                # If shared_generator_in_only_mode is active, also set prompt_generator references
                if self.shared_generator_in_only_mode and self.generator_only_mode:
                    if hasattr(self, 'prompt_generator') and self.prompt_generator is not None:
                        self.global_model.prompt_generator = self.prompt_generator
                        logger.info("Set unified prompt_generator in global_model for shared access")
                    if hasattr(self, 'prompt_generator_clip') and self.prompt_generator_clip is not None:
                        self.global_model.prompt_generator_clip = self.prompt_generator_clip
                        logger.info("Set prompt_generator_clip in global_model for shared access")
                    if hasattr(self, 'prompt_generator_t5') and self.prompt_generator_t5 is not None:
                        self.global_model.prompt_generator_t5 = self.prompt_generator_t5
                        logger.info("Set prompt_generator_t5 in global_model for shared access")
            else:
                logger.warning("Could not load generator checkpoint, initializing from scratch")
                self.initialize_generators()
                # Assign generators to global_model after initialization
                self.global_model.generators_dict = self.prompt_generators if hasattr(self, 'prompt_generators') else None

            # Generate synthetic samples for all nodes after loading generators
            logger.info("\n[Server] Generating synthetic samples for all nodes after generator loading")
            # DEBUG: Use random tensors instead of actual generation for speed
            self.generate_synthetic_samples_for_all_nodes()  # Original version - uncomment when done debugging
            # self.generate_random_synthetic_samples_for_all_nodes()

            # Distribute samples to nodes
            self.distribute_synthetic_samples_to_nodes()

        elif self.use_generator:
            for client in self.clients:
                for generator_class, generator in client.prompt_generators.items():
                    if generator_class not in self.generators:
                        self.generators[generator_class] = generator
                    else:
                        logging.warning(f"Generator class {generator_class} already exists in server generators, skipping addition from client {client.id}")


        return self.clients

    def set_clients(self, clientObj):
        return self.create_clients(clientObj)

    def _setup_synthetic_samples_count_for_federation(self):
        """
        Setup the number of synthetic samples per class for the server.
        Creates a dict structure:
        {
            'count': {node_id: {class_name: num_samples}},
            node_id: {class_name: tensor}  # populated by generate_synthetic_samples_for_all_nodes
        }

        If synthetic_samples_per_class is "auto" or None, calculate it based on each node's dataset size.
        """
        # Create dict structure with 'count' key for the counts
        self.synthetic_samples_per_node = {'count': {}}

        # Check if synthetic_samples_per_class is set to "auto" or None
        if self.synthetic_samples_per_class == "auto" or self.synthetic_samples_per_class is None:
            # Calculate samples per class for each node individually
            logger.info("[Server] Auto-calculating synthetic_samples_per_class for each node and class")

            for client in self.clients:
                node_id = client.id
                self.synthetic_samples_per_node['count'][node_id] = {}

                if hasattr(client.node_data, 'train_dataset') and client.node_data.train_dataset is not None:
                    train_dataset = client.node_data.train_dataset

                    # Get the actual dataset (unwrap Subset if needed)
                    if isinstance(train_dataset, torch.utils.data.Subset):
                        base_dataset = train_dataset.dataset
                    else:
                        base_dataset = train_dataset

                    # Get classes for this node
                    node_classes = base_dataset.active_classes if hasattr(base_dataset, 'active_classes') else []

                    if node_classes:
                        # Calculate samples per class for this node
                        total_node_samples = len(train_dataset)
                        num_node_classes = len(node_classes)
                        samples_per_class = max(1, total_node_samples // num_node_classes)

                        # Assign same count to all classes of this node
                        for class_name in node_classes:
                            self.synthetic_samples_per_node['count'][node_id][class_name] = samples_per_class

                        logger.info(f"[Server] Node {node_id}: {samples_per_class} synthetic samples/class "
                                   f"(total samples: {total_node_samples}, classes: {num_node_classes})")
                    else:
                        logger.warning(f"[Server] Node {node_id}: No classes found, using default = 5")
                else:
                    logger.warning(f"[Server] Node {node_id}: No train dataset, using default = 5")

            # Calculate average for fallback (if needed later)
            total_samples = sum(
                sum(counts.values()) for counts in self.synthetic_samples_per_node['count'].values()
            )
            total_classes = sum(
                len(counts) for counts in self.synthetic_samples_per_node['count'].values()
            )

            if total_classes > 0:
                avg_samples_per_class = max(1, total_samples // total_classes)
                self.synthetic_samples_per_class = avg_samples_per_class
                logger.info(f"[Server] Average synthetic_samples_per_class = {avg_samples_per_class}")
            else:
                self.synthetic_samples_per_class = 5
                logger.info(f"[Server] Could not determine dataset size, using default = 5")

        else:
            # Explicit value provided in configuration - use same for all nodes/classes
            logger.info(f"[Server] Using configured synthetic_samples_per_class = {self.synthetic_samples_per_class}")

            for client in self.clients:
                node_id = client.id
                self.synthetic_samples_per_node['count'][node_id] = {}

                # Get classes for this node
                if hasattr(client.node_data, 'dataset'):
                    dataset = client.node_data.dataset
                    if isinstance(dataset, torch.utils.data.Subset):
                        dataset = dataset.dataset

                    node_classes = dataset.available_classes if hasattr(dataset, 'available_classes') else []

                    # Assign configured count to all classes
                    for class_name in node_classes:
                        self.synthetic_samples_per_node['count'][node_id][class_name] = self.synthetic_samples_per_class

            logger.info(f"[Server] Will generate {self.synthetic_samples_per_class} synthetic samples for each class")

    def generate_synthetic_samples_for_all_nodes(self):
        """
        Generate synthetic samples for ALL nodes, considering each node's classes.

        Stores samples directly in self.synthetic_samples_per_node[node_id][class_name].
        Uses self.synthetic_samples_per_node['count'][node_id][class_name] for sample counts.

        Structure after generation:
        {
            'count': {node_id: {class_name: num_samples}},
            node_id: {class_name: tensor(num_samples, seq_len, 768)}
        }
        """
        if not hasattr(self, 'prompt_generators') or not self.prompt_generators:
            logger.warning("[Server] No generators available, cannot generate synthetic samples")
            return

        if not hasattr(self, 'clients') or not self.clients:
            logger.warning("[Server] No clients available, cannot generate synthetic samples")
            return

        logger.info(f"\n{'='*80}")
        logger.info(f"[Server] Generating synthetic samples for all {len(self.clients)} nodes")
        logger.info(f"{'='*80}")

        # Get target sequence length for generation
        target_seq_len = getattr(self, 'generator_output_sequence_length', None)
        training_seq_len = getattr(self, 'generator_training_sequence_length', 4)
        seq_info = f"[{target_seq_len if target_seq_len else training_seq_len}, 768]"

        total_samples_generated = 0
        total_classes_processed = 0

        # Generate samples for each node using info from 'count' dict
        if 'count' not in self.synthetic_samples_per_node:
            logger.error("[Server] 'count' key not found in synthetic_samples_per_node")
            return

        for node_id, classes_dict in self.synthetic_samples_per_node['count'].items():
            logger.info(f"\n[Server] Generating samples for Node {node_id} ({len(classes_dict)} classes): {list(classes_dict.keys())}")

            # Initialize node entry in synthetic_samples_per_node
            self.synthetic_samples_per_node[node_id] = {}

            node_generated = 0
            node_missing = 0

            # Generate samples for each class of this node
            for class_name, num_samples_for_class in classes_dict.items():
                if class_name not in self.prompt_generators:
                    logger.warning(f"  ✗ No generator for class '{class_name}'")
                    node_missing += 1
                    continue

                generator = self.prompt_generators[class_name]
                generator.eval()

                with torch.no_grad():
                    try:
                        # Generate samples on CPU (generators are kept on CPU for memory efficiency)
                        synthetic_audio_embs = generator.sample(
                            num_samples=num_samples_for_class,
                            device='cpu',
                            target_sequence_length=target_seq_len
                        )

                        # Store directly in self.synthetic_samples_per_node[node_id][class_name]
                        self.synthetic_samples_per_node[node_id][class_name] = synthetic_audio_embs
                        self.synthetic_samples_per_node[class_name] = synthetic_audio_embs
                        node_generated += 1
                        total_samples_generated += num_samples_for_class
                        logger.info(f"  ✓ Generated {num_samples_for_class} samples for '{class_name}' {seq_info}")

                    except Exception as e:
                        logger.error(f"  ✗ Error generating samples for class '{class_name}': {e}")
                        node_missing += 1

            total_classes_processed += node_generated
            logger.info(f"[Server] Node {node_id} summary: {node_generated} classes generated, {node_missing} missing")

        # Count actual nodes (excluding 'count' key)
        num_nodes = len([k for k in self.synthetic_samples_per_node.keys() if k != 'count'])

        logger.info(f"\n{'='*80}")
        logger.info(f"[Server] Generation complete:")
        logger.info(f"  - Nodes: {num_nodes}")
        logger.info(f"  - Total classes processed: {total_classes_processed}")
        logger.info(f"  - Total samples generated: {total_samples_generated}")
        logger.info(f"  - Sample shape: {seq_info}")
        logger.info(f"{'='*80}\n")

    def generate_random_synthetic_samples_for_all_nodes(self):
        """
        DEBUG VERSION: Generate RANDOM synthetic samples for ALL nodes, considering each node's classes.
        This version creates random tensors instead of using the actual generators for faster debugging.

        Stores samples directly in self.synthetic_samples_per_node[node_id][class_name].
        Uses self.synthetic_samples_per_node['count'][node_id][class_name] for sample counts.

        Structure after generation:
        {
            'count': {node_id: {class_name: num_samples}},
            node_id: {class_name: tensor(num_samples, seq_len, 768)}
        }
        """
        if not hasattr(self, 'clients') or not self.clients:
            logger.warning("[Server] No clients available, cannot generate synthetic samples")
            return

        logger.info(f"\n{'='*80}")
        logger.info(f"[Server] [DEBUG MODE] Generating RANDOM synthetic samples for all {len(self.clients)} nodes")
        logger.info(f"{'='*80}")

        # Get target sequence length for generation
        target_seq_len = getattr(self, 'generator_output_sequence_length', None)
        training_seq_len = getattr(self, 'generator_training_sequence_length', 4)
        seq_len = target_seq_len if target_seq_len else training_seq_len
        seq_info = f"[{seq_len}, 768]"

        total_samples_generated = 0
        total_classes_processed = 0

        # Generate samples for each node using info from 'count' dict
        if 'count' not in self.synthetic_samples_per_node:
            logger.error("[Server] 'count' key not found in synthetic_samples_per_node")
            return

        for node_id, classes_dict in self.synthetic_samples_per_node['count'].items():
            logger.info(f"\n[Server] Generating RANDOM samples for Node {node_id} ({len(classes_dict)} classes): {list(classes_dict.keys())}")

            # Initialize node entry in synthetic_samples_per_node
            self.synthetic_samples_per_node[node_id] = {}

            node_generated = 0

            # Generate samples for each class of this node
            for class_name, num_samples_for_class in classes_dict.items():
                # Generate random tensors with shape [num_samples, seq_len, 768]
                synthetic_audio_embs = torch.randn(num_samples_for_class, seq_len, 768)

                # Store directly in self.synthetic_samples_per_node[node_id][class_name]
                self.synthetic_samples_per_node[node_id][class_name] = synthetic_audio_embs
                node_generated += 1
                total_samples_generated += num_samples_for_class
                logger.info(f"  ✓ Generated {num_samples_for_class} RANDOM samples for '{class_name}' {seq_info}")

            total_classes_processed += node_generated
            logger.info(f"[Server] Node {node_id} summary: {node_generated} classes generated")

        # Count actual nodes (excluding 'count' key)
        num_nodes = len([k for k in self.synthetic_samples_per_node.keys() if k != 'count'])

        logger.info(f"\n{'='*80}")
        logger.info(f"[Server] [DEBUG MODE] Random generation complete:")
        logger.info(f"  - Nodes: {num_nodes}")
        logger.info(f"  - Total classes processed: {total_classes_processed}")
        logger.info(f"  - Total RANDOM samples generated: {total_samples_generated}")
        logger.info(f"  - Sample shape: {seq_info}")
        logger.info(f"{'='*80}\n")

    def distribute_synthetic_samples_to_nodes(self):
        """
        Distribute the generated synthetic samples to their respective nodes.

        Reads from self.synthetic_samples_per_node[node_id] and assigns to client.server_synthetic_samples
        """
        if not hasattr(self, 'synthetic_samples_per_node'):
            logger.warning("[Server] No synthetic_samples_per_node available")
            return

        # Count nodes (exclude 'count' key)
        node_ids = [k for k in self.synthetic_samples_per_node.keys() if k != 'count']

        if not node_ids:
            logger.warning("[Server] No synthetic samples to distribute")
            return

        logger.info(f"[Server] Distributing synthetic samples to {len(node_ids)} nodes")

        for client in self.clients:
            node_id = client.id

            if node_id not in self.synthetic_samples_per_node or node_id == 'count':
                logger.warning(f"[Server] No synthetic samples for Node {node_id}")
                continue

            # Assign synthetic samples to the client
            client.server_synthetic_samples = self.synthetic_samples_per_node[node_id]

            num_classes = len(client.server_synthetic_samples)
            total_samples = sum(samples.shape[0] for samples in client.server_synthetic_samples.values())

            logger.info(f"[Server] → Node {node_id}: {num_classes} classes, {total_samples} total samples")

        logger.info(f"[Server] Synthetic sample distribution complete\n")

    def define_metrics(self):
        # Define round as the primary step metric
        wandb.define_metric("round")

        # Define generator-specific metrics only if generator training is enabled
        if self.use_generator or self.global_model_train_from_nodes_adapters:
            wandb.define_metric("server/generator_loss", step_metric="round")
            wandb.define_metric("server/adapter_finetuning_loss", step_metric="round")
            wandb.define_metric("server/generator_validation_similarity", step_metric="round")
            wandb.define_metric("server/generator_validation_mse", step_metric="round")
            wandb.define_metric("server/generator_validation_l1", step_metric="round")

        # Define synthetic samples metrics (used in KD aggregation)
        wandb.define_metric("server/synthetic_samples_total", step_metric="round")
        wandb.define_metric("server/synthetic_samples_classes", step_metric="round")

        self.metrics_path = "server/"
        if self.global_model is not None:
            self.global_model.define_metrics( metrics_path=self.metrics_path, train_splits = self.server_train_metrics_splits, test_splits=self.server_test_metrics_splits )

        for client in self.clients:
            client.define_metrics()

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
            elif metrics.phase == NodeMetric.Phase.VALIDATE:
                metric_path = f"test/{self.metrics_path}{prefix}{metric_name}{suffix}"
                # metric_path = f"test/{self.metrics_path}a2v_{prefix}{metric_name}{suffix}"
            metric_value = metrics[metric_name]['mean']
            if round == None:
                round = self.round
            
            metrics_values[metric_path] = metric_value

        return metrics_values

    def evaluate(self):
        """Evaluate Audio2Visual models."""
        # Skip evaluation in generator-only training mode
        if self.generator_training_mode:
            print("[Generator Training Mode] Skipping model evaluation")
            return

        stats_test = self.test_metrics(standalone=True)
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
        """
        Calculate training metrics for Audio2Visual clients.
        Uses training dataset for metrics computation.

        Returns:
            Dictionary mapping client IDs to their training metrics
        """
        # Skip if adapters are not being trained
        if not self.train_adapters:
            print("[Server] Skipping train_metrics: adapters not being trained (train_adapters=False)")
            return {}

        nodes_train_metrics = {}

        for client_index, c in enumerate(self.clients):
            if c.node_data.train_dataset == None:
                print(f"Client {c.id} train data is None")
                continue
            c._move_to_gpu(self.device, force=True)
            print ( f"Node {c.id}" ) 
            node_train_metrics = c.train_metrics()
            nodes_train_metrics[c.id] = node_train_metrics
            c._move_to_cpu()

        return nodes_train_metrics

    def test_metrics(self, standalone=None):
        """
        Calculate test metrics for Audio2Visual clients using test dataset.
        Test is performed only when adapters are being trained.

        Args:
            standalone: If True, each client tests only on itself.
                       If None (default), automatically determined:
                       - True if adapter_aggregation_mode == 'none' (no aggregation)
                       - False otherwise (with aggregation, cross-client testing)

        Returns:
            Dictionary mapping client IDs to their test metrics
        """
        # Skip test if adapters are not being trained
        # This ensures test set is only used when adapters are actually learning
        if not self.train_adapters:
            print("[Server] Skipping test_metrics: adapters not being trained (train_adapters=False)")
            return {}

        # Auto-determine standalone mode based on aggregation configuration
        if standalone is None:
            standalone = (self.adapter_aggregation_mode == 'none')
            if standalone:
                print("[Server] No adapter aggregation configured: testing each node on its own test set only")
            else:
                print(f"[Server] Adapter aggregation mode '{self.adapter_aggregation_mode}': performing cross-client testing")

        test_clients_stats = {}

        for client_index, c in enumerate(self.clients):
            if c.node_data.test_dataset == None:
                print(f"Client {c.id} test data is None")
                continue

            test_clients_stats[c.id] = {}

            # Pre-load next client's data if applicable
            if client_index < len(self.clients) - 1:
                self.clients[client_index + 1].node_data.load_test_data(self.args.batch_size)

            # Determine which clients to test on
            # If standalone: only test on self
            # If cross-client: test on all clients
            test_clients = [c] if standalone else self.clients

            for t in test_clients:
                if c.node_data.test_dataset == None:
                    print(f"Client {c.id} test data is None")
                    continue

                c._move_to_gpu(c.device)

                node_test_metrics = c.test_metrics(t)
                # node_test_metrics_on_train = c.test_metrics(t, on_train=True)

                c._move_to_cpu()
                test_clients_stats[t.id] = node_test_metrics
                # test_clients_stats[f"{t.id}_on_train"] = node_test_metrics_on_train

            # Unload test data to save memory
            if client_index > 0 and self.reduce_memory_footprint == True:
                c.node_data.unload_test_data()

        return test_clients_stats

    def round_train_metrics(self, client, split='train'):

        train_metric = client.train_metrics(split=split)

        node_loss_aggregated = train_metric['text_loss']['mean'] if 'text_loss' in train_metric else 0.0
        round_loss = node_loss_aggregated
        logger.debug(f"Node {client.id} round train loss: {round_loss}")
        
        return train_metric

    def round_test_metrics(self, node, split='test'):

        metrics = node.test_metrics(use_generated_images=True, generation_split=split)
        return metrics
    
    def _move_to_device(self, device, models=['adapter', 'diffusion', 'zeroshot', 'ast']):
        if 'adapter' in models:
            self.global_model = self.global_model.to(device)

        if 'zeroshot' in models:
            self.global_model.zero_shot_model.model = self.global_model.zero_shot_model.model.to(device)

        if 'diffusion' in models:
            self.global_model.diffusion_model = self.global_model.diffusion_model.to(device)

        if 'ast' in models and self.global_model.ast_model is not None:
            self.global_model.ast_model = self.global_model.ast_model.to(device)

        # Move optimizer state if needed
        if hasattr(self, 'global_optimizers') and self.global_optimizers is not None:
            for optimizer_module, optimizer in self.global_optimizers.items():
                if isinstance(optimizer, torch.optim.AdamW):
                    move_optimizer_state(optimizer, device)

        # Move generators if they exist
        if 'generator' in models and hasattr(self, 'prompt_generator') and self.prompt_generator is not None:
            self.prompt_generator = self.prompt_generator.to(device)

        if 'generator' in models and hasattr(self, 'prompt_generator_clip') and self.prompt_generator_clip is not None:
            self.prompt_generator_clip = self.prompt_generator_clip.to(device)

        if 'generator' in models and hasattr(self, 'prompt_generator_t5') and self.prompt_generator_t5 is not None:
            self.prompt_generator_t5 = self.prompt_generator_t5.to(device)
        
        if 'nodes_adapters' in models:
            for client in self.clients:
                client._move_to_device(device, models=['adapter'])

        if 'global_adapters' in models:
            for adapter_name, adapter in self.global_model.adapters.items():
                adapter = adapter.to(device)
                if adapter_name in self.global_optimizers:
                    optimizer = self.global_optimizers[adapter_name]
                    move_optimizer_state(optimizer, device)

        # Move generator optimizer if it exists
        if hasattr(self, 'generator_optimizer') and self.generator_optimizer is not None:
            move_optimizer_state(self.generator_optimizer, device)

        torch.cuda.empty_cache()


    def _move_to_gpu(self, device, models=['adapter', 'diffusion', 'zeroshot', 'ast']):
        if self.optimize_memory_usage or self.round <= 1:
            logger.debug(f"Server moving to GPU: {device}")
            self._move_to_device(device, models=models)

    def _move_to_cpu(self, models=['adapter', 'diffusion', 'zeroshot', 'ast']):
        if self.optimize_memory_usage:
            logger.debug(f"Server moving to CPU for memory optimization")
            self._move_to_device('cpu', models=models)

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

def load_images_with_class_from_path(images_path, device='cuda', filter_round=None, use_last_round=False,
                                      filter_node=None, include_train=True, include_test=True,
                                      include_textembs=True):
    """
    Carica immagini da una directory e estrae classe e metadati dal nome file.

    Formato nome file atteso: round_X_node_Y_img_CLASSNAME_Z.png
                              round_X_node_Y_img_CLASSNAME_Z_train.png
                              round_X_node_Y_img_CLASSNAME_from_textembs.png

    Args:
        images_path: Path alla directory con le immagini
        device: Device su cui caricare i tensor
        filter_round: int o None - Carica solo immagini da un round specifico
        use_last_round: bool - Se True, carica solo le immagini dell'ultimo round disponibile
        filter_node: int o None - Carica solo immagini da un nodo specifico
        include_train: bool - Include immagini con suffisso "_train"
        include_test: bool - Include immagini senza suffisso (test/validation)
        include_textembs: bool - Include immagini generate da text embeddings

    Returns:
        dict: {
            class_name: {
                'images': torch.Tensor (N, C, H, W),
                'filenames': list of str,
                'paths': list of str,
                'rounds': list of int,
                'nodes': list of int,
                'image_types': list of str  # 'train', 'test', 'textembs'
            }
        }
    """
    import re
    from pathlib import Path
    from PIL import Image
    from collections import defaultdict

    # Trasformazione standard per le immagini
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    images_path = Path(images_path)
    data_by_class = defaultdict(lambda: {
        'images': [],
        'filenames': [],
        'paths': [],
        'rounds': [],
        'nodes': [],
        'image_types': []
    })

    # Pattern completo per estrarre round, node, classe e tipo
    # Formato: round_X_node_Y_img_CLASSNAME_Z.png
    pattern_standard = re.compile(r'round_(\d+)_node_(\d+)_img_(.+?)_(\d+)\.png')
    # Formato: round_X_node_Y_img_CLASSNAME_Z_train.png
    pattern_train = re.compile(r'round_(\d+)_node_(\d+)_img_(.+?)_(\d+)_train\.png')
    # Formato: round_X_node_Y_img_CLASSNAME_from_textembs.png
    pattern_textembs = re.compile(r'round_(\d+)_node_(\d+)_img_(.+?)_from_textembs\.png')

    # Prima fase: trova tutti i round disponibili se use_last_round è True
    all_rounds = set()
    if use_last_round:
        for img_file in images_path.glob('*.png'):
            filename = img_file.name
            for pattern in [pattern_standard, pattern_train, pattern_textembs]:
                match = pattern.match(filename)
                if match:
                    all_rounds.add(int(match.group(1)))
                    break

        if all_rounds:
            filter_round = max(all_rounds)
            print(f"Using last round: {filter_round}")
        else:
            print("Warning: No rounds found in image filenames")

    # Seconda fase: carica le immagini con i filtri
    for img_file in sorted(images_path.glob('*.png')):
        filename = img_file.name

        match = None
        image_type = None

        # Prova i pattern in ordine
        if include_test:
            match = pattern_standard.match(filename)
            if match:
                image_type = 'test'

        if not match and include_train:
            match = pattern_train.match(filename)
            if match:
                image_type = 'train'

        if not match and include_textembs:
            match = pattern_textembs.match(filename)
            if match:
                image_type = 'textembs'

        if match:
            round_num = int(match.group(1))
            node_num = int(match.group(2))
            class_name = match.group(3)

            # Applica filtri
            if filter_round is not None and round_num != filter_round:
                continue

            if filter_node is not None and node_num != filter_node:
                continue

            try:
                # Carica l'immagine
                img = Image.open(img_file).convert('RGB')
                img_tensor = transform(img)

                # Salva i dati
                data_by_class[class_name]['images'].append(img_tensor)
                data_by_class[class_name]['filenames'].append(filename)
                data_by_class[class_name]['paths'].append(str(img_file))
                data_by_class[class_name]['rounds'].append(round_num)
                data_by_class[class_name]['nodes'].append(node_num)
                data_by_class[class_name]['image_types'].append(image_type)

            except Exception as e:
                print(f"Error loading {filename}: {e}")
        else:
            print(f"Warning: Could not parse metadata from {filename}")

    # Converti liste di tensor in batch tensor
    result = {}
    for class_name, data in data_by_class.items():
        if len(data['images']) > 0:
            result[class_name] = {
                'images': torch.stack(data['images']).to(device),
                'filenames': data['filenames'],
                'paths': data['paths'],
                'rounds': data['rounds'],
                'nodes': data['nodes'],
                'image_types': data['image_types']
            }
            print(f"Loaded class '{class_name}': {len(data['filenames'])} images "
                  f"(rounds: {sorted(set(data['rounds']))}, nodes: {sorted(set(data['nodes']))})")

    return result


def prepare_for_metrics(data_by_class):
    """
    Prepara i dati per passarli alle metriche del modello.

    Args:
        data_by_class: Output di load_images_with_class_from_path()

    Returns:
        Tuple (images, class_names, filenames):
        - images: torch.Tensor (N, C, H, W)
        - class_names: List[str] - classe per ogni immagine
        - filenames: List[str] - nome file per ogni immagine
    """
    all_images = []
    all_class_names = []
    all_filenames = []

    for class_name, data in data_by_class.items():
        num_images = len(data['filenames'])

        all_images.append(data['images'])
        all_class_names.extend([class_name] * num_images)
        all_filenames.extend(data['paths'])

    # Concatena tutti i tensor
    all_images = torch.cat(all_images, dim=0)

    return all_images, all_class_names, all_filenames

    def audio_embeddings_dataset_cache(self, samples, outputs, dataloader=None):
        """
        Cache audio embeddings from model outputs to dataset.
        Used by server when it has fed_dataset (concatenated from all nodes).

        Args:
            samples: Batch samples
            outputs: Model outputs containing audio embeddings
            dataloader: DataLoader instance (optional)
        """
        audio_embeddings_batch = outputs['audio_embeddings']
        file_ids = samples.get('file_id', None)
        classes = samples.get('class_name', 'unknown')

        # Determine the base dataset
        if dataloader is None:
            # Server uses fed_dataset (concatenated dataset from nodes)
            if hasattr(self, 'fed_dataset'):
                base_dataset = self.fed_dataset
            else:
                logger.warning("Server: No dataloader or fed_dataset available for caching")
                return
        else:
            train_dataset = dataloader.dataset
            if hasattr(train_dataset, 'dataset'):
                base_dataset = train_dataset.dataset
            else:
                base_dataset = train_dataset

        if file_ids is not None:
            # Initialize audio_embs if not exists
            if not hasattr(base_dataset, 'audio_embs') or base_dataset.audio_embs is None:
                base_dataset.audio_embs = {}

            # Cache embeddings with format: file_id:class_name
            for file_id, class_name, audio_emb in zip(file_ids, classes, audio_embeddings_batch):
                file_index = f'{file_id}:{class_name}'
                # Store on CPU to save memory
                base_dataset.audio_embs[file_index] = audio_emb.detach().cpu()

            logger.debug(f"Server: stored {len(file_ids)} audio embeddings in dataset "
                        f"(total: {len(base_dataset.audio_embs)})")