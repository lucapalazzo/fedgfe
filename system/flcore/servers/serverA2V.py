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

# Import generators
from flcore.trainmodel.generators import ConditionedVAEGenerator, VAELoss, GANGenerator, GANDiscriminator
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

        self.adapter_aggregation_mode = self.config.feda2v.adapter_aggregation_mode
        self.global_model_train_from_nodes_adapters = self.config.feda2v.global_model_train_from_nodes_adapters
        self.global_model_train_from_generator = self.config.feda2v.global_model_train_from_generator

        # Generator configuration
        self.generator_type = getattr(self.config.feda2v, 'generator_type', 'vae')
        self.use_generator = getattr(self.config.feda2v, 'use_generator', False)
        self.prompt_generator = None
        self.prompt_generator_clip = None
        self.prompt_generator_t5 = None
        self.discriminator_clip = None
        self.discriminator_t5 = None
        self.generator_optimizer = None
        self.generator_loss_fn = None

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

    def test_node_metrics_from_images ( self, node, generated_images ):
        test_images = generated_images['on_test'] if 'on_test' in generated_images else {}
        train_images = generated_images['on_train'] if 'on_train' in generated_images else {}

        found_classes = list(test_images.values())
        filenames = list(test_images.keys())
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

        predictions = self.global_model.compute_zero_shot( filenames, federation_available_classes )
        labels = torch.tensor(list(candidate_labels.values()))
        metrics = self.global_model._compute_classification_metrics (predictions, ground_truth_classes)

        node_metrics = NodeMetric(phase=NodeMetric.Phase.TEST, task_count=1)
        node_metrics.define_metrics(self.global_model.defined_test_metrics, task_count=1)
        for metric in node_metrics.defined_metrics:
            node_metrics[0][metric] = metrics[metric]
        node_metrics['samples'] = len(generated_images)
        node_metrics['steps'] = 1
        print ( node_metrics )
        return node_metrics

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
            self.global_model.generate_low_memomy_footprint = self.generate_low_memomy_footprint
            self.global_model.start_diffusion( low_memory_footprint = self.global_model.generate_low_memomy_footprint)


        self.global_adapters = self.global_model.adapters
        self.global_adapters_modules = self.global_model.adapters_modules

        self.global_optimizers = self.create_global_model_optimizers()

        # Initialize generators if enabled
        if self.use_generator:
            self.initialize_generators()

        return self.global_model, self.generator_model
    
    def create_global_model_optimizers(self):
        optimizers = {}
        for module_name, module in self.global_adapters.items():
            if self.config.training.optimizer == "AdamW":
                optimizers[module_name] = AdamW(params=module.parameters(), lr=self.config.training.learning_rate)

        return optimizers

    def initialize_generators(self):
        """Initialize prompt generators (VAE or GAN) for server-side training."""
        logger.info(f"Initializing {self.generator_type} generator on server")

        if self.generator_type == 'vae':
            # VAE for conditioned prompt generation
            self.prompt_generator = ConditionedVAEGenerator(
                input_dim=512,      # Audio embeddings dimension from AST
                visual_dim=768,     # CLIP embeddings dimension
                hidden_dim=512,
                latent_dim=256,
                sequence_length=4
            ).to(self.device)

            self.generator_loss_fn = VAELoss()

            # Optimizer for VAE
            self.generator_optimizer = torch.optim.AdamW(
                self.prompt_generator.parameters(),
                lr=self.config.training.learning_rate * 0.1  # Lower LR for generator
            )

            logger.info("VAE generator initialized")

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
        self.baloon.deflate()
        generated_images = {}
        if self.generate_nodes_images_frequency > 0 and self.round % self.generate_nodes_images_frequency == 0:
            if self.optimize_memory_usage or self.round == 1:
                self.global_model.diffusion_model.to(self.diffusion_device)
            
            for client in self.clients:
                generated_images[client.id] = self.generate_images(client)
                node_metrics = self.test_node_metrics_from_images(client, generated_images=generated_images[client.id])
                if not self.no_wandb:
                    wandb_metrics = client.log_metrics(node_metrics, round=self.round)
                    wandb.log(wandb_metrics)

            if self.optimize_memory_usage:
                self.global_model.diffusion_model.to(torch.device('cpu'))

        if self.global_model_train and (self.config.feda2v.global_model_train_from_nodes_audio_embeddings or self.config.feda2v.global_model_train_from_nodes_adapters):
            self._move_to_gpu(self.device)

            if self.config.feda2v.global_model_train_from_nodes_audio_embeddings:
                loss = self.global_model_train_from_nodes_text_embeddings()
                self._move_to_cpu()
                print(f"\nGlobal Audio2Visual model trained from nodes embeddings with loss {loss:.4f}")

            if self.global_model_train_from_generator:
                self.train_generator_from_class_prompts()

            if self.feda2v.global_model_train_from_nodes_adapters:
                logger.info(f"\n=== Round {self.round}: Training Generator and Global Adapters ===")
                result = self.global_model_train_from_nodes_adapters_output()

                # Handle return values
                if isinstance(result, tuple) and len(result) == 2:
                    generator_loss, adapter_loss = result
                    print(f"\nGenerator Loss: {generator_loss:.4f}, Adapter Fine-tuning Loss: {adapter_loss:.4f}")

                    # Log to wandb
                    if not self.no_wandb:
                        self.data_log({
                            "server/generator_loss": generator_loss,
                            "server/adapter_finetuning_loss": adapter_loss,
                            "round": self.round
                        })
                else:
                    # Fallback for single value return
                    print(f"\nGlobal model trained with loss {result:.4f}")

                self._move_to_cpu()
        
        self.baloon.inflate()

        if self.generate_global_images_frequency > 0 and self.round % self.generate_global_images_frequency == 0:
            all_audio_embeddings = {}

            for client in self.clients:
                if hasattr(client, 'audio_embedding_store') and client.audio_embedding_store is not None:
                    all_audio_embeddings.update(client.audio_embedding_store)

        self.federation_test_metric(generated_images)

    def federation_test_metric ( self, generated_images):

        for node in self.clients:
            node_id = node.id
            if node_id not in generated_images:
                logger.warn(f"Node {node_id} not found in generated images")
                continue

            node_metrics = self.test_node_metrics_from_images(node, generated_images[node_id])

            

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
        """
        Train generator using adapter outputs from clients, then fine-tune global adapters.
        """
        self.global_model.to(self.device)

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
        Train the VAE/GAN generator to replicate adapter prompts from clients.

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

        for epoch in range(self.global_model_train_epochs):
            epoch_loss = 0.0

            for class_name, prompts_list in all_class_prompts.items():
                # Filter and stack valid prompts
                clip_prompts = [p['clip'] for p in prompts_list if p['clip'] is not None]
                audio_embs = [p['audio_embeddings'] for p in prompts_list if p['audio_embeddings'] is not None]

                if not clip_prompts or not audio_embs:
                    continue

                # Stack tensors
                clip_prompts = torch.stack(clip_prompts).to(self.device)
                audio_embs = torch.stack(audio_embs).to(self.device)

                if self.generator_type == 'vae':
                    # VAE Training
                    self.generator_optimizer.zero_grad()

                    # Forward pass through VAE
                    recon_prompts, mu, logvar = self.prompt_generator(
                        audio_embs,
                        visual_condition=clip_prompts
                    )

                    # Compute VAE loss
                    total_vae_loss, recon_loss, kl_loss, sim_loss = self.generator_loss_fn(
                        recon_prompts,
                        audio_embs,
                        mu,
                        logvar,
                        epoch
                    )

                    # Backward and optimize
                    total_vae_loss.backward()
                    self.generator_optimizer.step()

                    epoch_loss += total_vae_loss.item()

                    if num_batches % 5 == 0:
                        logger.info(f"  Class '{class_name}': VAE Loss={total_vae_loss.item():.4f} "
                                  f"(Recon={recon_loss.item():.4f}, KL={kl_loss.item():.4f}, Sim={sim_loss.item():.4f})")

                elif self.generator_type == 'gan':
                    # GAN Training
                    gan_loss = self.train_gan_step(clip_prompts, audio_embs, class_name)
                    epoch_loss += gan_loss

                num_batches += 1

            if len(all_class_prompts) > 0:
                avg_epoch_loss = epoch_loss / len(all_class_prompts)
                logger.info(f"Generator Epoch {epoch+1}/{self.global_model_train_epochs}: Loss = {avg_epoch_loss:.4f}")
                total_loss += avg_epoch_loss

        avg_loss = total_loss / self.global_model_train_epochs if self.global_model_train_epochs > 0 else 0.0
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
                    num_synthetic = 5  # Generate 5 synthetic samples per class
                    synthetic_audio_embs = self.prompt_generator.sample(
                        num_samples=num_synthetic,
                        visual_condition=mean_target_clip,
                        device=self.device
                    )
                # Combine real and synthetic
                combined_audio = torch.cat([mean_target_audio, synthetic_audio_embs], dim=0)
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
        """Generate images using the global Audio2Visual model."""
        self.global_model.diffusion_model.to(self.global_model.diffusion_model_device)

        prompt_embeds = audio_embeddings['t5'].to(self.global_model.diffusion_model_device).to(torch.bfloat16)
        pooled_prompt_embeds = audio_embeddings['clip'].to(self.global_model.diffusion_model_device).to(torch.bfloat16)

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
        
        prompt_embeds = text_embeddings['t5'].to(self.global_model.diffusion_dtype).to(self.diffusion_device)
        pooled_prompt_embeds = text_embeddings['clip'].to(self.global_model.diffusion_dtype).to(self.diffusion_device)


        if self.generate_low_memomy_footprint:
            imgs = self.global_model.diffusion_model(
                                        prompt_embeds=prompt_embeds,
                                        pooled_prompt_embeds=pooled_prompt_embeds,
                                        num_inference_steps=1,
                                        output_type="pt",
                                        ).images
        else:
            imgs = self.generate_single_images_from_diffusion(prompt_embeds, pooled_prompt_embeds)
        return imgs
    
    def generate_single_images_from_diffusion(self, prompt_embeds, pooled_prompt_embeds):
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
        saved_images = {}  
        for idx, img in enumerate(imgs):
            class_name = embeddings['class_name'][idx]
            img_save_path = os.path.join(self.images_output_dir, f"round_{self.round}_node_{client.id}_img_{class_name}_{idx}{suffix}.png")
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

        embeddings = client.get_audio_embeddings_for_generation(num_embeddings=5)

        on_test_imgs = self.generate_images_from_diffusion(embeddings)

        on_test_imgs_files = self.save_generated_images(on_test_imgs, client, embeddings)
        
        embeddings = client.get_audio_embeddings_for_generation(num_embeddings=2, from_train=True )

        on_train_imgs = self.generate_images_from_diffusion(embeddings)
        on_train_images_files = self.save_generated_images(on_train_imgs, client, embeddings, "_train")

        node_dataset = client.node_data.train_dataset.dataset
        text_embs = node_dataset.text_embs

        for node_class in node_dataset.active_classes.keys():
            embeddings = { 'clip': text_embs[node_class][self.diffusion_type]['pooled_prompt_embeds'],
                          't5': text_embs[node_class][self.diffusion_type]['prompt_embeds'],
                          'class_name': node_class }
                          
            from_text_imgs = self.generate_images_from_diffusion(embeddings)
            from_text_images_files = self.save_generated_images(from_text_imgs, client, embeddings, "_from_embs")
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
        return { 'on_train': on_train_images_files, 'on_test': on_test_imgs_files, 'from_embeddings': from_text_images_files }
            
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
                    # split='train',
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

        self.federation_available_classes = []
        self.federation_active_classes = []
        for node in self.clients:
            self.federation_available_classes.extend(node.node_data.train_dataset.dataset.available_classes)
            self.federation_available_classes.extend(node.node_data.test_dataset.dataset.available_classes)
            self.federation_active_classes.extend(node.node_data.train_dataset.dataset.available_classes)
            self.federation_active_classes.extend(node.node_data.test_dataset.dataset.available_classes)
            

        self.federation_available_classes = list((set(self.federation_available_classes)))
        self.federation_active_classes = list((set(self.federation_active_classes)))

        self.federation_available_classes.sort()
        self.federation_active_classes.sort()

        self.federation_available_classes = {v: i for i, v in enumerate(self.federation_available_classes)}
        self.federation_active_classes = {v: i for i, v in enumerate(self.federation_active_classes)}

        return self.clients

    def set_clients(self, clientObj):
        return self.create_clients(clientObj)

    def define_metrics(self):
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
            if c.node_data.train_dataset == None:
                print(f"Client {c.id} train data is None")
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
            if c.node_data.test_dataset == None:
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
        if self.optimize_memory_usage:
            logger.debug(f"Server moving to GPU: {device}")
            self._move_to_device(device)

    def _move_to_cpu(self):
        if self.optimize_memory_usage:
            logger.debug(f"Server moving to CPU for memory optimization")
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