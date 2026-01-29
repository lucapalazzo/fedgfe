from flcore.trainmodel.downstreamsinestesia import DownstreamSinestesia
import torch.nn as nn
import torch
import numpy as np
import wandb
import logging
from tqdm import tqdm

from utils.node_metric import NodeMetric
from flcore.trainmodel.Audio2Visual_NoData.src.models.sinestesia import SinestesiaWithClassifier
from transformers import ASTFeatureExtractor, ASTModel
# Use adapter with INPUT BatchNorm for heterogeneous batch stability
from flcore.trainmodel.Audio2Visual_NoData.src.models.projection import Adapter, ASTAdapter
from flcore.trainmodel.Audio2Visual_NoData.src.models.multi_head_attention import MAPBlock

logger = logging.getLogger(__name__)

class DownstreamSinestesiaAdapters(DownstreamSinestesia):
    def __init__(self,
                 args,
                 num_classes=10,
                 wandb_log=False,
                 device=None,
                 use_classifier_loss=True,
                 diffusion_type=None,
                 loss_weights=None,
                 torch_dtype=torch.float32,
                 enable_diffusion=False,
                 use_cls_token_only=False,
                 adapter_dropout=0.1,
                 ast_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
                 use_pretrained_generators=False,
                 generators_dict=None,
                 init_ast_model=True):

        # Initialize without backbone (we'll use sinestesia_model directly)
        super(DownstreamSinestesiaAdapters, self).__init__(
            args,
            num_classes=num_classes,
            wandb_log=wandb_log,
            device=device,
            diffusion_type=diffusion_type,
            use_classifier_loss=use_classifier_loss,
            loss_weights=loss_weights
        )

        self.torch_dtype = torch_dtype
        self.mse = nn.MSELoss().to(self.torch_dtype)

        # Callback function for caching embeddings (set by client)
        self.cache_callback = None

        self.use_map_adapters = False

        self.init_ast_model = init_ast_model

        self.adapters = {}
        self.adapters_modules = None
        self.adapters_losses = {}

        if adapter_dropout is None:
            adapter_dropout = 0.0

        # Dropout rate for adapters to prevent overfitting
        self.adapter_dropout = adapter_dropout
        self.adapter_dropout_layer = nn.Dropout(p=adapter_dropout)

        # Configuration for using only CLS token from AST output
        self.use_cls_token_only = use_cls_token_only

        if self.diffusion_type == 'flux':
            self.t5_n_latents = 17
            self.clip_n_latents = 1
        else:
            self.clip_n_latents = 77
            self.t5_n_latents = None

        # Store the flag for later use
        self.use_pretrained_generators = use_pretrained_generators

        # Store generators dictionary: {class_id: generator_model}
        # This will be populated by the server via set_generators() method
        self.generators_dict = generators_dict if generators_dict is not None else {}

        # Initialize AST model and feature extractor only if NOT using pretrained generators
        # When using pretrained generators, we generate synthetic audio embeddings directly
        # without needing to process real audio through AST
        self.ast_feature_extractor = None
        self.ast_model = None

        if self.init_ast_model:
            logger.info(f"Initializing AST model: {ast_model_name}")
            self.ast_feature_extractor = ASTFeatureExtractor.from_pretrained(ast_model_name)
            self.ast_model = ASTModel.from_pretrained(ast_model_name)
            logger.info("AST model initialized successfully")
        else:
            logger.info("="*80)
            logger.info("SKIPPING AST INITIALIZATION - Using pretrained generators mode or disabled by constructor")
            logger.info("="*80)

        self.setup_adapters()

        # self.get_sinestesia_adapters()
        self.setup_losses()

    def setup_losses(self):
        # Use reduction='none' to get per-sample losses, then we can average them properly
        # This helps with batch stability when different classes have very different embeddings
        for adapter_name in self.adapters.keys():
            self.adapters_losses[adapter_name] = nn.MSELoss(reduction='mean')

    def setup_adapters(self):
        self.adapters_modules = nn.ModuleList()

        if self.diffusion_type == 'sd' or self.diffusion_type == 'flux':
            self.adapters['clip'] = torch.nn.Sequential()
            if self.use_map_adapters:
                self.adapters['clip'].add_module ( "adapter_clip", ASTAdapter(embed_dim=768, n_latents=64, n_heads=8))
            else:
                self.adapters['clip'].add_module ( "adapter_clip", Adapter(input_dim=768, hidden_dims=[1024,2048], output_dim=768))
            self.adapters['clip'].add_module ( "projection_clip", MAPBlock(n_latents=self.clip_n_latents, embed_dim=768, n_heads=8))
            self.adapters_modules.add_module ( "clip", self.adapters['clip'])

        if self.diffusion_type == 'flux':
            self.adapters['t5'] = torch.nn.Sequential()
            if self.use_map_adapters:
                self.adapters['t5'].add_module ( "adapter_t5", ASTAdapter(embed_dim=768, n_latents=64, n_heads=8, output_dim=4096))
            else:
                self.adapters['t5'].add_module ( "adapter_t5", Adapter(input_dim=768, hidden_dims=[1024,2048,2048], output_dim=4096))
            self.adapters['t5'].add_module (  "projection_t5", MAPBlock(n_latents=self.t5_n_latents, embed_dim=4096, n_heads=8))
            self.adapters_modules.add_module ( "t5", self.adapters['t5'])
        
        return self.adapters

    def to(self, device):
        # Only move AST model to device if it was initialized
        if self.ast_model is not None:
            self.ast_model.to(device)
        self.adapters_modules.to(device)
        self.device = device
        return self

    def parameters(self, recurse = True):
        return self.adapters_modules.parameters()        
    
    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        if self.adapters_modules is not None:
            return self.adapters_modules.named_parameters(prefix, recurse, remove_duplicate)
        return []

    def get_ast_transformer(self):
        return self.ast_model
    
    def get_ast_feature_extractor(self):
        return self.ast_feature_extractor

    def set_generators(self, generators_dict):
        """
        Set the dictionary of class-specific generators.
        Called by the server to pass pretrained generators to the client model.

        Args:
            generators_dict (dict): Dictionary mapping class_id (int) -> generator model
                                   Example: {0: generator_class0, 1: generator_class1, ...}
        """
        self.generators_dict = generators_dict
        logger.info(f"Set {len(generators_dict)} class-specific generators")

        # Move generators to the correct device
        if self.device is not None:
            for class_id, generator in self.generators_dict.items():
                if generator is not None:
                    generator.to(self.device)
                    generator.eval()  # Set to eval mode for inference

    # def get_sinestesia_adapters(self):

    #     a2i_model = self.get_audio2image_model_from_sinestesia()

    #     self.adapters_modules = nn.ModuleList()

    #     if self.diffusion_type == 'sd' or self.diffusion_type == 'flux':
    #         self.adapters['clip'] = torch.nn.Sequential()
    #         self.adapters['clip'].add_module ( "adapter_clip", a2i_model.clip_adapter)
    #         self.adapters['clip'].add_module ( "projection_clip", a2i_model.clip_projection )
    #         self.adapters_modules.add_module ( "clip", self.adapters['clip'])

    #     if self.diffusion_type == 'flux':
    #         self.adapters['t5'] = torch.nn.Sequential()
    #         self.adapters['t5'].add_module ( "adapter_t5", a2i_model.t5_adapter)
    #         self.adapters['t5'].add_module (  "projection_t5", a2i_model.t5_projection )
    #         self.adapters_modules.add_module ( "t5", self.adapters['t5'])
        
    #     return self.adapters
        
    def train(self, mode: bool = True) -> None:
        for adapter in self.adapters.values():
            adapter.train(mode)
        self.adapter_dropout_layer.train()

    def eval(self):
        for adapter in self.adapters.values():
            adapter.eval()
        self.adapter_dropout_layer.eval()



    def define_metrics(self, metrics_path=None, train_splits = None, test_splits = None ):
        # if self.wandb_log == False:
        #     return None

        super().define_metrics(metrics_path=metrics_path)
        defined_metrics = []
        path = "/" if metrics_path is None else "/" + metrics_path 

        # Train metrics
        if train_splits is not None and type(train_splits) == list:
            for split in train_splits:
                for metric in self.defined_train_metrics:
                    metric_name = f"train{path}{metric}_on_{split}"
                    self.defined_train_metrics[metric] = metric_name
                    defined_metrics.append(metric_name)
        else:
            for metric in self.defined_train_metrics:
                metric_name = f"train{path}{metric}"
                # metric_name = f"train{path}{self.task_name}_{metric}"
                self.defined_train_metrics[metric] = metric_name
                defined_metrics.append(metric_name)

        # Test metrics
        if test_splits is not None and type(test_splits) == list:
            for split in test_splits:
                for metric in self.defined_test_metrics:
                    metric_name = f"test{path}{metric}_on_{split}"
                    self.defined_test_metrics[metric] = metric_name
                    defined_metrics.append(metric_name)

        else:
            for metric in self.defined_test_metrics:
                # metric_name = f"test{path}{self.task_name}_{metric}"
                metric_name = f"test{path}{metric}"
                self.defined_test_metrics[metric] = metric_name
                defined_metrics.append(metric_name)

        for metric in defined_metrics:
            wandb.define_metric(metric, step_metric="round")

    def forward(self,
                audio,
                target_image=None,
                img_target_prompt_embeds=None,
                img_target_pooled_prompt_embeds=None,
                audio_target_prompt_embeds=None,
                baseline_prompt_embeds=None,
                baseline_pooled_prompt_embeds=None,
                audio_embedding=None,
                labels=None,
                class_names=None,
                return_loss=True):

        if audio_embedding is not None:
            x = audio_embedding.to(self.ast_model.device, self.torch_dtype)
        else:
            x = self.ast_feature_extractor(audio,sampling_rate=16000,
                                           return_tensors="pt",
                                           padding = True).input_values.to(self.ast_model.device, self.torch_dtype)
            x = self.ast_model(x).last_hidden_state

        if self.use_cls_token_only:
            x_cls = x[:, 0:1, :]  # Shape: (batch_size, 1, hidden_dim) - keep dimension for compatibility
            logger.debug(f"Using CLS token only. Original shape: {x.shape}, CLS shape: {x_cls.shape}")
            x = x_cls

        output = {}
        losses = {}
        targets = { 'clip': img_target_pooled_prompt_embeds, 't5': img_target_prompt_embeds}
        output['audio_embeddings'] = x.detach().cpu()
        for adapter_name, adapter in self.adapters.items():
            output[adapter_name] = adapter(x)

        if return_loss: 
            text_loss = self.text_mse( output, targets)
            output['text_loss'] = text_loss

        return output
    
    def text_mse(self, output, targets):

        losses = {} 
        if self.diffusion_type == 'flux':
            mse_t5 = self.mse(output['t5'], targets['t5'])
            losses['t5'] = mse_t5

        if self.diffusion_type == 'sd' or self.diffusion_type == 'flux':
            mse_clip = self.mse(output['clip'], targets['clip'])
            losses['clip'] = mse_clip
        
        if len(losses) == 0:
            logger.warn("No target prompt embeddings provided for text MSE computation.")

        return losses

    def train_metrics( self, dataloader, audio2image_only=False, target_image=None,
                img_target_prompt_embeds=None,
                img_target_pooled_prompt_embeds=None,
                audio_target_prompt_embeds=None,
                baseline_prompt_embeds=None,
                baseline_pooled_prompt_embeds=None ):
        train_num = 0
        train_loss = 0.0
    
        metrics = NodeMetric(phase=NodeMetric.Phase.TRAIN)
        metrics.define_metrics(self.defined_train_metrics)
        metrics.task_name = self.task_name
        metrics.task_type = NodeMetric.TaskType.UNKNOWN
        metrics[0]['text_loss'] = 0.0
        metrics[0]['clip_loss'] = 0.0
        metrics[0]['t5_loss'] = 0.0

        steps = 0
        samples_count = 0

        self.adapters_modules.eval()

        with torch.no_grad():
            split = dataloader.dataset.split if hasattr(dataloader.dataset, 'split') else 'train'
            pbar = tqdm(dataloader, desc=f"Computing train metrics on {split}", unit="batch")
            for samples in pbar:

                text_embeddings = samples.get('text_emb', None)
                target_prompt_embeds = None
                target_pooled_prompt_embeds = None

                if isinstance(text_embeddings, dict) and self.diffusion_type in text_embeddings:
                    if self.diffusion_type == 'sd':
                        target_prompt_embeds = text_embeddings['sd'].to(self.device)
                    elif self.diffusion_type == 'flux':
                        target_prompt_embeds = text_embeddings['flux'].get('prompt_embeds', None).to(self.device)
                        target_pooled_prompt_embeds = text_embeddings['flux'].get('pooled_prompt_embeds', None).to(self.device)
                else:
                    raise ValueError("Text embeddings not found or invalid format in audio_data.")

                steps += 1
                audio_embedding = samples.get('audio_emb', None)
                if ( 'audio' in samples and isinstance(samples['audio'], torch.Tensor) ):
                    # print ( f"Filenames {audio_data['audio_filename']} classes {audio_data['class_name']}")
                    audio_data = samples['audio']

                samples_count += audio_data.shape[0]

                if isinstance(audio_data, torch.Tensor) and isinstance(self.ast_feature_extractor, ASTFeatureExtractor):
                    audio_data = audio_data.cpu().numpy()
                else:
                    audio_data = audio_data.to(self.device)
                if audio_embedding is not None:
                    audio_embedding = audio_embedding.to(self.device)
                
                class_names = samples.get('class_name', None)

                output = self(audio_data,img_target_prompt_embeds=target_prompt_embeds,
                        img_target_pooled_prompt_embeds=target_pooled_prompt_embeds,audio_embedding=audio_embedding, class_names=class_names)
                if output == None:
                    continue

                # Call cache callback if provided (set by client)
                if hasattr(self, 'cache_callback') and self.cache_callback is not None:
                    self.cache_callback(samples, output, dataloader=dataloader)

                # if ( "text_loss" in output ) and ( "recon_loss" in output ) and ( "audio_loss" in output ):
                if ( "text_loss" in output ):
                    text_loss = output['text_loss']
                    text_loss_avg = sum(text_loss.values()).item()/ len(text_loss)
                    metrics[0]['text_loss'] += text_loss_avg
                    if self.diffusion_type in ['sd', 'flux']:
                        if "clip" in text_loss:
                            metrics[0]['clip_loss'] += text_loss['clip'].item()
                    if self.diffusion_type == 'flux':
                        if "t5" in text_loss:
                            metrics[0]['t5_loss'] += text_loss['t5'].item()
                    metrics[0]['samples'] += audio_data.shape[0]
                    metrics[0]['steps'] += 1

                    # Update progress bar with current metrics
                    if metrics[0]['steps'] > 0:
                        current_avg_loss = metrics[0]['text_loss'] / metrics[0]['steps']
                        pbar.set_postfix({'avg_text_loss': f"{current_avg_loss:.4f}",
                                          'clip_loss': f"{metrics[0]['clip_loss'] / metrics[0]['steps']:.4f}" if self.diffusion_type in ['sd','flux'] else 'N/A',
                                          't5_loss': f"{metrics[0]['t5_loss'] / metrics[0]['steps']:.4f}" if self.diffusion_type == 'flux' else 'N/A'})

        return metrics
    
     
