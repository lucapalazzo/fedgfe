from flcore.trainmodel.downstreamsinestesia import DownstreamSinestesia
import torch.nn as nn
import torch
import numpy as np
import wandb
import logging

from utils.node_metric import NodeMetric
from flcore.trainmodel.Audio2Visual_NoData.src.models.sinestesia import SinestesiaWithClassifier
from transformers import ASTFeatureExtractor

logger = logging.getLogger(__name__)

class DownstreamSinestesiaAdapters(DownstreamSinestesia):
    """
    Downstream task wrapper for Sinestesia audio-visual model.

    This class wraps the SinestesiaWithClassifier model to work with the
    federated learning framework, handling forward passes, loss computation,
    and metrics tracking.
    """

    def __init__(self,
                 args,
                 num_classes=10,
                 wandb_log=False,
                 device=None,
                 use_classifier_loss=True,
                 diffusion_type=None,
                 loss_weights=None,
                 torch_dtype=torch.float32,
                 enable_diffusion=False):
    
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

        self.adapters = {}
        self.projections = {}
        self.adapters_modules = None

        self.ast_transformer = self.get_ast_transformer()
        self.ast_feature_extractor = self.get_ast_feature_extractor()

        self.get_sinestesia_adapters()

    def parameters(self, recurse = True):
        return self.adapters_modules.parameters()        
    
    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        if self.adapters_modules is not None:
            return self.adapters_modules.named_parameters(prefix, recurse, remove_duplicate)
        return []

    def get_ast_transformer(self):
        return self.audio2image_model.ast_model
    
    def get_ast_feature_extractor(self):
        return self.audio2image_model.feature_extractor

    def get_sinestesia_adapters(self):

        a2i_model = self.get_audio2image_model_from_sinestesia()

        self.adapters_modules = nn.ModuleList()

        if self.diffusion_type == 'sd' or self.diffusion_type == 'flux':
            self.adapters['clip'] = torch.nn.Sequential()
            self.adapters['clip'].add_module ( "adapter_clip", a2i_model.clip_adapter)
            self.adapters['clip'].add_module ( "projecton_clip", a2i_model.clip_projection )
            self.adapters_modules.add_module ( "clip", self.adapters['clip'])

        if self.diffusion_type == 'flux':
            self.adapters['t5'] = torch.nn.Sequential()
            self.adapters['t5'].add_module ( "adapter_t5", a2i_model.t5_adapter)
            self.adapters['t5'].add_module (  "projection_t5", a2i_model.t5_projection )
            self.adapters_modules.add_module ( "t5", self.adapters['t5'])
        
    def train(self, mode: bool = True) -> None:
        """Set training mode."""
        super(DownstreamSinestesia, self).train(mode)
        if self.sinestesia_model is not None:
            self.sinestesia_model.train(mode)

    def eval(self, mode: bool = True) -> None:
        """Set evaluation mode."""
        super(DownstreamSinestesia, self).eval()
        if self.sinestesia_model is not None:
            self.sinestesia_model.eval()



    def define_metrics(self, metrics_path=None):
        """Define wandb metrics."""
        if self.wandb_log == False:
            return None

        super().define_metrics(metrics_path=metrics_path)
        defined_metrics = []
        path = "/" if metrics_path is None else "/" + metrics_path + "/"

        # Train metrics
        for metric in self.defined_train_metrics:
            metric_name = f"train{path}{self.task_name}_{metric}"
            self.defined_train_metrics[metric] = metric_name
            defined_metrics.append(metric_name)

        # Test metrics
        for metric in self.defined_test_metrics:
            metric_name = f"test{path}{self.task_name}_{metric}"
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
                baseline_pooled_prompt_embeds=None, audio_embedding=None):



        if audio_embedding is None:
            x = self.ast_feature_extractor(audio,sampling_rate=16000, 
                                           return_tensors="pt", 
                                           padding = True).input_values.to(self.ast_transformer.device, self.torch_dtype) 
            x = self.ast_transformer(x).last_hidden_state
        else:
            x = audio_embedding.to(self.ast_transformer.device, self.torch_dtype)
            
        output = {}

        output['audio_embeddings'] = x
        for model in self.adapters.keys():
            output[model] = self.adapters[model](x)
            # x = self.adapters[model](x)
            # output[model] = self.projections[model](x)

        text_loss = self.text_mse( output, target_prompt_embeds= img_target_prompt_embeds, target_pooled_prompt_embeds=img_target_pooled_prompt_embeds)

        output['text_loss'] = text_loss

        return output
    
    def text_mse(self, output, target_prompt_embeds, target_pooled_prompt_embeds=None,):
        
        for model in self.adapters.keys():
            if model == 'clip':
                pooled_prompt_embeds = output['clip']
            elif model == 't5':
                prompt_embeds = output['t5']

        if self.diffusion_type == 'flux' and target_pooled_prompt_embeds is not None:
            mse1 = self.mse(target_prompt_embeds, prompt_embeds)
            mse2 = self.mse(target_pooled_prompt_embeds, pooled_prompt_embeds)
            return (mse1 , mse2) 
        elif self.diffusion_type == 'sd' and target_prompt_embeds is not None:
            return (self.mse(target_prompt_embeds, pooled_prompt_embeds),0)
        else:
            logger.warn("No target prompt embeddings provided for text MSE computation.")

        return torch.tensor(0.0, device=self.device, requires_grad=True)

    def downstream_loss(self, outputs, labels=None, samples=None):
        """
        Compute combined loss from Sinestesia outputs.

        Args:
            outputs: Dictionary with model outputs from forward()
            labels: Ground truth labels (for classifier)
            samples: Additional sample information

        Returns:
            Combined loss tensor
        """
        total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        loss_components = {}

        # Image text loss (from audio2image)
        if outputs.get('img_text_loss') is not None:
            img_text_loss = outputs['img_text_loss']
            total_loss = total_loss + self.loss_weights['img_text'] * img_text_loss
            loss_components['img_text_loss'] = img_text_loss.item()

        # Image reconstruction loss
        if outputs.get('image_loss') is not None:
            image_loss = outputs['image_loss']
            total_loss = total_loss + self.loss_weights['image'] * image_loss
            loss_components['image_loss'] = image_loss.item()

        # Audio text loss (from image2audio)
        if outputs.get('audio_text_loss') is not None:
            audio_text_loss = outputs['audio_text_loss']
            total_loss = total_loss + self.loss_weights['audio_text'] * audio_text_loss
            loss_components['audio_text_loss'] = audio_text_loss.item()

        # Classification loss (if using classifier)
        if self.use_classifier_loss and outputs.get('logits') is not None and labels is not None:
            logits = outputs['logits']

            # Handle different label formats
            if isinstance(labels, dict) and 'labels' in labels:
                labels = labels['labels']

            # Ensure labels are on correct device and long type
            if isinstance(labels, torch.Tensor):
                labels = labels.to(self.device).long()

                # Handle multi-dimensional labels
                if len(labels.shape) > 1:
                    labels = labels[:, 0]  # Take first label if multi-task

                classifier_loss = self.classification_loss(logits, labels)
                total_loss = total_loss + self.loss_weights['classifier'] * classifier_loss
                loss_components['classifier_loss'] = classifier_loss.item()

        return total_loss, loss_components

    def loss(self, outputs, labels=None, samples=None):
        """
        Compute loss (wrapper for downstream_loss).

        Args:
            outputs: Model outputs
            labels: Ground truth labels
            samples: Additional sample information

        Returns:
            Loss tensor
        """
        total_loss, _ = self.downstream_loss(outputs, labels, samples)
        return total_loss

    def train_metrics(self, dataloader, metrics=None):
        """
        Compute training metrics on a dataloader.

        Args:
            dataloader: DataLoader with training data
            metrics: Optional existing metrics object

        Returns:
            NodeMetric object with computed metrics
        """
        train_num = 0
        total_loss = 0.0
        total_components = {
            'img_text_loss': 0.0,
            'image_loss': 0.0,
            'audio_text_loss': 0.0,
            'classifier_loss': 0.0
        }
        total_correct = 0

        self.eval()

        with torch.no_grad():
            for batch_data in dataloader:
                # Handle different data formats
                if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 2:
                    samples, labels = batch_data[0], batch_data[1]
                elif isinstance(batch_data, dict):
                    samples = batch_data.get('audio', batch_data.get('samples'))
                    labels = batch_data.get('label', batch_data.get('labels'))
                else:
                    continue

                # Move to device
                if isinstance(samples, torch.Tensor):
                    samples = samples.to(self.device)
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(self.device)

                # Forward pass
                outputs = self(samples)

                if outputs is None:
                    continue

                # Compute loss
                loss, components = self.downstream_loss(outputs, labels)

                total_loss += loss.item()
                for key, value in components.items():
                    if key in total_components:
                        total_components[key] += value

                # Compute accuracy if using classifier
                if self.use_classifier_loss and outputs.get('logits') is not None:
                    logits = outputs['logits']
                    if isinstance(labels, dict) and 'labels' in labels:
                        labels_for_acc = labels['labels']
                    else:
                        labels_for_acc = labels

                    if isinstance(labels_for_acc, torch.Tensor):
                        labels_for_acc = labels_for_acc.to(self.device).long()
                        if len(labels_for_acc.shape) > 1:
                            labels_for_acc = labels_for_acc[:, 0]

                        predictions = torch.argmax(logits, dim=1)
                        total_correct += (predictions == labels_for_acc).sum().item()

                train_num += 1

        # Average metrics
        if train_num > 0:
            avg_loss = total_loss / train_num
            avg_components = {k: v / train_num for k, v in total_components.items()}
            avg_accuracy = total_correct / (train_num * dataloader.batch_size) if self.use_classifier_loss else 0.0

            # Log to wandb
            if self.wandb_log:
                log_dict = {
                    self.defined_train_metrics['loss']: avg_loss,
                    "round": self.round
                }

                for key in ['img_text_loss', 'image_loss', 'audio_text_loss', 'classifier_loss']:
                    if key in self.defined_train_metrics and key in avg_components:
                        log_dict[self.defined_train_metrics[key]] = avg_components[key]

                if self.use_classifier_loss and 'accuracy' in self.defined_train_metrics:
                    log_dict[self.defined_train_metrics['accuracy']] = avg_accuracy

                wandb.log(log_dict)

        return avg_loss if train_num > 0 else 0.0

    def test_metrics(self, dataloader, metrics=None):
        """
        Compute test metrics on a dataloader.

        Args:
            dataloader: DataLoader with test data
            metrics: Optional existing metrics object

        Returns:
            Dictionary with test metrics
        """
        test_num = 0
        total_loss = 0.0
        total_correct = 0

        self.eval()

        with torch.no_grad():
            for batch_data in dataloader:
                # Handle different data formats
                if isinstance(batch_data, (tuple, list)) and len(batch_data) >= 2:
                    samples, labels = batch_data[0], batch_data[1]
                elif isinstance(batch_data, dict):
                    samples = batch_data.get('audio', batch_data.get('samples'))
                    labels = batch_data.get('label', batch_data.get('labels'))
                else:
                    continue

                # Move to device
                if isinstance(samples, torch.Tensor):
                    samples = samples.to(self.device)
                if isinstance(labels, torch.Tensor):
                    labels = labels.to(self.device)

                # Forward pass
                outputs = self(samples)

                if outputs is None:
                    continue

                # Compute loss
                loss, _ = self.downstream_loss(outputs, labels)
                total_loss += loss.item()

                # Compute accuracy if using classifier
                if self.use_classifier_loss and outputs.get('logits') is not None:
                    logits = outputs['logits']
                    if isinstance(labels, dict) and 'labels' in labels:
                        labels_for_acc = labels['labels']
                    else:
                        labels_for_acc = labels

                    if isinstance(labels_for_acc, torch.Tensor):
                        labels_for_acc = labels_for_acc.to(self.device).long()
                        if len(labels_for_acc.shape) > 1:
                            labels_for_acc = labels_for_acc[:, 0]

                        predictions = torch.argmax(logits, dim=1)
                        total_correct += (predictions == labels_for_acc).sum().item()

                test_num += 1

        # Average metrics
        test_metrics = {}
        if test_num > 0:
            test_metrics['loss'] = total_loss / test_num
            if self.use_classifier_loss:
                test_metrics['accuracy'] = total_correct / (test_num * dataloader.batch_size)

            # Log to wandb
            if self.wandb_log:
                log_dict = {
                    self.defined_test_metrics['loss']: test_metrics['loss'],
                    "round": self.round
                }

                if self.use_classifier_loss and 'accuracy' in self.defined_test_metrics:
                    log_dict[self.defined_test_metrics['accuracy']] = test_metrics['accuracy']

                wandb.log(log_dict)

        return test_metrics
    
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

        steps = 0
        samples_count = 0
    
        with torch.no_grad():
            for audio_data in dataloader:
                
                text_embeddings = audio_data.get('text_emb', None)
                target_prompt_embeds = None
                target_pooled_prompt_embeds = None

                if isinstance(text_embeddings, dict) and self.diffusion_type in text_embeddings:
                    if self.diffusion_type == 'sd':
                        target_prompt_embeds = text_embeddings['sd']
                    elif self.diffusion_type == 'flux':
                        target_prompt_embeds = text_embeddings['flux'].get('prompt_embeds', None)
                        target_pooled_prompt_embeds = text_embeddings['flux'].get('pooled_prompt_embeds', None)
                else:
                    raise ValueError("Text embeddings not found or invalid format in audio_data.")

                steps += 1
                if ( 'audio' in audio_data and isinstance(audio_data['audio'], torch.Tensor) ):
                    audio_data = audio_data['audio']

                samples_count += audio_data.shape[0]

                if isinstance(audio_data, torch.Tensor) and isinstance(self.audio2image_model.feature_extractor, ASTFeatureExtractor):
                    audio_data = audio_data.cpu().numpy()
                else:
                    audio_data = audio_data.to(self.device)

                output = self(audio_data,img_target_prompt_embeds=target_prompt_embeds,
                        img_target_pooled_prompt_embeds=target_pooled_prompt_embeds) 
                if output == None:
                    continue
                
                # if ( "text_loss" in output ) and ( "recon_loss" in output ) and ( "audio_loss" in output ):
                if ( "text_loss" in output ):
                    metrics[0]['text_loss'] += output['text_loss'][0].item()
                    metrics[0]['samples'] += audio_data.shape[0]
                    metrics[0]['steps'] += 1

        return metrics
    
     
