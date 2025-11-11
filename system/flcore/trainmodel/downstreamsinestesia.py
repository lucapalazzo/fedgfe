from flcore.trainmodel.downstream import Downstream
import torch.nn as nn
import torch
import numpy as np
import wandb

from utils.node_metric import NodeMetric
from flcore.trainmodel.Audio2Visual_NoData.src.models.sinestesia import SinestesiaWithClassifier
from flcore.trainmodel.Audio2Visual_NoData.src.models.audio2image import Audio2Image
from transformers import ASTFeatureExtractor, pipeline
from diffusers import StableDiffusionPipeline, LCMScheduler, FluxPipeline, CogVideoXPipeline

from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, precision_score, recall_score

class DownstreamSinestesia(Downstream):
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
        """
        Initialize DownstreamSinestesia.

        Args:
            sinestesia_model: Instance of SinestesiaWithClassifier or Sinestesia
            num_classes: Number of classes for classification (if using classifier)
            wandb_log: Whether to log to wandb
            device: Device to run on
            use_classifier_loss: Whether to include classifier loss
            loss_weights: Dictionary with weights for different losses
                         {'img_text': w1, 'image': w2, 'audio_text': w3, 'classifier': w4}
        """
        # Initialize without backbone (we'll use sinestesia_model directly)
        super(DownstreamSinestesia, self).__init__(
            backbone=None,
            wandb_log=wandb_log,
            device=device
        )
        self.device = device
        self.diffusion_model_device = torch.device("cuda:1") if torch.cuda.device_count() > 1 else torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device1 = self.device
        self.device2 = torch.device("cuda:1" if torch.cuda.device_count() > 1 else self.device)
        self.device3 = torch.device("cuda:2" if torch.cuda.device_count() > 2 else self.device)
        self.torch_dtype = torch_dtype

        self.task_name = "sinestesia"

        self.diffusion_type = args.diffusion_type if diffusion_type is None else diffusion_type
        self.enable_diffusion = enable_diffusion
        self.diffusion_model = None
        self.generate_low_memomy_footprint = False
        self.diffusion_dtype = torch.float32




        if self.diffusion_type not in ['sd', 'flux']:
            raise ValueError(f"Unsupported diffusion type: {self.diffusion_type}. Supported types are 'sd' and 'flux'.")
        
        if self.diffusion_type == 'flux':
            args.img_pipe_name = "black-forest-labs/FLUX.1-dev"
            args.img_lcm_lora_id = "latent-consistency/lcm-lora-flux-base"
            args.img_lcm_lora_id = "latent-consistency/lcm-lora-sdxl"



        self.sinestesia_model = SinestesiaWithClassifier (
            args.audio_model_name,
            args.image_model_name,
            args.img_pipe_name,
            args.img_lcm_lora_id,
            args.audio_pipe_name,
            args.diffusion_type,
            args.use_act_loss,
            self.device1,
            self.device2,
            self.device3,
            args.mode
        )

        self.num_classes = num_classes
        self.use_classifier_loss = use_classifier_loss

        # Loss weights - default to equal weights
        if loss_weights is None:
            self.loss_weights = {
                'img_text': 1.0,
                'image': 1.0,
                'audio_text': 1.0,
                'classifier': 1.0
            }
        else:
            self.loss_weights = loss_weights

        # Classification loss function
        self.classification_loss = nn.CrossEntropyLoss()

        self.zero_shot_model = pipeline(
            "zero-shot-image-classification",
            model="openai/clip-vit-base-patch32",
            device=self.device
        )

        # Define metrics
        self.defined_test_metrics = {
            'accuracy': None,
            'balanced_accuracy': None,
            'f1_score': None,
            'precision': None,
            'recall': None,
            'f1_score_weighted': None
        }

        self.defined_train_metrics = {"text_loss": None}
        # self.defined_train_metrics = {"loss": None, "text_loss": None, "img_text_loss": None,
        #                              "image_loss": None, "audio_text_loss": None}

        # if self.use_classifier_loss:
        #     self.defined_train_metrics["classifier_loss"] = None
        #     self.defined_train_metrics["accuracy"] = None
        #     self.defined_test_metrics["accuracy"] = None

        # self.init_metrics()

        max_lenght1 = 77
        if self.diffusion_type == 'flux':
            max_lenght2 = 17
        else:
            max_lenght2 = None
        use_act_loss=True
        self.mode = 'train_nodata'
        ast_hf_model_name = 'MIT/ast-finetuned-audioset-10-10-0.4593' 

        if self.sinestesia_model is not None and self.sinestesia_model.sinestesia.imagediffusion is None:
            self.audio2image_model = Audio2Image(max_lenght1, max_lenght2, ast_hf_model_name, self.diffusion_type, use_act_loss, self.torch_dtype, self.mode)
            self.sinestesia_model.to(self.device)
        elif self.sinestesia_model is not None:
            self.audio2image_model = self.get_audio2image_model_from_sinestesia()

    # @property
    # def audio2image_model(self):
    #     return self.get_audio2image_model_from_sinestesia()
    def to(self, device):
        self.audio2image_model.to(device)
        return self
    
    def get_audio2image_model_from_sinestesia(self):
        if self.sinestesia_model and self.sinestesia_model.sinestesia.imagediffusion is not None:
            self.audio2image_model = self.sinestesia_model.sinestesia.imagediffusion.audio2image
        return self.audio2image_model
    
    def get_audio2image_model(self):
        return self.audio2image_model

    # def init_metrics(self):
    #     """Initialize metrics storage."""
    #     self.train_metrics_storage = {metric: [] for metric in self.defined_train_metrics}
    #     self.test_metrics_storage = {metric: [] for metric in self.defined_test_metrics}
    #     return self.train_metrics_storage

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

    def parameters(self, recurse=True):
        """Return model parameters."""
        if self.sinestesia_model is not None:
            return self.sinestesia_model.parameters(recurse)
        return []

    def named_parameters(self, prefix='', recurse=True, remove_duplicate=True):
        """Return named model parameters."""
        if self.sinestesia_model is not None:
            return self.sinestesia_model.named_parameters(prefix, recurse, remove_duplicate)
        return []

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
                baseline_pooled_prompt_embeds=None):
    
        return self.sinestesia_model(
            audio=audio,
            target_image=target_image,
            img_target_prompt_embeds=img_target_prompt_embeds,
            img_target_pooled_prompt_embeds=img_target_pooled_prompt_embeds,
            audio_target_prompt_embeds=audio_target_prompt_embeds,
            baseline_prompt_embeds=baseline_prompt_embeds,
            baseline_pooled_prompt_embeds=baseline_pooled_prompt_embeds
        )


    def start_diffusion(self, low_memory_footprint = False):

        if low_memory_footprint:
            self.diffusion_dtype = torch.bfloat16

        if self.enable_diffusion:
            img_lcm_lora_id="latent-consistency/lcm-lora-sdv1-5"

            if self.diffusion_type == 'sd':
                img_pipe_name="runwayml/stable-diffusion-v1-5",
                self.diffusion_model = StableDiffusionPipeline.from_pretrained(img_pipe_name, torch_dtype=self.diffusion_dtype).to(self.diffusion_model_device)
            elif self.diffusion_type == 'flux':
                img_pipe_name="black-forest-labs/FLUX.1-schnell"
                img_lcm_lora_id="strangerzonehf/Flux-Midjourney-Mix2-LoRA"
                self.diffusion_model = FluxPipeline.from_pretrained(img_pipe_name, torch_dtype=self.diffusion_dtype)

            print( f"Started diffusion model {self.diffusion_type}")
            # self.diffusion_model.to(torch.device('cpu'))
            


    def generate_image(self, prompt_embeds, pooled_prompt_embeds=None):
        self.img_pipe.set_progress_bar_config(disable=True)
        negative_prompt = ''
        if self.diffusion_type == 'sd':
            # Store unconditional prompt
            negative_prompt = self.img_pipe.tokenizer(negative_prompt,
                                                        padding="max_length",
                                                        max_length=77,
                                                        truncation=True,
                                                        return_tensors="pt",
                                                        )['input_ids'].to(self.device1)

            negative_prompt_embeds = self.img_pipe.text_encoder(negative_prompt).last_hidden_state

            with torch.no_grad():
                if negative_prompt_embeds.shape[0] != prompt_embeds.shape[0]:
                    negative_prompt_embeds = negative_prompt_embeds.expand(prompt_embeds.shape[0], -1, -1)
                images = self.img_pipe(
                                        prompt_embeds= prompt_embeds,
                                        negative_prompt_embeds=negative_prompt_embeds,
                                        num_inference_steps=4,
                                        output_type="pt",
                                        guidance_scale=1
                                        ).images
            
        elif self.diffusion_type == 'flux':
            assert pooled_prompt_embeds is not None, "pooled_prompt_embeds must be provided for Flux"
            prompt_embeds = prompt_embeds.to(self.device2)
            pooled_prompt_embeds = pooled_prompt_embeds.to(self.device2)
            with torch.no_grad():
                images = self.img_pipe(
                                        prompt_embeds= prompt_embeds,
                                        pooled_prompt_embeds=pooled_prompt_embeds,
                                        num_inference_steps=1,
                                        output_type="pt",
                                        ).images
                
        elif self.diffusion_type == 'cogx':
            with torch.no_grad():
                images = self.img_pipe(
                            prompt_embeds=prompt_embeds,
                            num_videos_per_prompt=1,
                            num_inference_steps=50,
                            num_frames=20,
                            guidance_scale=6
                        ).frames[0]
            
        return images

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
    
    def compute_zero_shot(self, imgs_out_path, classes):
        prediction = self.zero_shot_model(imgs_out_path, candidate_labels=classes.keys()) #classes is a list of all the textual labels [train, airplane, ...]
        preds = []

        # result = [
        #     {"score": score, "label": candidate_label}
        #     for score, candidate_label in sorted(zip(prediction, classes.keys()), key=lambda x: -x[0])
        # ]
        for i, pred in enumerate(prediction):
            predicted_label = classes[pred[0]['label']] #classes_dict è un dizionario che puoi creare che mappa l'id testuale in id numerico
            preds.append(predicted_label)

        preds = torch.tensor(preds)
        return preds

    def _compute_classification_metrics(self, preds, labels):
        preds = preds.numpy()
        labels = labels.cpu().numpy() # qui labels sono tutte le annotazioni corrette dei sample 

        accuracy = accuracy_score(labels, preds)
        balanced_acc = balanced_accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro')
        precision = precision_score(labels, preds, average='macro', zero_division=0)
        recall = recall_score(labels, preds, average='macro', zero_division=0)
        f1w = f1_score(labels, preds, average='weighted', zero_division=0)

        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'f1_score_weighted': f1w
        }

#1) calcola le pred passando immagini e lista classi a compute_zero_shot
#2) passa predizioni e ground truth a _compute_classification_metrics

    def test_metrics(self, dataloader, test_images = None):
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
        metrics[0]['loss'] = 0 

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

                if isinstance(audio_data, torch.Tensor) and isinstance(self.sinestesia_model.sinestesia.imagediffusion.audio2image.feature_extractor, ASTFeatureExtractor):
                    audio_data = audio_data.cpu().numpy()
                else:
                    audio_data = audio_data.to(self.device)

                if audio2image_only:
                    output = self.sinestesia_model.sinestesia.imagediffusion.audio2image(
                        audio=audio_data,
                        target_prompt_embeds=target_prompt_embeds,
                        target_pooled_prompt_embeds=target_pooled_prompt_embeds
                    )
                else:   
                    output = self(  audio_data, target_image=target_image,
                                img_target_prompt_embeds=img_target_prompt_embeds,
                                img_target_pooled_prompt_embeds=img_target_pooled_prompt_embeds,
                                audio_target_prompt_embeds=audio_target_prompt_embeds,
                                baseline_prompt_embeds=baseline_prompt_embeds,
                                baseline_pooled_prompt_embeds=baseline_pooled_prompt_embeds)

                if output == None:
                    continue
                
                # if ( "text_loss" in output ) and ( "recon_loss" in output ) and ( "audio_loss" in output ):
                if ( "text_loss" in output ):
                    metrics[0]['text_loss'] += output['text_loss'][0].item()
                    metrics[0]['samples'] += audio_data.shape[0]
                    metrics[0]['steps'] += 1

        return metrics
    
     
