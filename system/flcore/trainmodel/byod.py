import os
from torch import nn
import torch
from flcore.trainmodel.pretexttask import PretextTask
from typing import Iterator

from timm.models.vision_transformer import VisionTransformer
from transformers import ViTModel
from torch.nn import functional as F
import wandb
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset


class BYOD(PretextTask):
    """
    BYOD (Bring Your Own Data) Pretext Task
    
    This task allows users to bring their own external data for self-supervised learning.
    It extends PretextTask and can be used for domain adaptation, data augmentation,
    or leveraging external datasets for better representations.
    """
    pretext_task_name = "byod"
    task_name = pretext_task_name
    defined_test_metrics = {"similarity_score": None, "data_consistency": None}
    defined_train_metrics = {"loss": None, "alignment_loss": None}
    task_weight_decay = 1e-5  # Default weight decay for optimizer
    task_learning_rate = 1e-4  # Default learning rate for optimizer
    
    def __init__(self, backbone=None, input_dim=768, output_dim=768, debug_images=False, img_size=224, 
                 patch_size=-1, patch_count=-1, external_data_path=None, alignment_weight=0.5, 
                 projection_dim=128, similarity_threshold=0.7, augment_strength=1.0, cls_token_only=True):
        super(BYOD, self).__init__(backbone=backbone, input_dim=input_dim, output_dim=output_dim, 
                                   debug_images=debug_images, img_size=img_size, patch_size=patch_size, 
                                   patch_count=patch_count)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = BYOD.pretext_task_name
        self.external_data_path = external_data_path
        self.alignment_weight = alignment_weight
        self.similarity_threshold = similarity_threshold
        self.cls_token_only = cls_token_only
        
        # Projection head for feature alignment between internal and external data
        self.projection_head = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.output_dim, projection_dim),
            nn.LayerNorm(projection_dim)
        ).to(self.device)
        
        # Alignment head for domain adaptation
        self.alignment_head = nn.Sequential(
            nn.Linear(projection_dim, projection_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim // 2, projection_dim),
            nn.Dropout(0.1)
        ).to(self.device)
        
        # External data components
        self.external_data_loader = None
        self.external_features_cache = []
        
        # Data augmentation for consistency
        self.transform = self._get_augmentation_pipeline(strength=augment_strength)
        
        # Load external data if path provided
        if external_data_path and os.path.exists(external_data_path):
            self._load_external_data()

    def _get_augmentation_pipeline(self, strength=1.0):
        """Create data augmentation pipeline for BYOD consistency"""
        s = strength
        return transforms.Compose([
            transforms.RandomResizedCrop(size=self.img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4*s, contrast=0.4*s, saturation=0.4*s, hue=0.1*s)
            ], p=0.6),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=int(0.1 * self.img_size) // 2 * 2 + 1, sigma=(0.1, 2.0))
            ], p=0.2),
        ])
    
    def _load_external_data(self):
        """Load external data for BYOD training"""
        try:
            # This is a placeholder - in real implementation, you would load actual external data
            # based on the format (images, features, etc.)
            print(f"Loading external data from: {self.external_data_path}")
            
            # For now, create dummy external data
            # In practice, this would load real external datasets
            external_data = torch.randn(100, 3, self.img_size, self.img_size)
            external_dataset = TensorDataset(external_data)
            self.external_data_loader = DataLoader(external_dataset, batch_size=32, shuffle=True)
            
            print(f"Loaded external dataset with {len(external_dataset)} samples")
            
        except Exception as e:
            print(f"Warning: Could not load external data from {self.external_data_path}: {e}")
            self.external_data_loader = None
    
    @staticmethod
    def define_metrics(metrics_path=None):
        pretext_task_name = BYOD.pretext_task_name

        path = "/" if metrics_path is None else "/"+metrics_path+"/"
        metrics = []

        for metric in BYOD.defined_test_metrics:
            defined_metric = f"test{path}{BYOD.task_name}_{metric}"
            BYOD.defined_test_metrics[metric] = defined_metric
            metrics.append(defined_metric)

        for metric in BYOD.defined_train_metrics:
            defined_metric = f"train{path}{BYOD.task_name}_{metric}"
            BYOD.defined_train_metrics[metric] = defined_metric
            metrics.append(defined_metric)

        for metric in metrics:
           wandb.define_metric(metric, step_metric="round")

        return metrics
    
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        modules = nn.ModuleList()
        modules.add_module("projection_head", self.projection_head)
        modules.add_module("alignment_head", self.alignment_head)
        parameters = modules.parameters(recurse)
        return parameters
    
    def preprocess_sample(self, x):
        """Generate augmented views and prepare for BYOD training"""
        batch_size = x.shape[0]
        
        # Create two augmented views of the internal data
        augmented_views = []
        for img in x:
            aug1 = self.transform(img)
            aug2 = self.transform(img)
            augmented_views.extend([aug1, aug2])
        
        # Stack augmented views
        augmented = torch.stack(augmented_views)
        augmented = augmented.view(batch_size, 2, x.shape[1], x.shape[2], x.shape[3])
        
        return augmented
    
    def get_external_batch(self, batch_size=None):
        """Get a batch of external data"""
        if self.external_data_loader is None:
            return None
        
        try:
            external_batch = next(iter(self.external_data_loader))[0]
            if batch_size is not None:
                external_batch = external_batch[:batch_size]
            return external_batch.to(self.device)
        except:
            return None
        
    def forward(self, samples):
        """
        Forward pass for BYOD
        samples: Input tensor of shape [batch_size, channels, height, width]
        """
        batch_size = samples.shape[0]
        
        # Get augmented views of internal data
        augmented = self.preprocess_sample(samples)  # [batch_size, 2, C, H, W]
        
        # Reshape to process all augmentations in a single batch
        augmented_flat = augmented.view(-1, augmented.shape[2], augmented.shape[3], augmented.shape[4])
        
        # Pass through the backbone
        if self.backbone is not None:
            features = self.backbone(augmented_flat)
            
            # Handle different backbone types
            if isinstance(self.backbone, VisionTransformer):
                features = features[:, 0, :]  # Use CLS token
            elif isinstance(self.backbone, ViTModel):
                if self.cls_token_only:
                    features = features.last_hidden_state[:, 0, :]  # Use CLS token
                else:
                    features = features.last_hidden_state[:, 1:, :]  # Use patch embeddings
                    features = features.mean(dim=1)  # Average over patches
        else:
            features = augmented_flat.mean(dim=[2, 3])  # Global average pooling if no backbone
            
        # Project features to alignment space
        projections = self.projection_head(features)
        projections = F.normalize(projections, dim=1)
        
        # Reshape back to [batch_size, 2, projection_dim]
        projections = projections.view(batch_size, 2, -1)
        
        # Get external data features if available
        external_features = None
        if self.external_data_loader is not None:
            external_batch = self.get_external_batch(batch_size)
            if external_batch is not None:
                with torch.no_grad():
                    if self.backbone is not None:
                        ext_features = self.backbone(external_batch)
                        if isinstance(self.backbone, VisionTransformer):
                            ext_features = ext_features[:, 0, :]
                        elif isinstance(self.backbone, ViTModel):
                            if self.cls_token_only:
                                ext_features = ext_features.last_hidden_state[:, 0, :]
                            else:
                                ext_features = ext_features.last_hidden_state[:, 1:, :].mean(dim=1)
                    else:
                        ext_features = external_batch.mean(dim=[2, 3])
                    
                    external_features = F.normalize(self.projection_head(ext_features), dim=1)
        
        return {
            'projections': projections,
            'external_features': external_features,
            'internal_features': projections.view(-1, projections.shape[-1])
        }
    
    def loss(self, outputs, y=None):
        """
        BYOD loss combining consistency loss and alignment loss
        """
        projections = outputs['projections']
        external_features = outputs.get('external_features', None)
        internal_features = outputs['internal_features']
        
        batch_size = projections.shape[0]
        
        # Consistency loss (similar to SimCLR)
        z1, z2 = projections[:, 0], projections[:, 1]  # [batch_size, projection_dim]
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        z = torch.cat([z1, z2], dim=0)  # [2B, D]
        B = z1.size(0)
        
        # Consistency similarity matrix
        sim = torch.matmul(z, z.T) / 0.1  # temperature = 0.1
        
        # Mask diagonal
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float('-inf'))
        
        # Positive pairs: first B -> second B, second B -> first B
        targets = torch.arange(B, 2 * B, device=z.device)
        targets = torch.cat([targets, torch.arange(0, B, device=z.device)], dim=0)
        
        consistency_loss = F.cross_entropy(sim, targets)
        
        # Alignment loss with external data
        alignment_loss = torch.tensor(0.0, device=z.device)
        if external_features is not None:
            # Encourage similarity between internal and external features
            internal_norm = F.normalize(internal_features, dim=1)
            external_norm = F.normalize(external_features, dim=1)
            
            # Calculate cross-modal similarity
            cross_sim = torch.matmul(internal_norm, external_norm.T)
            
            # Alignment objective: maximize similarity above threshold
            alignment_target = torch.clamp(cross_sim, min=self.similarity_threshold)
            alignment_loss = F.mse_loss(cross_sim, alignment_target)
        
        # Combined loss
        total_loss = consistency_loss + self.alignment_weight * alignment_loss
        
        return total_loss
    
    def test_metrics(self, predictions, y=None, samples=None, metrics=None):
        """Calculate metrics for evaluation"""
        if isinstance(predictions, dict):
            outputs = predictions
        else:
            # For backward compatibility
            outputs = {'projections': predictions}
        
        loss_value = self.loss(outputs).item()
        
        if metrics is None:
            return None
        
        # Calculate similarity score between views
        projections = outputs['projections']
        z1, z2 = projections[:, 0], projections[:, 1]
        similarity_score = F.cosine_similarity(z1, z2, dim=1).mean().item()
        
        # Data consistency metric
        data_consistency = self.data_consistency_metric(outputs)
        
        metrics[0]["similarity_score"] = similarity_score
        metrics[0]["data_consistency"] = data_consistency
        metrics[0]['loss'] = loss_value
        metrics[0]['steps'] = 1
        metrics[0]['samples'] = projections.shape[0]
        return metrics
    
    def data_consistency_metric(self, outputs):
        """
        Calculate data consistency between internal and external features
        """
        external_features = outputs.get('external_features', None)
        if external_features is None:
            return 1.0  # No external data, assume perfect consistency
        
        internal_features = outputs['internal_features']
        
        # Calculate distributional similarity
        internal_mean = internal_features.mean(dim=0)
        external_mean = external_features.mean(dim=0)
        
        # Cosine similarity between mean representations
        consistency = F.cosine_similarity(internal_mean.unsqueeze(0), external_mean.unsqueeze(0), dim=1).item()
        
        return max(0.0, consistency)  # Ensure non-negative
    
    def accuracy(self, predictions, y=None):
        """
        Calculate accuracy metric for BYOD
        Returns the consistency score as a proxy for accuracy
        """
        if isinstance(predictions, dict):
            outputs = predictions
            projections = outputs['projections']
        else:
            projections = predictions
        
        batch_size = projections.shape[0]
        z1, z2 = projections[:, 0], projections[:, 1]
        
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Calculate similarity between paired views
        similarities = F.cosine_similarity(z1, z2, dim=1)
        
        # Accuracy is the fraction of pairs above similarity threshold
        accuracy = (similarities > self.similarity_threshold).float().mean().item()
        
        return accuracy
    
    def debug_output_images(self, originals, augmented=None, max_saved=1, output_filename=None, prefix=None, postfix=None, node_id=None):
        return