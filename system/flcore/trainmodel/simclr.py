import os
from torch import nn
import torch
from flcore.trainmodel.patchpretexttask import PatchPretextTask
from modelutils.custompatchembedding import CustomPatchEmbed
from typing import Iterator

from timm.models.vision_transformer import VisionTransformer
from transformers import ViTModel
from torch.nn import functional as F
import wandb
import torchvision.transforms as transforms


class SimCLR(PatchPretextTask):
    pretext_task_name = "simclr"
    task_name = pretext_task_name
    defined_test_metrics = {"accuracy": None}
    defined_train_metrics = {"loss": None}
    task_weight_decay = 1e-5  # Default weight decay for optimizer
    task_learning_rate = 1e-4  # Default learning rate for optimizer
    
    def __init__(self, backbone=None, input_dim=768, output_dim=768, debug_images=False, img_size=224, 
                 patch_size=-1, patch_count=-1, temperature=0.5, projection_dim=128, augment_strength=1.0):
        super(SimCLR, self).__init__(backbone=backbone, input_dim=input_dim, output_dim=output_dim, 
                                     debug_images=debug_images, img_size=img_size, patch_size=patch_size, 
                                     patch_count=patch_count)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = SimCLR.pretext_task_name
        self.temperature = temperature
        
        # Projection head (MLP) as described in the SimCLR paper
        self.projection_head = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, projection_dim)
        ).to(self.device)

        self.augmented_images = []
        
        # Data augmentation pipeline
        self.transform = self._get_augmentation_pipeline(strength=augment_strength)

    def _get_augmentation_pipeline(self, strength=1.0):
        """Create data augmentation pipeline for SimCLR"""
        s = strength
        color_jitter = transforms.ColorJitter(
            brightness=0.8*s, contrast=0.8*s, saturation=0.8*s, hue=0.2*s)
            
        return transforms.Compose([
            transforms.RandomResizedCrop(size=self.img_size, scale=(0.6, 0.9)),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomApply([color_jitter], p=0.8),
            # transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(1.0, 3.0))], p=0.3),
            # transforms.GaussianBlur(kernel_size=max(3, int(0.1 * self.img_size) // 2 * 2 + 1)),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @staticmethod
    def define_metrics(metrics_path=None):
        pretext_task_name = SimCLR.pretext_task_name

        path = "/" if metrics_path is None else "/"+metrics_path+"/"
        metrics = []

        for metric in SimCLR.defined_test_metrics:
            defined_metric = f"test{path}{SimCLR.task_name}_{metric}"
            SimCLR.defined_test_metrics[metric] = defined_metric
            metrics.append(defined_metric)

        for metric in SimCLR.defined_train_metrics:
            defined_metric = f"train{path}{SimCLR.task_name}_{metric}"
            SimCLR.defined_train_metrics[metric] = defined_metric
            metrics.append(defined_metric)

        for metric in metrics:
           wandb.define_metric(metric, step_metric="round")

        return metrics
    
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        modules = nn.ModuleList()
        modules.add_module("projection_head", self.projection_head)
        parameters = modules.parameters(recurse)
        return parameters
    
    def preprocess_sample(self, x):
        """Generate two augmented views of the same batch"""
        batch_size = x.shape[0]
        
        # Apply different augmentations to create two views
        transformed_imgs = []
        self.augmented_images = []
        for img in x:
            # Apply two different sets of transformations to each image
            aug1 = self.transform(img)
            aug2 = self.transform(img)
            # aug2 = self._apply_transforms_to_tensor(img)
            transformed_imgs.extend([aug1, aug2])
            self.augmented_images.append([aug1, aug2])

        # self.debug_output_images(x, self.save_images, max_saved=1, prefix='simclr')    
        transformed = torch.stack(transformed_imgs)
        # Reshape to have (batch_size, 2, C, H, W)
        transformed = transformed.view(batch_size, 2, x.shape[1], x.shape[2], x.shape[3])
        
        return transformed
    
    def _apply_transforms_to_tensor(self, img):
        """Apply transformations to a single tensor image"""
        # Convert to PIL for transforms that require it
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_tensor = torch.tensor(img_np).permute(2, 0, 1).to(self.device)
        
        # Apply individual transforms that work with tensors
        if torch.rand(1).item() > 0.5:
            img_tensor = transforms.functional.hflip(img_tensor)
            
        # Add random color jitter
        if torch.rand(1).item() > 0.2:
            img_tensor = transforms.functional.adjust_brightness(img_tensor, 1.0 + (torch.rand(1).item() - 0.5))
            img_tensor = transforms.functional.adjust_contrast(img_tensor, 1.0 + (torch.rand(1).item() - 0.5))
            img_tensor = transforms.functional.adjust_saturation(img_tensor, 1.0 + (torch.rand(1).item() - 0.5))
        
        return img_tensor
        
    def forward(self, samples):
        """
        Forward pass for SimCLR
        samples: Input tensor of shape [batch_size, channels, height, width]
        """
        # Get two differently augmented versions of the same batch
        augmented = self.preprocess_sample(samples)  # [batch_size, 2, C, H, W]
        batch_size = augmented.shape[0]
        
        # Reshape to process all augmentations in a single batch
        augmented_flat = augmented.view(-1, augmented.shape[2], augmented.shape[3], augmented.shape[4])  # [batch_size*2, C, H, W]
        
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
                    features = features.last_hidden_state[:, 1:, :]  # Use mean of patch embeddings
                    features = features.mean(dim=1)  # Average over patches


                    #####
                    # patches = features.view(batch_size, 2, -1, features.size(-1))  # [B, 2, P, D]
                    # patches_flat = features.reshape(-1, patches.size(-1))   # [B*P, D]
                    # z_tokens = self.projection_head(patches_flat)          # [B*P, proj_dim]
                    # z_tokens = z_tokens.features(features.size(0), -1, z_tokens.size(-1))  # [B, P, proj_dim]
        else:
            features = augmented_flat
            
        # Project features to the latent space for contrastive loss
        projections = self.projection_head(features)
        projections = F.normalize(projections, dim=1)  # Normalize embeddings to unit sphere
        
        # Reshape back to [batch_size, 2, projection_dim]
        projections = projections.view(batch_size, 2, -1)
        
        # # For visualization purposes
        # self.debug_output_images(samples, augmented, max_saved=1)
        
        return projections
    
    def loss(self, projections, y=None):
        """
        NT-Xent loss as defined in the SimCLR paper
        projections: tensor of shape [batch_size, 2, projection_dim]
        """
        batch_size = projections.shape[0]
        
        # Reshape to get all projections in a single dimension
        z1, z2 = projections[:, 0], projections[:, 1]  # [batch_size, projection_dim]
        # L2 normalize
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1) 
        
        z = torch.cat([z1, z2], dim=0)                        # [2B, D]
        B = z1.size(0)

        sim = torch.matmul(z, z.T) / self.temperature              # [2B, 2B]

        # Maschera diagonale
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float('-inf'))

        # Target: per i primi B -> indice +B; per i secondi B -> indice -B
        targets = torch.arange(B, 2 * B, device=z.device)
        targets = torch.cat([targets, torch.arange(0, B, device=z.device)], dim=0)  # [2B]

        
        # # Compute similarity matrix
        # similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        
        # # Remove diagonal for same sample comparisons (the embedding itself)
        # mask = ~torch.eye(2 * batch_size, device=self.device, dtype=torch.bool)
        # similarity_matrix = similarity_matrix.masked_select(mask).view(2 * batch_size, -1)
        
        # # Create positive pair labels
        # positives = torch.cat([
        #     torch.arange(batch_size, 2 * batch_size),
        #     torch.arange(batch_size)
        # ], dim=0).to(self.device)
        
        # # Calculate NT-Xent loss (normalized temperature-scaled cross-entropy)
        # logits = similarity_matrix / self.temperature
        # labels = positives
        loss = F.cross_entropy(sim, targets)
        
        return loss
    
    def test_metrics(self, predictions, y=None, samples=None, metrics=None):
        """Calculate metrics for evaluation including improved accuracy metric"""
        # For SimCLR, a common metric is the NT-Xent loss on test data
        loss_value = self.loss(predictions).item()
        
        if metrics is None:
            return None
        
        # Calculate improved accuracy metric for SimCLR
        accuracy = self.accuracy(predictions)
        
        metrics[0]["accuracy"] = accuracy
        metrics[0]['loss'] = loss_value
        metrics[0]['steps'] = 1
        metrics['samples'] = predictions.shape[0]
        return metrics
    
    def accuracy(self, predictions, y=None):
        """
        Calculate accuracy for SimCLR using nearest neighbor approach.
        
        This metric measures how often the positive pair (z1, z2) from the same image
        is more similar than random negative pairs, which is more meaningful than 
        just cosine similarity.
        
        Args:
            predictions: tensor of shape [batch_size, 2, projection_dim]
            
        Returns:
            Accuracy as fraction of correctly identified positive pairs
        """
        batch_size = predictions.shape[0]
        z1, z2 = predictions[:, 0], predictions[:, 1]  # [batch_size, projection_dim]
        
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        total_correct = 0
        
        # For each z1, check if z2 is the most similar among all z2s
        for i in range(batch_size):
            query = z1[i].unsqueeze(0)  # [1, projection_dim]
            
            # Calculate similarity with all z2 candidates
            similarities = F.cosine_similarity(query, z2, dim=1)  # [batch_size]
            
            # The correct match should have the highest similarity
            predicted_match = similarities.argmax().item()
            if predicted_match == i:
                total_correct += 1
        
        # Also check the reverse: for each z2, is z1 the closest?
        for i in range(batch_size):
            query = z2[i].unsqueeze(0)  # [1, projection_dim]
            
            # Calculate similarity with all z1 candidates
            similarities = F.cosine_similarity(query, z1, dim=1)  # [batch_size]
            
            # The correct match should have the highest similarity
            predicted_match = similarities.argmax().item()
            if predicted_match == i:
                total_correct += 1
        
        # Calculate accuracy: correct matches / total possible matches
        accuracy = total_correct / (2 * batch_size)
        return accuracy
    
    def debug_output_images(self, originals, augmented, max_saved=1, output_filename=None, prefix=None, postfix=None, node_id=None):
        """Save debug images showing original and augmented samples"""
        if not (self.debug_images or os.path.exists("save_debug_images")):
            return
            
        if node_id is None:
            node_id = getattr(self, 'id', '0')
            
        if prefix is not None:
            prefix = f"{prefix}_"
        else:
            prefix = ""
            
        if postfix is not None:
            postfix = f"_{postfix}"
        else:
            postfix = ""
            
        filename = output_filename or f"{prefix}simclr_{node_id}{postfix}.png"
        
        augmented1_images = torch.stack([aug[0] for aug in self.augmented_images])
        augmented2_images = torch.stack([aug[1] for aug in self.augmented_images])
        # Create a grid of images: original, aug1, aug2 stacked horizontally
        b, c, h, w = originals.shape
        combined_images = torch.zeros(b, c, h, w*3, device=originals.device)
        combined_images[:, :, :, 0:w] = originals
        combined_images[:, :, :, w:2*w] = augmented1_images
        combined_images[:, :, :, 2*w:3*w] = augmented2_images
        save_images = [ combined_images ]
        
        self.save_images(save_images, output_filename=filename, max_saved=max_saved)
