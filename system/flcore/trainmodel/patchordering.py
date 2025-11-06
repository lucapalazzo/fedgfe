import os
from torch import nn
import torch
from flcore.trainmodel.patchpretexttask import PatchPretextTask
from modelutils.patchorderloss import PatchOrderLoss
from modelutils.custompatchembedding import CustomPatchEmbed
from typing import Iterator

from timm.models.vision_transformer import VisionTransformer
from transformers import ViTModel
from torch.nn import functional as F
import sys

from flcore.trainmodel.map import MAPBlock
import wandb

class MAPPatchOrderingHead(nn.Module):
    """
    Head for Jigsaw task using MAPBlock for pooling.
    """
    def __init__(self,
                 input_dim: int = 768,
                 n_latents: int = 1,
                 n_heads: int = 8,
                 latent_dim: int = 768):
        super().__init__()
        # Backbone ViT without classification head
        self.embed_dim = input_dim
        self.map_block = MAPBlock(embed_dim=input_dim, n_heads=n_heads, n_latents=n_latents)
        self.n_latents = n_latents
        self.hidden_dim = 256
        # Classifier for permutation index
        # self.classifier = nn.Sequential(
        #     nn.Linear(input_dim, self.hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(self.hidden_dim, patch_count),
        # )
        self.classifier = nn.Linear(self.embed_dim, self.n_latents)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        # ViT returns tokens: (B, N+1, D) with CLS at pos 0
        # MAP pooling -> (B, K, D)
        latents = self.map_block(x)
        if latents.ndim == 3 and latents.shape[1] == 1:
            latents = latents.squeeze(1)  # (B, D)
        # classification logits
        logits = self.classifier(latents)
        return logits


class AttentionPoolingHead(nn.Module):
    def __init__(self, input_dim=768, n_heads=8, n_queries=1, n_perm=1000, patch_count = 16):
        super(AttentionPoolingHead, self).__init__()
        self.input_dim = input_dim
        # self.output_dim = output_dim
        self.patch_count = patch_count

        self.n_queries = n_queries
        self.query = nn.Parameter(torch.randn(n_queries, input_dim))  # (q, D)
        self.mha = nn.MultiheadAttention(embed_dim=input_dim, num_heads=n_heads,
                                         batch_first=True)      # PyTorch >=1.10
        self.norm = nn.LayerNorm(input_dim)
        self.classifier = nn.Linear(input_dim, patch_count)

    def forward(self, x):
        B, N, D = x.shape
        h = self.norm(x)

        # Costruisci Q ripetendo la query per il batch
        Q = self.query.unsqueeze(0).expand(B, self.n_queries, D)  # (B,q,D)

        # Cross-attention: Q attende su K=V=h
        pooled, _ = self.mha(Q, h, h)   # pooled: (B,q,D)

        # Se più query, fai media (o concat + MLP, se vuoi più potenza)
        pooled = pooled.mean(dim=1)     # (B,D)

        logits = self.classifier(pooled)
        return logits
        # Implement the forward pass for the attention pooling head
        x = x.transpose(0, 1)  # Change shape to (patch_count, batch_size, input_dim)
        attn_output, _ = self.attention(x, x, x)
        x = attn_output.mean(dim=0)  # Average over patches
        x = self.fc(x)  # Final linear layer
        return x

class ConvolutionalPatchOrderingHead(nn.Module):
    def __init__(self, input_dim, output_dim, patch_count, jigsaw_patch_count, img_size=224):
        super(ConvolutionalPatchOrderingHead, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.patch_count = patch_count
        self.jigsaw_patch_count = jigsaw_patch_count
        self.img_size = img_size

        # Define the convolutional layers and other components here
        # Example:
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1)
        self.fc = nn.Linear(output_dim * (img_size // 2) * (img_size // 2), jigsaw_patch_count ** 2)
    
    def forward(self, x):
        # Implement the forward pass for the convolutional head
        # Example:
        x = self.conv1(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (self.img_size // 2, self.img_size // 2))
        x = x.view(x.size(0), -1)

class LinearPatchOrderingHead(nn.Module):
    def __init__(self, input_dim, output_dim, patch_count, jigsaw_patch_count):
        super(LinearPatchOrderingHead, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.patch_count = patch_count
        self.jigsaw_patch_count = jigsaw_patch_count
        self.head_hidden_size1 = output_dim * 4
        self.head_hidden_size2 = output_dim * 2

        self.fc1 = nn.Linear(self.input_dim * self.patch_count, jigsaw_patch_count**2)
        # self.fc1 = nn.Linear(self.input_dim * self.patch_count, self.head_hidden_size1)
        # self.fc2 = nn.Linear(self.head_hidden_size1, self.head_hidden_size2)
        # self.fc3 = nn.Linear(self.head_hidden_size2, self.jigsaw_patch_count**2)
        # self.act = nn.ReLU()
        # self.dropout = nn.Dropout(0.1)
        # self.norm1 = nn.LayerNorm(self.head_hidden_size1)
        # self.norm2 = nn.LayerNorm(self.head_hidden_size2)

    def forward(self, x):
        B, P, D = x.shape
        x = x.view(B, -1)  # Flatten the input to (B, P * D)
        x = self.fc1(x)
        # x = self.norm1(x)
        # x = self.act(x)
        # x = self.dropout(x)
        # x = self.fc2(x)
        # x = self.norm2(x)
        # x = self.act(x)
        # x = self.fc3(x)
        # x = x.view(-1, self.jigsaw_patch_count, self.jigsaw_patch_count)
        # x = x.permute(0,2,1)
        return x


class PatchOrdering (PatchPretextTask):
    """
    Patch Ordering Pretext Task for Self-Supervised Learning.
    
    Task: Given an image divided into patches that have been shuffled,
    predict the correct position for each patch to reconstruct the original image.
    
    Architecture:
    - Input: Image divided into jigsaw_patch_count patches (e.g., 16 = 4x4 grid)
    - Shuffling: Patches are shuffled according to a random permutation
    - Output: For each patch, predict its correct position in the grid
    - Loss: CrossEntropyLoss for each patch position prediction
    
    Fixed Issues:
    - Corrected loss function to handle [B, P, P] output properly
    - Fixed accuracy calculation to match the task objective
    - Added proper dimension handling and documentation
    """
    pretext_task_name = "patch_ordering"
    task_name = pretext_task_name
    defined_test_metrics = { "accuracy": None }
    defined_train_metrics = { "loss": None }
    task_learning_rate = 1e-4
    task_weight_decay = 1e-5

    def __init__(self, backbone=None, input_dim = 768, output_dim = 768, debug_images=False, img_size=224, patch_size=-1, patch_count = -1, jigsaw_patch_count=49, cls_token_only=False):
        super(PatchOrdering, self).__init__(backbone=backbone, input_dim = input_dim, output_dim = output_dim, debug_images=debug_images, img_size=img_size, patch_size=patch_size, patch_count = patch_count)

        # check if igm_size is a multiple of jigsaw_patch_count**0.5
        if img_size % jigsaw_patch_count**0.5 != 0:
            raise ValueError(f"img_size {img_size} must be a multiple of jigsaw_patch_count**0.5 {jigsaw_patch_count**0.5}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = PatchOrdering.pretext_task_name
        self.patch_ordering = None
        self.head_hidden_size = self.output_dim * 4 
        self.head_hidden_size2 = self.output_dim * 2 
        self.jigsaw_patch_count = jigsaw_patch_count
        self.jigsaw_patch_size = int(img_size // jigsaw_patch_count**0.5)
        self.jigsaw_patch_count_per_row = int(jigsaw_patch_count ** 0.5)
        self.custom_order = self.patch_order_create()
        # self.loss = PatchOrderLoss(self.num_patches, self.head_hidden_size)
        # self.pretext_loss = nn.CrossEntropyLoss(label_smoothing=0.1).to(self.device)
        self.pretext_loss = nn.CrossEntropyLoss().to(self.device)
        self.dropout = 0.1
        self.pretext_modules = nn.ModuleList()
        
        # self.dropout = 0.1
        # self.pretext_head = nn.Sequential( nn.Linear(output_dim, 64).requires_grad_(False),
        #         # nn.ReLU(),
        #         nn.Linear(output_dim, self.patch_count)
        #     ) for _ in range(patch_count)
        # ).to(self.device)

        # self.pretext_head = nn.Linear(output_dim, self.jigsaw_patch_count).to(self.device)

        self.cls_token_only = cls_token_only

        # self.patch_ordering_head = MAPPatchOrderingHead(
        #     input_dim=input_dim,
        #     n_latents=jigsaw_patch_count,  # Number of latents = number of patches
        #     n_heads=8,
        #     latent_dim=output_dim,
        # )
        # self.patch_ordering_head = AttentionPoolingHead(input_dim=input_dim, n_heads=8, n_queries=4, n_perm=1000, patch_count=self.jigsaw_patch_count**2)
        self.patch_ordering_head = LinearPatchOrderingHead(input_dim, output_dim, patch_count, jigsaw_patch_count)
        self.pretext_modules.add_module('pretext_head', self.patch_ordering_head)
        
    @staticmethod
    def define_metrics( metrics_path = None ):
        pretext_task_name = PatchOrdering.pretext_task_name

        path = "/" if metrics_path is None else "/"+metrics_path+"/"
        metrics = []

        for metric in PatchOrdering.defined_test_metrics:
            defined_metric = f"test{path}{PatchOrdering.task_name}_{metric}"
            PatchOrdering.defined_test_metrics[metric] = defined_metric
            metrics.append(defined_metric)

        for metric in PatchOrdering.defined_train_metrics:
            defined_metric = f"train{path}{PatchOrdering.task_name}_{metric}"
            PatchOrdering.defined_train_metrics[metric] = defined_metric
            metrics.append(defined_metric)

        for metric in metrics:
           a = wandb.define_metric(metric, step_metric="round")

        return metrics

        path = "/" if metrics_path is None else "/"+metrics_path+"/"
        metrics = []
        metrics.append(f"train{path}pretext_train_loss_{pretext_task_name}")
        metrics.append(f"test{path}pretext_train_ds_loss_{pretext_task_name}")
        metrics.append(f"test{path}pretext_test_acc_{pretext_task_name}")

        for metric in metrics:
           a = wandb.define_metric(metric, step_metric="ssl_round")
    def patch_embed ( self, x ):
        return self.custom_patch_embed(x)
    
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        parameters = self.pretext_modules.parameters(recurse)
        return parameters
    
    def named_parameters(self, prefix = '', recurse = True, remove_duplicate = True):
        if remove_duplicate:
            seen = set()
            for name, param in self.pretext_modules.named_parameters(prefix=prefix, recurse=recurse):
                if param not in seen:
                    seen.add(param)
                    yield name, param
        else:
            for name, param in self.pretext_modules.named_parameters(prefix=prefix, recurse=recurse):
                yield name, param
        return name, param
    
    def adjust_optimizer(self):
        return None       # self.classifier = nn.Sequential(
        #     nn.Linear(input_dim, self.hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(0.1),
        #     nn.Linear(self.hidden_dim, patch_count),
        #
        decay, no_decay = [], []
        whitelist = (nn.Linear, nn.Conv2d)             # di solito con WD
        blacklist = (nn.BatchNorm1d, nn.BatchNorm2d,
                     nn.LayerNorm, nn.GroupNorm)       # senza WD

        for module_name, module in self.pretext_modules.named_modules():
            for name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue
                full_name = f"{module_name}.{name}" if module_name else name
                if isinstance(module, blacklist) or name == "bias" or param.ndim == 1:
                    no_decay.append(param)
                else:
                    decay.append(param)
        return [
            {"params": decay, "weight_decay": self.task_weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    def preprocess_sample(self, x):
        custom_order = self.patch_order_create(random_order=True, sample_count=x.shape[0])
        self.custom_order = custom_order.to(self.device)
        # Cleanup previous custom_order if it exists on GPU
        if hasattr(self, '_prev_custom_order'):
            del self._prev_custom_order
        self._prev_custom_order = self.custom_order
        images, self.image_ordering_labels = self.shuffle_patches(x,self.custom_order)
        return images
    
 
    def test_metrics(self, predictions, y = None, samples = None, metrics = None):
        accuracy = self.accuracy(predictions, y)
        if metrics is None:
            return None
        metrics[0]["accuracy"] = accuracy.item()
        metrics[0]['steps'] = 1
        metrics[0]['samples'] = predictions.shape[0]
        return metrics
    
    def accuracy_old(self, x, y = None):
        """Old accuracy function - kept for reference."""
        # targets = torch.tensor(self.custom_order).long().to(self.device).unsqueeze(0).expand(x[0].shape[0],-1)
        targets = torch.tensor(self.custom_order).long().to(self.device)

        true_order = 0
        patches = 0
        # x: [batch, num_patches, num_classes], targets: [batch, num_patches]
        predicted = x.argmax(dim=-1)  # [batch, num_patches]
        correct = (predicted == targets).float()
        true_order = correct.sum()
        patches = correct.numel()

        accuracy = true_order / patches  
        return accuracy
        for output in x:
            true_order = (output.argmax(dim=1) == targets).float()
        return (x.argmax(dim=1) == targets).float
    
    def accuracy(self, x, y = None):
        """
        Corrected accuracy function for patch ordering.
        
        Args:
            x: Model output [batch_size, jigsaw_patch_count, jigsaw_patch_count]
            y: Not used, targets are taken from self.custom_order
            
        Returns:
            Accuracy as fraction of correctly predicted patch positions
        """
        targets = torch.tensor(self.custom_order).long().to(self.device)  # [B, patch_count]
        
        # x shape: [B, patch_count, patch_count] - predictions for each patch position
        # targets shape: [B, patch_count] - correct position for each patch
        
        B, P, C = x.shape
        
        # For each patch, get the predicted position (argmax over possible positions)
        predicted_positions = x.argmax(dim=-1)  # [B, patch_count] - predicted position for each patch
        
        # Compare with target positions
        correct_predictions = (predicted_positions == targets).float()
        
        # Calculate accuracy: total correct predictions / total predictions
        total_correct = correct_predictions.sum()
        total_predictions = correct_predictions.numel()
        
        accuracy = total_correct / total_predictions if total_predictions > 0 else torch.tensor(0.0)
        return accuracy

    def loss_old(self, x, y = None):
        """Old loss function with dimension mismatch - kept for reference."""
        targets = torch.tensor(self.custom_order).to(self.device)
        # 1) Rumplasci logits da (B, P, C) a (B*P, C)
        # logits_flat = x.view(-1, self.jigsaw_patch_count)

        # 2) Rumplasci target da (B, P) a (B*P,)
        # target_flat = targets.view(-1)
        # targets = targets.unsqueeze(0).expand(x.shape[0],-1)
        loss = self.pretext_loss(x, targets)
        # loss = self.pretext_loss(x, targets)
        
        return loss
    
    def loss(self, x, y = None):
        """
        Corrected loss function for patch ordering.
        
        Args:
            x: Model output [batch_size, jigsaw_patch_count, jigsaw_patch_count]
            y: Not used, targets are taken from self.custom_order
            
        Returns:
            CrossEntropyLoss averaged over all patches
        """
        targets = torch.tensor(self.custom_order).to(self.device)  # [B, patch_count]
        
        # x shape: [B, patch_count, patch_count] - predictions for each patch position
        # targets shape: [B, patch_count] - correct position for each patch
        
        B, P, C = x.shape
        assert P == C == self.jigsaw_patch_count, f"Expected [{B}, {self.jigsaw_patch_count}, {self.jigsaw_patch_count}], got [{B}, {P}, {C}]"
        
        total_loss = 0.0
        
        # For each patch, predict its correct position
        for patch_idx in range(P):
            patch_logits = x[:, patch_idx, :]  # [B, jigsaw_patch_count] - position predictions for patch_idx
            patch_targets = targets[:, patch_idx]  # [B] - correct position for patch_idx
            
            # CrossEntropyLoss expects logits [B, C] and targets [B]
            patch_loss = self.pretext_loss(patch_logits, patch_targets.long())
            total_loss += patch_loss
        
        # Average loss over all patches
        avg_loss = total_loss / P
        return avg_loss
    
    def forward(self, samples):
        batch_shape = samples.shape
        x = self.preprocess_sample(samples)
        if self.backbone is not None:
            # self.backbone.logits_only = True
            x = self.backbone(x)

        if isinstance(self.backbone, VisionTransformer):
            if self.cls_token_only:
                x = x[:, 0, :]
            else:
                x = x[:, 1:, :]
        elif isinstance(self.backbone, ViTModel):
            if self.cls_token_only:
                x = x.last_hidden_state[:, 0, :]
            else:
                x = x.last_hidden_state[:, 1:, :]
                # x = x.reshape(batch_shape[0], -1 )
                # if self.jigsaw_patch_count != self.patch_count:
                #     x = self.pool_output_embedding(x, int(self.jigsaw_patch_count**0.5) * 2)
        B, P, D_in = x.shape

        x = self.patch_ordering_head(x)  # (B, jigsaw_patch_count, jigsaw_patch_count)
        logits = x.view(B, self.jigsaw_patch_count, self.jigsaw_patch_count)  # (B, jigsaw_count, jigsaw_count)
        # logits = logits.permute(0, 2, 1)  # (B, jigsaw_patch_count, jigsaw_patch_count)
        return logits 
    
    def debug_output_images(self, samples, predictions, max_saved=1, output_filename=None, prefix=None, postfix=None, node_id=None):
        if node_id is None:
            node_id = self.id

        if self.debug_images or os.path.exists("save_debug_images"):

            # self.save_images(x, predictions, max_saved=1)
            custom_order = self.custom_order.to(self.device)
            original_patches = self.patchify(samples, self.jigsaw_patch_count)
            shuffled_images = []
            reordered_images = []
            original_images = []
            # shuffled_patches: [batch, num_patches, ...]
            # custom_order: [batch, num_patches]
            batch_indices = torch.arange(original_patches.shape[0]).unsqueeze(-1).to(self.custom_order.device)
            shuffled_patches = original_patches[batch_indices, self.custom_order, :].clone()
            predicted_order = predictions.argmax(dim=-1) # shape: [batch, num_patches]
            inverse_custom_order = torch.argsort(custom_order)
            # reconstructed_indices = custom_order[predicted_order]
            # Both predicted_order and custom_order are [b, 16]
            # For each batch, we want to invert the permutation: for each predicted position, find which patch it corresponds to in the custom order.
            # This is equivalent to: for each batch, argsort(predicted_order) gives the indices to reorder to original order.
            # But to reconstruct the original image, we want to map predicted positions back to the original patch indices.
            # So, for each batch, we want to invert the permutation in predicted_order.
            # The correct way is:
            reconstructed_indices = torch.argsort(predicted_order, dim=1)
            reordered_patches = shuffled_patches[batch_indices, reconstructed_indices]
            # patches_reordered = shuffled_patches[:, reconstructed_indices, :]
            # patches_reordered = shuffled_patches[torch.arange(shuffled_patches.size(0)).unsqueeze(1), inverse_predicted_order]
            # patches_reordered = patches_reordered.reshape(original_patches.shape[0], -1, 3, self.patch_size, self.patch_size)
            # patches_reordered = shuffled_patches[::][:, predicted_order, :]
            for i in range(original_patches.shape[0]):
                # reordered_patches = shuffled_patches[i][reconstructed_indices[i]]
                # patches_reordered = shuffled_patches[i][reconstructed_indices[i]]
                original_image = self.reconstruct_image_from_patches(original_patches[i], self.jigsaw_patch_count_per_row, self.jigsaw_patch_size)
                shuffled_image = self.reconstruct_image_from_patches(shuffled_patches[i], self.jigsaw_patch_count_per_row, self.jigsaw_patch_size)
                reordered_image = self.reconstruct_image_from_patches(reordered_patches[i], self.jigsaw_patch_count_per_row, self.jigsaw_patch_size)
                original_images.append(original_image)
                shuffled_images.append(shuffled_image)
                reordered_images.append(reordered_image)

            original_images = torch.stack(original_images)
            shuffled_images = torch.stack(shuffled_images)
            reordered_images = torch.stack(reordered_images)

            save_images = [original_images, shuffled_images, reordered_images]

            if prefix is not None:
                prefix = f"{prefix}_"
            else:
                prefix = ""

            if postfix is not None:
                postfix = f"_{postfix}"
            else:
                postfix = ""
            filename = f"{prefix}patch_ordering_{node_id}{postfix}.png"
            self.save_images(save_images, output_filename=filename, max_saved=max_saved)
    
    def patch_order_create ( self, random_order = True, sample_count = 1):
        if random_order:
            order = torch.stack([torch.randperm(self.jigsaw_patch_count) for _ in range(sample_count)], dim=0)
            # order = torch.stack([torch.arange(self.jigsaw_patch_count-1,-1,-1) for _ in range(sample_count)], dim=0)
            # create reversed order for entire batch
            # order = torch.randperm(self.jigsaw_patch_count)
            # order = order.unsqueeze(0).expand(sample_count, -1)
        else:
            order = torch.stack([torch.arange(self.jigsaw_patch_count) for _ in range(sample_count)], dim=0)
        return order
    
    def shuffle_patches(self, images, order):
        B, C, H, W = images.shape
        # num_patches_per_row = int(math.sqrt(N))
        # device = patches.device

        # shuffled_batches = []

        # shuffled_batches = patches[:, -1,:][:,order,:]
        num_patches_per_row = int(self.jigsaw_patch_count ** 0.5)
        # num_patches_per_row = H // self.patch_size
        # num_patches_per_row = self.num_patches**0.5
        # self.patch_size = int(H // num_patches_per_row)

        patches = self.patchify(images, patch_count=self.jigsaw_patch_count)
        # order: [batch_size, num_patches]
        # patches: [batch_size, num_patches, ...]
        # We want to shuffle patches in each batch according to its order
        batch_indices = torch.arange(patches.shape[0]).unsqueeze(-1).to(order.device)
        shuffled_batches = patches[batch_indices, order]
        reconstructed_imgs = []
        patch_size = H // num_patches_per_row
        # self.save_images(images, shuffled_batches, max_saved=1)
        for b in range(shuffled_batches.shape[0]):
            reconstructed_img = self.reconstruct_image_from_patches(shuffled_batches[b], num_patches_per_row, patch_size)

            reconstructed_imgs.append(reconstructed_img)
        #     shuffled_patches = []
        #     batch_patches = patches[b]
        #     for i, patch in enumerate(batch_patches):
        #         shuffled_patches.append(batch_patches[order[i]])
        #     shuffled_patches = torch.stack(shuffled_patches) 
        #     # shuffled_patches = torch.Tensor(shuffled_patches).to(device)
        #     reconstructed_img = self.reconstruct_image_from_patches(shuffled_patches, num_patches_per_row, self.patch_size)
        #     save_grid_images([reconstructed_img], nrow=1, output_path=os.path.join(self.image_output_directory, f"shuffled_{b}.png"))
        #     shuffled_batches.append(shuffled_patches)
        reconstructed_imgs = torch.stack(reconstructed_imgs)
        # filename = f"jigsaw_{self.id}.png"
        # self.save_images([images, reconstructed_imgs], output_filename=filename, max_saved=1)

        return reconstructed_imgs, order