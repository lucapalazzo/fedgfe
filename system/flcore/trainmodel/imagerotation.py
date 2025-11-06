import os
from torch import nn
import torch
from flcore.trainmodel.patchpretexttask import PatchPretextTask
from typing import Iterator
from torchvision import transforms
import numpy as np

from timm.models.vision_transformer import VisionTransformer
from transformers import ViTModel
import wandb


class ImageRotation (PatchPretextTask):
    pretext_task_name = "image_rotation"
    task_name = pretext_task_name
    defined_test_metrics = { "accuracy": None }
    defined_train_metrics = { "loss": None }
    task_weight_decay = 0.00001
    def __init__(self, backbone=None, input_dim = 768, output_dim = 768, debug_images=False, img_size=224, patch_size=-1, patch_count = -1, cls_token_only=False):
        super(ImageRotation, self).__init__(backbone=backbone, input_dim = input_dim, output_dim = output_dim, debug_images=debug_images, img_size=img_size, patch_size=patch_size, patch_count = patch_count, cls_token_only=cls_token_only)
        self.task_name = ImageRotation.pretext_task_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.backbone = backbone
        self.debug_images = debug_images
        self.head = None
        self.name = ImageRotation.pretext_task_name

        # Rotation
        # self.rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        self.rotation_angles = [0, 90, 180, 270]
        self.rotation_angles_count = len(self.rotation_angles)
        self.image_rotation_angles = self.rotation_angles

        self.image_rotation_labels = None
        
        self.pretext_loss = nn.CrossEntropyLoss()

        # if self.cls_token_only == False:
        #     output_dim = output_dim * self.patch_count

        self.hidden_dim = output_dim
        self.pooled_output_dim = 1
        self.dropout = 0.1

        self.fl = nn.Flatten(1)
        self.ln = nn.LayerNorm(output_dim) if self.cls_token_only else nn.LayerNorm(output_dim * self.patch_count)
        self.fc1 = nn.Linear(output_dim, self.hidden_dim) if self.cls_token_only else nn.Linear(output_dim * self.patch_count, self.hidden_dim) 
        self.relu = nn.ReLU(inplace=True)
        self.dropout_layer = nn.Dropout(self.dropout)
        self.fc2 = nn.Linear(self.hidden_dim, self.rotation_angles_count)

        self.pretext_head = nn.Sequential(
            self.fl,
            self.ln,
            self.fc1,
            self.relu,
            self.dropout_layer,
            self.fc2
            # nn.LayerNorm(output_dim),
            # nn.Linear(output_dim, self.hidden_dim),
            # nn.ReLU(inplace=True),
            # nn.Dropout(self.dropout),
            # nn.Linear(self.hidden_dim, self.rotation_angles_count),
        )

        # self.pretext_head = nn.Sequential(
        #     nn.Linear(output_dim, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, self.rotation_angles_count)
            
        #     ).to(self.device)
    
    @staticmethod
    def define_metrics( metrics_path = None ):
        pretext_task_name = ImageRotation.pretext_task_name

        path = "/" if metrics_path is None else "/"+metrics_path+"/"
        metrics = []

        for metric in ImageRotation.defined_test_metrics:
            defined_metric = f"test{path}{ImageRotation.task_name}_{metric}"
            ImageRotation.defined_test_metrics[metric] = defined_metric
            metrics.append(defined_metric)

        for metric in ImageRotation.defined_train_metrics:
            defined_metric = f"train{path}{ImageRotation.task_name}_{metric}"
            ImageRotation.defined_train_metrics[metric] = defined_metric
            metrics.append(defined_metric)

        for metric in metrics:
           a = wandb.define_metric(metric, step_metric="round")

        return metrics

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        modules = nn.ModuleList()
        modules.add_module("pretext_head", self.pretext_head)
        parameters = modules.parameters(recurse)
        return parameters
    
    def debug_output_images(self, samples, predictions, max_saved=1, output_filename=None, prefix=None, postfix=None, node_id=None):
        rotated_images = []
        images = []
        reconstructed_images = []
        for image, label, prediction in zip(samples, self.image_rotation_labels, predictions):
            # if isinstance(image, torch.Tensor):
            #     image = image.cpu().numpy()
            # if isinstance(label, torch.Tensor):
            #     label = label.cpu().numpy()
            # if isinstance(prediction, torch.Tensor):
            #     prediction = prediction.cpu().numpy()

            rotated = transforms.functional.rotate(image, self.image_rotation_angles[label.item()]).to(image.device)
            rotated_images.append(rotated)
            images.append(image)
            reconstructed = transforms.functional.rotate(image, self.image_rotation_angles[prediction.argmax().item()]).to(image.device)
            reconstructed_images.append(reconstructed)


        if prefix is None:
            prefix = ""
        else:
            prefix = f"{prefix}_"

        if postfix is None:
            postfix = ""
        else:
            postfix = f"_{postfix}"
                
        rotated_images = torch.stack(rotated_images, dim=0)
        images = torch.stack(images, dim=0)
        reconstructed_images = torch.stack(reconstructed_images, dim=0)
        if output_filename is None:
            output_filename = f"{prefix}imagerotation_{node_id}{postfix}.png"
        debug_images = [images, rotated_images, reconstructed_images]
        self.save_images(debug_images, max_saved=max_saved, output_filename=output_filename)
        return
    
    def preprocess_sample(self, x):
        images, self.image_rotation_labels = self.rotate_images(x, self.image_rotation_angles)
        return images

    def test_metrics(self, predictions, y = None, samples = None, metrics = None):
        accuracy = self.accuracy(predictions, y)
        if metrics is None:
            return None
        metrics[0]["accuracy"] = accuracy.item()
        metrics[0]['steps'] = 1
        metrics[0]['samples'] = predictions.shape[0]
        return metrics

    def accuracy(self, x, y = None):
        target = torch.tensor(self.image_rotation_labels).long().to(x.device)
        truecount = (x.argmax(dim=1) == target).float().sum()
        # print ( "true count ", truecount.item(),  x.argmax(dim=1), target, x.argmax(dim=1) == target)
        accuracy = truecount / x.shape[0]
        return accuracy
        return (x.argmax(dim=1) == target).float().mean()
     
    def loss(self, x, y = None):
        target = torch.tensor(self.image_rotation_labels).long().to(x.device)

        return self.pretext_loss(x, target)

    
    def forward(self, x):
        x = self.preprocess_sample(x)

        if self.backbone is not None:
            # self.backbone.logits_only = True
            x = self.backbone(x)
            
        if self.cls_token_only:
            if isinstance(self.backbone, VisionTransformer):
                x = x[:,0,:]
            elif isinstance(self.backbone, ViTModel):
                x = x.last_hidden_state[:,0,:]
        else:
            if isinstance(self.backbone, VisionTransformer):
                x = x[:,1:,:]
            elif isinstance(self.backbone, ViTModel):
                x = x.last_hidden_state[:,1:,:]
                # x = self.pool_output_embedding(x, self.pooled_output_dim)
                # x = x.view(x.shape[0], -1)
        # x = self.fl(x)
        # x = self.ln(x)
        # x = self.fc1(x)
        # x = self.relu(x)
        # x = self.dropout_layer(x)
        # x = self.fc2(x)
        x = self.pretext_head(x)

        return x

    def rotate_images(self, imgs, image_rotation_angles = None):
        B, C, H, W = imgs.shape

        device = imgs.device
        num_patches_per_row = H // self.patch_count
        num_patches = num_patches_per_row ** 2
        labels = torch.zeros(B, dtype=torch.long, device=device)
        rotated_imgs = torch.zeros_like(imgs).to(imgs.device    )

        for b in range(B):
            angle = np.random.choice(self.image_rotation_angles)

            angle_index = self.image_rotation_angles.index(angle.item())
            labels[b] = angle_index
            img = imgs[b]
            rotated = transforms.functional.rotate(img, angle.item()).to(device)
            rotated_imgs[b] = rotated
        return rotated_imgs, labels 