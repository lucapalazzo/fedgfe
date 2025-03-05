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
    def __init__(self, backbone=None, input_dim = 768, output_dim = 768, debug_images=False, img_size=224, patch_size=-1, patch_count = -1):
        super(ImageRotation, self).__init__(backbone=backbone, input_dim = input_dim, output_dim = output_dim, debug_images=debug_images, img_size=img_size, patch_size=patch_size, patch_count = patch_count)

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
        self.pretext_head = nn.Sequential( nn.Linear(output_dim, self.rotation_angles_count) ).to(self.device)
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
        metrics.append(f"train{path}pretext_train_loss_{pretext_task_name}")
        metrics.append(f"test{path}pretext_train_ds_loss_{pretext_task_name}")
        metrics.append(f"test{path}pretext_test_acc_{pretext_task_name}")

        for metric in metrics:
           a = wandb.define_metric(metric, step_metric="ssl_round")

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        modules = nn.ModuleList()
        modules.add_module("pretext_head", self.pretext_head)
        parameters = modules.parameters(recurse)
        return parameters

    def preprocess_sample(self, x):
        images, self.image_rotation_labels = self.rotate_images(x, self.image_rotation_angles)
        if self.debug_images or os.path.exists("save_debug_images"):
            save_images = [x,images]
            self.save_images(save_images, None, max_saved=1)

        return images

    def accuracy(self, x, y = None):
        target = torch.tensor(self.image_rotation_labels).long().to(x.device)
        truecount = (x.argmax(dim=1) == target).float().sum()
        # print ( "true count ", truecount.item(),  x.argmax(dim=1), target, x.argmax(dim=1) == target)
        accuracy = truecount / x.shape[0]
        return accuracy
        return (x.argmax(dim=1) == target).float().mean()
     
    def loss(self, x, y = None):
        target = torch.tensor(self.image_rotation_labels).long().to(x.device)
        # target = target.unsqueeze(0).expand(x.shape[0],-1)
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

        return self.pretext_head(x)

    def rotate_images(self, imgs, image_rotation_angles = None):
        B, C, H, W = imgs.shape

        device = imgs.device
        num_patches_per_row = H // self.patch_count
        num_patches = num_patches_per_row ** 2
        labels = torch.zeros(B, dtype=torch.long, device=device)
        rotated_imgs = torch.zeros_like(imgs).to(imgs.device)

        for b in range(B):
            angle = np.random.choice(self.image_rotation_angles)

            angle_index = self.image_rotation_angles.index(angle.item())
            labels[b] = angle_index
            img = imgs[b]
            rotated = transforms.functional.rotate(img, angle.item()).to(device)
            rotated_imgs[b] = rotated
        return rotated_imgs, labels 