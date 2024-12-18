from torch import nn
import torch
from flcore.trainmodel.patchpretexttask import PatchPretextTask
from torchvision import transforms
from typing import Iterator
import numpy as np


class PatchRotation (PatchPretextTask):
    def __init__(self, backbone=None, input_dim = 768, output_dim = 768, debug_images=False, img_size=224, patch_size=-1, patch_count = -1):
        super(PatchRotation, self).__init__(backbone=backbone, input_dim = input_dim, output_dim = output_dim, debug_images=debug_images, img_size=img_size, patch_size=patch_size, patch_count = patch_count)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = "patch_rotation"
        # self.loss = PatchOrderLoss(self.num_patches, self.head_hidden_size)

        self.rotation_angles = [0, 90, 180, 270]
        self.rotation_angles_count = len(self.rotation_angles)
        self.patch_rotation_labels = None

        self.patch_rotation_angles_order = None
        
        self.pretext_loss = nn.CrossEntropyLoss()
        self.pretext_head = nn.Sequential( nn.Linear(output_dim, self.patch_count) ).to(self.device)

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        modules = nn.ModuleList()
        modules.add_module("pretext_head", self.pretext_head)
        parameters = modules.parameters(recurse)
        return parameters

    def preprocess_sample(self, x):
        images, self.patch_rotation_labels = self.rotate_patches(x,self.patch_rotation_angles_order)
        self.save_images(x, images, max_saved=1)
        return images
    
    def loss(self, x, y = None):
        target = torch.tensor(self.patch_rotation_labels).long().to(x.device)
        return self.pretext_loss(x, target)

    
    def forward(self, x):
        x = self.preprocess_sample(x)
        if self.backbone is not None:
            # self.backbone.logits_only = True
            x = self.backbone(x)


        return self.pretext_head(x)

    def rotate_patches(self, imgs, rotation_angles_order = None):
        B, C, H, W = imgs.shape
        device = imgs.device
        num_patches_per_row = int(self.patch_count ** 0.5)
        num_patches = self.patch_count
        labels = torch.zeros(B, num_patches, dtype=torch.long, device=device)
        rotated_imgs = torch.zeros_like(imgs)

        patched_imgs = self.patchify(imgs)
        for b in range(B):
            patches = patched_imgs[b]
            rotated_patches = []
            for i,patch in enumerate(patches):
                if (rotation_angles_order is not None):
                    angle = self.rotation_angles[rotation_angles_order[i]]
                else:
                    angle = np.random.choice(self.rotation_angles)
                rotated_patch = transforms.functional.rotate(patch, angle.item()).to(device)
                rotated_patches.append(rotated_patch.unsqueeze(0))
                labels[b, i] = self.rotation_angles.index(angle)
            rotated_patches = torch.cat(rotated_patches, dim=0)
            reconstructed_img = self.reconstruct_image_from_patches(rotated_patches, num_patches_per_row, self.patch_size)
            rotated_imgs[b] = reconstructed_img

        return rotated_imgs, labels