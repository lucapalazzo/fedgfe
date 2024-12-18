from torch import nn
import torch
from flcore.trainmodel.patchpretexttask import PatchPretextTask
from torchvision import transforms
from typing import Iterator
import numpy as np

from timm.models.vision_transformer import Block

class PatchMasking (PatchPretextTask):
    def __init__(self, backbone=None, input_dim = 768, output_dim = 768, debug_images=False, img_size=224, patch_size=-1, patch_count = -1, mask_ratio = 0.15):
        super(PatchMasking, self).__init__(backbone=backbone, input_dim = input_dim, output_dim = output_dim, debug_images=debug_images, img_size=img_size, patch_size=patch_size, patch_count = patch_count)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = "patch_masking"
        # self.loss = PatchOrderLoss(self.num_patches, self.head_hidden_size)

        self.mask_ratio = mask_ratio
        self.masked_count = int(self.mask_ratio * self.patch_count)
        self.masked_indices = []
        self.pretext_loss = nn.MSELoss(reduction='none')
        self.pretext_head = nn.Sequential( nn.Linear(self.output_dim, self.masked_count * self.patch_size * 3 ) ).to(self.device)

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        modules = nn.ModuleList()
        modules.add_module("pretext_head", self.pretext_head)
        parameters = modules.parameters(recurse)
        return parameters

    def preprocess_sample(self, x):
        
        images, self.masked_indices = self.random_masking(x, self.mask_ratio) 
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
    
    def decoder_create(self, x):
        self.decoder_embed = nn.Linear(self.vit.embed_dim, self.decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(self.decoder_embed_dim, self.decoder_num_heads, self.mlp_ratio, qkv_bias=True, norm_layer=self.norm_layer)
            for _ in range(self.decoder_depth)
        ])
        self.decoder_norm = self.norm_layer(self.decoder_embed_dim)
        self.decoder_pred = nn.Linear(self.decoder_embed_dim, self.patch_count * self.patch_count * self.in_chans, bias=True)
        return self.decoder_embed, self.mask_token, self.decoder_pos_embed, self.decoder_blocks, self.decoder_norm, self.decoder_pred
    
    def decode(self, x):

        x = self.decoder_embed(x)
        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        return x
   
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        """
        B, C, H, W = x.shape
        patches = self.patchify(x)

        # num_masked = int(self.mask_ratio * self.patch_count)
        mask = torch.rand(B, self.patch_count).topk(self.masked_count, dim=1, largest=False).indices  # Indici delle patch mascherate
        masked_images = []
        # patches = x.clone()
        for b in range(B):
            patches[b, mask[b]] = 0  # Sostituisci le patch mascherate con zero
            masked_image = self.reconstruct_image_from_patches(patches[b], self.patch_count_per_row, self.patch_size)
            masked_images.append(masked_image)
        masked_images = torch.stack(masked_images)
        return masked_images, mask

        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]

        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]

        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # Generate the binary mask
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore