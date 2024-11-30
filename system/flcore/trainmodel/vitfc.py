# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

#
# decoder: 2 linear (256-64) + relu

import math
import torch
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import nn, Tensor
from torchvision.models import VisionTransformer
import torch.nn.functional as F
import numpy as np
from modelutils.custompatchembedding import CustomPatchEmbed
from modelutils.patchorderloss import PatchOrderLoss
from modelutils.patchmaskloss import PatchMaskLoss
from flcore.trainmodel.downstream import DownstreamClassification
from timm.models.vision_transformer import Block
from utils.variablewatcher import VariableWatcher
from torchvision import transforms
from utils.image_utils import plot_image, plot_grid_images, save_grid_images
import os
from typing import Iterator
import math


class VITFC(nn.Module):

    def __init__(self, model, num_classes, pretext_task=None, img_size=224, patch_size=16, mask_ratio=0.15, pretrained=True, in_chans=3, decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,mlp_ratio=4., norm_layer=nn.LayerNorm, downstream_loss=None, debug_images = False):
        super(VITFC, self).__init__()
        self.vit = model
        self.starting_head = self.vit.head
        self.starting_patch_embed = self.vit.patch_embed
        self.vit.num_classes = num_classes

        self.embedding_size = model.embed_dim
        self.patch_size = patch_size
        self.num_patches = (img_size // self.patch_size) ** 2
        self.custom_patch_embed = None
        self.patch_position_predictor = None
        self.custom_order = None
        self.custom_order_tensor = None
        self.pretext_loss = None
        self.patch_embedding = None
        self.optimizer = None

        self.downstream_loss = nn.CrossEntropyLoss()
        self.downstream_task = None

        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.in_chans = in_chans
        self.image_output_directory = "output_images"

        self.debug_images = debug_images

        
        if not os.path.exists(self.image_output_directory):
            os.makedirs(self.image_output_directory)

        # self.heads_size = self.vit.head.size

        # Rotation
        self.rotation_angles = [0, 90, 180, 270]
        self.image_rotation_angles = self.rotation_angles

        # self.image_rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]
        self.image_rotation_labels = None


        # Masking
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.vit.embed_dim))
        self.mask_ratio = mask_ratio
        self.img_size = img_size


        self.pretext_train = False
        self.pretext_task = pretext_task
        self.pretext_head = None

        self.patch_rotation_head = None
        self.patch_mask_head = None
        self.patch_order_head = None
        self.image_rotation_head = None

        print ( "Created VITFC model %s optimizer %s" %( hex(id(self)), hex(id(self.optimizer)) ) )

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        modules = nn.ModuleList()
        modules.add_module("vit", self.vit)
        if self.pretext_train == False:
            modules.add_module("head", self.vit.head)
        # modules.add_module("vit_head", self.vit.head)

        parameters = modules.parameters(recurse)
        # parameters = self.vit.parameters(recurse)
        # for param in parameters:
        #     yield param 
        return parameters
    
    def prepare_masking(self):
        if self.pretext_task == "patch_mask":
            self.vit.head = nn.Identity()  # Rimuove il classificatore finale

            # Decoder
            if self.decoder_embed_dim != self.vit.embed_dim:
                self.decoder_embed_dim = self.vit.embed_dim
                self.decoder_embed = nn.Linear(self.vit.embed_dim, self.decoder_embed_dim, bias=True)
                self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))
                self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.decoder_embed_dim))
                self.decoder_blocks = nn.ModuleList([
                    Block(self.decoder_embed_dim, self.decoder_num_heads, self.mlp_ratio, qkv_bias=True, norm_layer=self.norm_layer)
                    for _ in range(self.decoder_depth)
                ])
                self.decoder_norm = self.norm_layer(self.decoder_embed_dim)
                self.decoder_pred = nn.Linear(self.decoder_embed_dim, self.patch_size * self.patch_size * self.in_chans, bias=True)
            self.pretext_loss = PatchMaskLoss(self.num_patches, self.decoder_embed_dim)
            self.pretext_head = self.decoder_pred
            self.vit.patch_size = self.patch_size
            self.vit.patch_embed = self.starting_patch_embed


    def prepare_patch_ordering(self):
        if self.custom_patch_embed is None:
            self.custom_order = np.random.permutation(self.num_patches)
            # custom_order = list(range(self.num_patches))[::-1]

            # Crea il modulo di embedding delle patch personalizzato
            self.custom_patch_embed = CustomPatchEmbed(
                img_size=self.img_size,
                patch_size=self.patch_size,
                in_channels=3,
                embed_dim=self.embedding_size,
                custom_order=self.custom_order
            )
            self.custom_patch_embed.to("cuda")

            self.patch_position_predictor = nn.Linear(self.vit.head_hidden_size, self.patch_size)
        else:
            self.custom_order = np.random.permutation(self.num_patches)

        self.custom_patch_embed.custom_order = self.custom_order
        print ( "Patch custom order: ", self.custom_order )
        self.vit.patch_embed = self.custom_patch_embed
        self.pretext_head = self.patch_position_predictor
        self.pretext_loss = PatchOrderLoss(self.num_patches, self.vit.head_hidden_size)
        self.vit.patch_size = self.patch_size

        return self.custom_patch_embed
    
    def prepare_patch_rotation(self):
        if self.patch_rotation_head is None:
            num_classes = len(self.rotation_angles) * self.num_patches
            self.patch_rotation_head = nn.Linear(self.vit.embed_dim, num_classes).to("cuda")
        # self.vit.head = self.patch_rotation_head
        self.pretext_head = self.patch_rotation_head
        self.head = self.patch_rotation_head 
        self.vit.patch_embed = self.starting_patch_embed

        self.pretext_loss = nn.CrossEntropyLoss()

    def prepare_image_rotation(self):
        if self.image_rotation_head is None:
            num_classes = len(self.image_rotation_angles)
            self.image_rotation_head = nn.Linear(self.vit.embed_dim, num_classes).to("cuda")
        # self.vit.head = self.image_rotation_head
        self.pretext_head = self.image_rotation_head
        self.head = self.image_rotation_head
        self.vit.patch_embed = self.starting_patch_embed

        self.pretext_loss = nn.CrossEntropyLoss().to("cuda")


    def loss(self):
        if self.pretext_train:
            if self.pretext_task == "patch_mask":
                loss = self.pretext_loss(self.patches, self.preds, self.mask).to("cuda")
            if self.pretext_task == "patch_ordering":
                loss = self.pretext_loss(self.patch_embedding).to("cuda")
            if self.pretext_task == "patch_rotation":
                loss = self.pretext_loss(self.output,self.patch_rotation_labels).to("cuda")
            if self.pretext_task == "image_rotation":
                loss = self.pretext_loss(self.output,self.image_rotation_labels).to("cuda")
        elif self.downstream_loss is not None:
            loss = self.downstream_loss(self.patch_embedding).to("cuda")
        return loss

    def forward(self, x):
        self.output = None
        if self.pretext_train:
            if self.pretext_task == "patch_mask":
                # if not isinstance(self.pretext_loss, PatchMaskLoss):
                #     self.prepare_masking()
                self.output = self.forward_mask(x)
            elif self.pretext_task == "patch_ordering":
                # if not isinstance(self.pretext_loss, PatchOrderLoss):
                #     self.custom_patch_embed = self.prepare_patch_ordering()
                #     self.vit.patch_embed = self.custom_patch_embed
                self.output =  self.forward_patch_ordering(x)
            elif self.pretext_task == "patch_rotation":
                # if not isinstance(self.pretext_loss, PatchOrderLoss):
                #     self.custom_patch_embed = self.prepare_patch_ordering()
                #     self.vit.patch_embed = self.custom_patch_embed
                self.output =  self.forward_patch_rotation(x)
            elif self.pretext_task == "image_rotation":
                self.output =  self.forward_image_rotation(x)
        else: 
            # self.vit.head = self.starting_classifier
            self.output = self.forward_classify(x)
        return self.output
    
    def forward_image_rotation(self, x):

        images, self.image_rotation_labels = self.rotate_images(x)
        self.save_images(x, images, max_saved=1)

        output = self.vit(images)
        # logits = self.vit.head(patches)

        # B, num_patches, num_classes = logits.shape
        # outputs = logits.permute(0, 2, 1)
        return output
    
    def forward_patch_rotation(self, x):

        patches, self.patch_rotation_labels = self.rotate_patches(x)
        self.save_images(x, patches,1)

        x = patches

        B = x.shape[0]
        x = self.vit.patch_embed(x)
        x = x + self.vit.pos_embed[:, 1:, :]
        cls_tokens = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.vit.pos_embed[:, :1, :]
        x = self.vit.pos_drop(x)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        patch_tokens = x[:, 1:, :].to(x.device)
        self.head = self.head.to(x.device)
        logits = self.head(patch_tokens)

        # B, num_patches, num_classes = logits.shape
        outputs = logits.permute(0, 2, 1)
        return outputs

    def forward_classify(self, x):
        x = self.vit(x)
        return x

    def forward_patch_ordering(self, x):
        images = x
        x, order = self.shuffle_patches(x, self.custom_order)

        B, C, H, W = x.size()
        x = self.vit.patch_embed(x)  # [B, num_patches, embed_dim]
        
        # Riordina le patch se custom_order è definito
        if self.custom_order is not None:
            self.custom_order_tensor = torch.tensor(self.custom_order).to(x.device)
            # x = x[:, self.custom_order_tensor, :]
            # Riordina anche gli embeddings posizionali
            pos_embed = self.vit.pos_embed[:, 1:, :][:, self.custom_order_tensor, :]
        else:
            pos_embed = self.vit.pos_embed[:, 1:, :]
        

        # Aggiungi l'embedding posizionale
        cls_token = self.vit.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = x + pos_embed
        x = torch.cat((cls_token, x), dim=1)  # [B, 1 + num_patches, embed_dim]
        
        # Passa attraverso i Transformer Encoder layers
        x = self.vit.pos_drop(x)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        
        self.patch_embedding = x[:, 1:].to(x.device)  # Ignora il token di classificazione
        # logits = self.vit.head(x[:, 0])
        return x

    def forward_mask(self, x):
        # Encoder
        x_encoded, mask, ids_restore = self.forward_encoder_mask(x, self.mask_ratio)

        # Decoder
        preds = self.forward_decoder_mask(x_encoded, ids_restore)  # Reconstructed patches

        B, C, H, W = x.shape

        # Correctly retrieve patch_size
        if isinstance(self.vit.patch_embed.patch_size, tuple):
            patch_size = self.vit.patch_embed.patch_size[0]  # Assuming square patches
        else:
            patch_size = self.patch_embed.patch_size

        # Extract patches from images
        patches = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.contiguous().view(B, C, -1, patch_size, patch_size)
        patches = patches.permute(0, 2, 1, 3, 4)  # [B, num_patches, C, patch_size, patch_size]
        patches = patches.reshape(B, -1, C * patch_size * patch_size)  # [B, num_patches, patch_dim]

        target = patches  # [B, num_patches, patch_dim]

        # Print shapes for debugging
        # print(f"patch_size: {patch_size}")
        # print(f"preds shape: {preds.shape}")
        # print(f"target shape: {target.shape}")
        # print(f"mask shape: {mask.shape}")
        # Compute loss
        # loss = self.pretext_loss(patches, preds, mask)
        self.patches = patches
        self.mask = mask
        self.preds = preds
        return preds



    # def _mask_patches(self, patches):
    #     """Applica il mascheramento a una percentuale delle patch."""
    #     batch_size, num_patches, _ = patches.shape
    #     num_masked = int(self.mask_ratio * num_patches)
        
    #     # Maschera casualmente le patch
    #     mask_indices = np.random.choice(num_patches, num_masked, replace=False)
    #     visible_indices = np.array([i for i in range(num_patches) if i not in mask_indices])
    #     mask = torch.ones(patches.shape, dtype=torch.bool)
    #     mask[:, mask_indices, :] = False
        
    #     masked_patches = patches[:, mask_indices, :]
    #     visible_patches = patches[:, visible_indices, :]
        
    #     return masked_patches, mask_indices, visible_patches

    # def _calculate_loss(self, masked_patches, predicted_patches):
    #     """Calcola la loss tra le patch mascherate originali e quelle predette."""
    #     loss = F.mse_loss(predicted_patches, masked_patches)
    #     return loss
    

    def forward_decoder_mask(self, x, ids_restore):
        # Embed tokens
        self.decoder_embed = self.decoder_embed.to(x.device)
        x = self.decoder_embed(x)

        # Append mask tokens to the sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1).to(x.device)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # Exclude cls token

        # Unshuffle
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])).to(x.device)

        # Add positional encoding
        x_ = x_ + self.decoder_pos_embed[:, 1:, :].to(x.device)

        # Add cls token
        cls_token = x[:, :1, :]
        x_ = torch.cat([cls_token, x_], dim=1)
        x_ = x_ + self.decoder_pos_embed[:, :1, :].to(x.device)

        # Apply Transformer blocks
        for blk in self.decoder_blocks:
            blk.to(x.device)
            x_ = blk(x_)
        self.decoder_norm = self.decoder_norm.to(x.device)
        x_ = self.decoder_norm(x_)

        # Predict pixels
        self.decoder_pred = self.decoder_pred.to(x.device)
        x_rec = self.decoder_pred(x_)  # [B, num_patches, patch_dim]

        # Remove cls token
        x_rec = x_rec[:, 1:, :]  # [B, num_patches, patch_dim]

        return x_rec
    
    def forward_encoder_mask(self, x, mask_ratio):
        # Embed patches
        x = self.vit.patch_embed(x)  # [B, num_patches, embed_dim]

        # Add positional encoding
        x = x + self.vit.pos_embed[:, 1:, :]

        # Apply masking
        x_masked, mask, ids_restore = self.random_masking(x, mask_ratio)

        # Add class token
        cls_token = self.vit.cls_token + self.vit.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x_masked.shape[0], -1, -1)
        x_masked = torch.cat((cls_tokens, x_masked), dim=1)

        # Apply Transformer blocks
        x_masked = self.vit.pos_drop(x_masked)
        for blk in self.vit.blocks:
            x_masked = blk(x_masked)
        x_masked = self.vit.norm(x_masked)

        return x_masked, mask, ids_restore

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        """
        B, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

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
    
    @property
    def pretext_train(self):
        return self._pretext_train
    
    @pretext_train.setter
    def pretext_train(self, new_value):
        print ( "Setting pretext train to %s" % new_value )
        self._pretext_train = new_value
        self.pretext_train_change_callback(new_value)
    
    def pretext_train_change_callback(self, new_value):
        self._pretext_train = new_value
        print(f"Pretext training set to: {new_value}")
        if not self._pretext_train:
            self.vit.head = self.starting_head
            if self.downstream_task != None:
                self.vit.head = self.downstream_task
            self.loss = self.downstream_loss
    

    @property
    def pretext_task(self):
        return self._pretext_task
    
    @pretext_task.setter
    def pretext_task(self, new_value):
        print ( "Setting pretext task to %s" % new_value )
        self._pretext_task = new_value
        self.pretext_task_change_callback(new_value)


    def pretext_task_change_callback(self, new_value):
        self._pretext_task = new_value
        print(f"Pretext task callback: {new_value}")
        if self._pretext_task == "patch_mask":
            self.prepare_masking()
        elif self._pretext_task == "patch_ordering":
            self.prepare_patch_ordering()
        elif self._pretext_task == "patch_rotation":
            self.prepare_patch_rotation()
        elif self._pretext_task == "image_rotation":
            self.prepare_image_rotation()
        else:
            print ( "Pretext task not recognized" )

    def rotate_images(self, imgs):
        B, C, H, W = imgs.shape


        device = imgs.device
        num_patches_per_row = H // self.patch_size
        num_patches = num_patches_per_row ** 2
        labels = torch.zeros(B, dtype=torch.long, device=device)
        rotated_imgs = torch.zeros_like(imgs).to(imgs.device)

        a = transforms
        for b in range(B):

            angle = np.random.choice(self.image_rotation_angles)
            angle_index = self.image_rotation_angles.index(angle)
            labels[b] = angle_index
            img = imgs[b]
            rotated = transforms.functional.rotate(img, angle.item()).to(device)
            rotated_imgs[b] = rotated
        return rotated_imgs, labels
    
    def shuffle_patches(self, images, order):
        B, C, H, W = images.shape
        # num_patches_per_row = int(math.sqrt(N))
        # device = patches.device

        # shuffled_batches = []

        # shuffled_batches = patches[:, -1,:][:,order,:]
        num_patches_per_row = H // self.patch_size
        patches = self._patchify(images)
        shuffled_batches = patches[::][:, order, :]
        reconstructed_imgs = []
        self.save_images(images, shuffled_batches, max_saved=1)
        for b in range(shuffled_batches.shape[0]):
            reconstructed_img = self.reconstruct_image_from_patches(shuffled_batches[b], num_patches_per_row, self.patch_size)
            
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
        return reconstructed_imgs, order

    def rotate_patches(self, imgs):
        B, C, H, W = imgs.shape
        device = imgs.device
        num_patches_per_row = H // self.patch_size
        num_patches = num_patches_per_row ** 2
        labels = torch.zeros(B, num_patches, dtype=torch.long, device=device)
        rotated_imgs = torch.zeros_like(imgs)

        for b in range(B):
            img = imgs[b]
            patches = img.unfold(1, self.patch_size, self.patch_size).unfold(2, self.patch_size, self.patch_size)
            patches = patches.contiguous().view(C, -1, self.patch_size, self.patch_size)
            rotated_patches = []
            for i, patch in enumerate(patches.transpose(0, 1)):
                angle = np.random.choice(self.rotation_angles)
                rotated_patch = transforms.functional.rotate(patch, angle.item()).to(device)
                rotated_patches.append(rotated_patch.unsqueeze(0))
                labels[b, i] = self.rotation_angles.index(angle)
            rotated_patches = torch.cat(rotated_patches, dim=0)
            reconstructed_img = self.reconstruct_image_from_patches(rotated_patches, num_patches_per_row, self.patch_size)
            rotated_imgs[b] = reconstructed_img

        return rotated_imgs, labels
    
    def _patchify(self, x):
        """Divide l'immagine in patch."""
        p = self.patch_size
        batch_size, channels, img_h, img_w = x.shape
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(batch_size, channels, -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4)
        # patches = patches.permute(0, 2, 1, 3, 4).reshape(batch_size, -1, p * p * channels)
        return patches
    
    def reconstruct_image_from_patches(self, patches, num_patches_per_row, patch_size):
        C = patches.shape[1]
        image = patches.view(num_patches_per_row, num_patches_per_row, C, patch_size, patch_size)
        image = image.permute(2, 0, 3, 1, 4).contiguous()
        image = image.view(C, num_patches_per_row * patch_size, num_patches_per_row * patch_size)
        return image
    
    def save_images ( self, images, patches, max_saved = 0 ):

        if self.debug_images:

            B, C, H, W = images.shape
            # saved_order = np.random(0, ) 
            num_patches_per_row = H // self.patch_size
            recontructed_patches = []
            count = 0
            for batch_id in range(B):
                reconstructed = patches[batch_id]
                if ( patches[batch_id].shape[1] != images[batch_id].shape[1] ):
                    reconstructed = self.reconstruct_image_from_patches(patches[batch_id], num_patches_per_row, self.patch_size)
                output_filename = os.path.join(self.image_output_directory, f"reconstructed_{batch_id}.png")
                save_grid_images([images[batch_id], reconstructed], nrow=2, output_path=output_filename)
                recontructed_patches.append(reconstructed)
                count += 1
                if max_saved != 0 and count >= max_saved:
                    break 