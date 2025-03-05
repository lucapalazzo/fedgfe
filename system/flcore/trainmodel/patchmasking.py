import os
from torch import nn
import torch
from flcore.trainmodel.patchpretexttask import PatchPretextTask
from torchvision import transforms
from typing import Iterator
import numpy as np

from timm.models.vision_transformer import Block
from transformers import ViTMAEModel, ViTMAEForPreTraining, ViTMAEConfig, ViTModel, AutoImageProcessor, ViTFeatureExtractor
import wandb


class PatchMasking (PatchPretextTask):
    pretext_task_name = "patch_masking"
    def __init__(self, backbone=None, input_dim = 768, output_dim = 768, debug_images=False, img_size=224, patch_size=-1, patch_count = -1, mask_ratio = 0.15):
        super(PatchMasking, self).__init__(backbone=backbone, input_dim = input_dim, output_dim = output_dim, debug_images=debug_images, img_size=img_size, patch_size=patch_size, patch_count = patch_count)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = PatchMasking.pretext_task_name
        # self.loss = PatchOrderLoss(self.num_patches, self.head_hidden_size)

        self.mask_ratio = mask_ratio
        self.masked_count = int(self.mask_ratio * self.patch_count)
        self.masked_indices = []
        self.pretext_loss = nn.MSELoss(reduction='none')
        if isinstance(self.backbone, ViTModel):
            self.mask_model_config = ViTMAEConfig()
            self.mask_model_config.hidden_size = input_dim
            self.mask_model_config.num_channels = 3
            self.mask_model_config.mask_ratio = mask_ratio
            self.mask_model_config.patch_size = patch_size
            self.mask_model_config.num_hidden_layers = backbone.config.num_hidden_layers
            self.mask_model_config.num_patches = patch_count
            self.mask_feature_extractor = ViTFeatureExtractor()
            
            self.mask_model = ViTMAEForPreTraining(self.mask_model_config).to(self.device)
            # self.mask_model.vit.encoder = self.backbone.encoder
            self.pretext_head = self.mask_model.decoder
            # self.mask_image_processor = AutoImageProcessor.from_pretrained("vit-mae-base")
            # self.mask_model.encoder = self.backbone.encoder
        else:
            self.pretext_head = nn.Sequential( nn.Linear(self.output_dim, self.masked_count * self.patch_size * 3 ) ).to(self.device)

    @staticmethod
    def define_metrics( metrics_path = None ):
        pretext_task_name = PatchMasking.pretext_task_name

        path = "/" if metrics_path is None else "/"+metrics_path+"/"
        metrics = []
        metrics.append(f"train{path}pretext_train_loss_{pretext_task_name}")
        metrics.append(f"test{path}pretext_train_ds_loss_{pretext_task_name}")
        metrics.append(f"test{path}pretext_test_acc_{pretext_task_name}")

        for metric in metrics:
           a = wandb.define_metric(metric, step_metric="ssl_round")

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        modules = nn.ModuleList()
        if self.mask_model.vit.encoder != self.backbone.encoder:
            modules.add_module("mask_model", self.mask_model)
        else:
            modules.add_module("pretext_head", self.pretext_head)
        # modules.add_module("mae_decoder", self.mask_model.decoder)
        parameters = modules.parameters(recurse)
        return parameters

    def accuracy(self, x, target):
        return 0
        return super().accuracy(x, target)
    
    def preprocess_sample(self, x):
        
        # images, self.masked_indices = self.random_masking(x, self.mask_ratio) 
        # self.save_images(x, images, max_saved=1)
        return x 
    
    def loss(self, x, y = None):
        loss = x.loss
        return loss
    
    def forward(self, x):
        if self.debug_images:
            original_images = x.clone()
        # x = self.preprocess_sample(x)

        if isinstance(self.backbone, ViTModel):
            # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
            # image = Image.open(requests.get(url, stream=True).raw)
            # inputs = self.image_processor(images=image, return_tensors="pt")
            # inputs = inputs.to(self.device)
            original_images = x.clone() 
            x = self.mask_model(x)

            # y = original_images
            y = self.mask_model.unpatchify(x.logits)
            # y = torch.einsum('nchw->nhwc', y).detach().cpu()

            pixel_values = x.logits.detach().cpu()
            # pixel_values = inputs['pixel_values'].detach().cpu()
            mask = x.mask.detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, self.mask_model.config.patch_size**2 *3)  # (N, H*W, p*p*3)
            mask = self.mask_model.unpatchify(mask)  # 1 is removing, 0 is keeping
            # mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    
            # x = torch.einsum('nchw->nhwc', pixel_values)

    # masked image
            im_masked = original_images * (1 - mask)

    # MAE reconstruction pasted with visible patches
            im_paste = original_images * (1 - mask) + y * mask
            images = [ im_masked, im_paste]

            # x = self.mask_model(x[:,0:1,:,:])
        elif self.backbone is not None:
            # self.backbone.logits_only = True
            x = self.backbone(x)
            x = self.pretext_head(x)
        if self.debug_images or os.path.exists("save_debug_images"):
            # if images.shape[1] == 1:
            #     images = images.repeat(1, 3, 1, 1)
            # images = torch.einsum('nchw->nhwc', images)
            save_images = [original_images] + images
            self.save_images(save_images, None, max_saved=1)
        return x
    
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