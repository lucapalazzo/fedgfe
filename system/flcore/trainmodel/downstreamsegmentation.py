from flcore.trainmodel.downstream import Downstream
import torch.nn as nn
import torch
from flcore.trainmodel.unet import UNet
from timm.models.vision_transformer import VisionTransformer
from transformers import pipeline


from transformers import ViTModel

from flcore.trainmodel.TransUNet.networks.vit_seg_modeling import VisionTransformer as ViT_seg
from flcore.trainmodel.TransUNet.networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

from torch.nn.modules.loss import CrossEntropyLoss
from flcore.trainmodel.TransUNet.utils import DiceLoss


class DownstreamSegmentation (Downstream):
    def __init__(self, backbone, num_classes=10, cls_token_only=False, img_size=224, patch_count=-1, patch_size=-1):
        super(DownstreamSegmentation, self).__init__(backbone, cls_token_only=cls_token_only, img_size=img_size, patch_count=patch_count, patch_size=patch_size)

        input_dim = 0

        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        # config_vit = CONFIGS_ViT_seg['ViT-B_16']
        config_vit.n_classes = num_classes
        config_vit.n_skip = 3
        config_vit.pretrained_path = None
        config_vit.patches.grid = [ int(self.patch_count**0.5), int(self.patch_count**0.5) ]
        # config_vit.patches.grid = None
        config_vit.patches.size = [ self.patch_size, self.patch_size ]
        config_vit.patch_size = backbone.config.patch_size
        config_vit.hidden_size = backbone.config.hidden_size
        config_vit.transformer.num_heads = backbone.config.num_attention_heads
        config_vit.transformer.num_layers = backbone.config.num_hidden_layers

        self.vitseg = ViT_seg(config_vit, img_size=224, num_classes=config_vit.n_classes)
        if isinstance(backbone, VisionTransformer):
            input_dim = backbone.embed_dim
        elif isinstance(backbone, ViTModel):
            input_dim = backbone.config.hidden_size
            # self.segmentation = self.create_segmentation_pipeline()
            # self.vitseg.encoder = backbone

        output_dim = num_classes

        # self.downstream_head = nn.Sequential(
        #     UNet(input_dim, num_classes),
        # )

        self.loss = self.segmentation_loss
        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes)

    def segmentation_loss ( self, logits, labels ):
        return torch.Tensor([1])
        loss_ce = self.ce_loss(logits, labels)
        loss_dice = self.dice_loss(logits, labels)
        return loss_ce + loss_dice
    
    def create_segmentation_pipeline(self):
        segmentation = pipeline("image-segmentation", "facebook/maskformer-swin-base-coco")
        return segmentation
    

    def parameters(self, recurse=True):
        moduleList = nn.ModuleList()
        moduleList.add_module("downstream_head", self.downstream_head)
        return moduleList.parameters(recurse)

    def forward(self, x):

        # if self.backbone is not None:
        #     backbone_x = self.backbone(x)

        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.vitseg.transformer(x)  # (B, n_patch, hidden)
        x = self.vitseg.decoder(x, features)
        logits = self.vitseg.segmentation_head(x)
        return logits