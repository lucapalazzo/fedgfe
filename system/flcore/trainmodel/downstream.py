from torch import nn
import torch

from timm.models.vision_transformer import VisionTransformer
from transformers import ViTModel

class Downstream (nn.Module):
    def __init__(self, backbone, cls_token_only = True, img_size = 224, patch_count = -1, patch_size = -1):
        super(Downstream, self).__init__()

        self.downstream_head = nn.Identity()

        self.img_size = img_size

        if patch_count > 0:
            self.patch_count = int(patch_count)
            self.patch_size = int(img_size // (patch_count ** 0.5))
        elif patch_size > 0:
            self.patch_size = int(patch_size)
            self.patch_count = int((img_size // patch_size) ** 2)

        self.backbone = backbone
        self.backbone_enabled = True if backbone is not None else False
        self.cls_token_only = cls_token_only

    def parameters(self, recurse = True):
        print ( "Warning: Downstream parameters not implemented" )
        return None
        
    def forward(self, x):
        if self.backbone is not None:
            x = self.backbone(x)

        if self.cls_token_only:
            # prende solo il cls token
            if isinstance(self.backbone, VisionTransformer):
                x = x[:,0,:]
            elif isinstance(self.backbone, ViTModel):
                x = x.last_hidden_state[:,0,:]
            # x = x[:,0,:]
        return x
    
    def backbone_forward(self,x):
        return self.backbone.backbone_forward(x)
    

