from torch import nn
import torch

from timm.models.vision_transformer import VisionTransformer
from transformers import ViTModel

class Downstream (nn.Module):
    def __init__(self, backbone, cls_token_only = True):
        super(Downstream, self).__init__()

        self.downstream_head = nn.Identity()

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
    

