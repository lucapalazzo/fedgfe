from flcore.trainmodel.downstream import Downstream
import torch.nn as nn
from flcore.trainmodel.unet import UNet
from timm.models.vision_transformer import VisionTransformer

from transformers import ViTModel

class DownstreamSegmentation (Downstream):
    def __init__(self, backbone, num_classes=10, cls_token_only=False):
        super(DownstreamSegmentation, self).__init__(backbone, cls_token_only=cls_token_only)

        input_dim = 0
        if isinstance(backbone, VisionTransformer):
            input_dim = backbone.embed_dim
        elif isinstance(backbone, ViTModel):
            input_dim = backbone.config.hidden_size

        output_dim = num_classes

        self.downstream_head = nn.Sequential(
            UNet(input_dim, num_classes),
        )

        self.loss = nn.CrossEntropyLoss()
    
    def parameters(self, recurse=True):
        moduleList = nn.ModuleList()
        moduleList.add_module("downstream_head", self.downstream_head)
        return moduleList.parameters(recurse)

    def forward(self, x):
        x = super().forward(x)
        x = x[:,1:,:] # remove cls token
        embedding_size = x.size(2)
        batch_size = x.shape[0]
        num_patches = int(x.size(1) ** 0.5)

        x = x.permute(0, 2, 1).reshape(batch_size, embedding_size, num_patches, num_patches)
        x = self.downstream_head(x)

        return x
    
    def backbone_forward(self, x):
        return super().backbone_forward(x)
