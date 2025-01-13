from flcore.trainmodel.downstream import Downstream
import torch.nn as nn

from timm.models.vision_transformer import VisionTransformer
from transformers import pipeline

from transformers import ViTModel

class DownstreamSegmentation (Downstream):
    def __init__(self, backbone, num_classes=10):
        super(DownstreamSegmentation, self).__init__(backbone)

        self.input_dim = 0
        if isinstance(backbone, VisionTransformer):
            self.input_dim = backbone.embed_dim
        elif isinstance(backbone, ViTModel):
            self.input_dim = backbone.config.hidden_size
            self.segmentation = self.create_segmentation_pipeline()
            

        
        self.output_dim = num_classes

        self.downstream_head = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
        )

        self.loss = nn.CrossEntropyLoss()

    def create_segmentation_pipeline(self):
        segmentation = pipeline("image-segmentation", "facebook/maskformer-swin-base-coco")
        return segmentation
    
    def parameters(self, recurse=True):
        moduleList = nn.ModuleList()
        moduleList.add_module("downstream_head", self.downstream_head)
        return moduleList.parameters(recurse)

    def forward(self, x):
        x = self.downstream_head(x)

        return x
    
    def backbone_forward(self, x):
        return super().backbone_forward(x)
