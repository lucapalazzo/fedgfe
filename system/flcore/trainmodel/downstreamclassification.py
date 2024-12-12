from flcore.trainmodel.downstream import Downstream
import torch.nn as nn

class DownstreamClassification (Downstream):
    def __init__(self, backbone, num_classes=10):
        super(DownstreamClassification, self).__init__(backbone)

        input_dim = backbone.embed_dim
        output_dim = num_classes

        self.downstream_head = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

        self.loss = nn.CrossEntropyLoss()
    
    def parameters(self, recurse=True):
        moduleList = nn.ModuleList()
        moduleList.add_module("downstream_head", self.downstream_head)
        return moduleList.parameters(recurse)

    def forward(self, x):
        x = super().forward(x)
        x = self.downstream_head(x)

        return x
    
    def backbone_forward(self, x):
        return super().backbone_forward(x)
