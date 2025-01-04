from flcore.trainmodel.downstreamclassification import DownstreamClassification
import torch.nn as nn

class DownstreamFiveLayerClassification (DownstreamClassification):
    def __init__(self, backbone, num_classes=10):
        super(DownstreamFiveLayerClassification, self).__init__(backbone)

        self.downstream_head = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
        )
