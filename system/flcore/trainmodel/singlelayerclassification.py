from flcore.trainmodel.downstream import DownstreamClassification

import torch.nn as nn

class SingleLayerClassification(DownstreamClassification):
    def __init__(self, input_dim, output_dim):
        super(SingleLayerClassification, self).__init__(input_dim, output_dim)

        self.fc = nn.Sequential (
            nn.Linear(input_dim, output_dim)
        )
        # self.relu = nn.ReLU().to(self.device)

    def forward(self, x):
        x = self.fc(x)
        # x = self.relu(x)

        return x

