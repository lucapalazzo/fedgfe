from flcore.trainmodel.downstream import DownstreamClassification
import torch.nn as nn

class DownstreamClassification (nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DownstreamClassification, self).__init__()

        self.fc = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.fc1 = nn.Linear(512, 64)
        self.relu2 = nn.ReLU()
        self.fc2 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu1(x)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)

        return x
