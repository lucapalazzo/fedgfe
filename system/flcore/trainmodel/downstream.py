from torch import nn
import torch

class DownstreamTask (nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DownstreamTask, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.fc = nn.Linear(input_dim, 512).to(self.device)
        self.fc1 = nn.Linear(512, 64).to(self.device)
        self.relu = nn.ReLU().to(self.device)
        self.fc2 = nn.Linear(64, output_dim).to(self.device)

    def to(self, device):
        self.device = device
        self.fc.to(device)
        self.fc1.to(device)
        self.fc2.to(device)
        self.relu.to(device)

        return self

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x
