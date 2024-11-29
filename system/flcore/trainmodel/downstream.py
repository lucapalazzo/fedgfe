from torch import nn
import torch

class DownstreamClassification (nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DownstreamClassification, self).__init__()

        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.fc = nn.Linear(input_dim, 512).to(self.device)
        # self.relu1 = nn.ReLU().to(self.device)
        # self.fc1 = nn.Linear(512, 64).to(self.device)
        # self.relu2 = nn.ReLU().to(self.device)
        # self.fc2 = nn.Linear(64, output_dim).to(self.device)

    # def to(self, device):
    #     self.device = device
    #     self.fc.to(device)
    #     self.relu1.to(device)
    #     self.fc1.to(device)
    #     self.relu2.to(device)

    #     self.fc2.to(device)


    #     return self

    def forward(self, x):
        # x = self.fc(x)
        # x = self.relu1(x)
        # x = self.fc1(x)
        # x = self.relu2(x)
        # x = self.fc2(x)

        return x
