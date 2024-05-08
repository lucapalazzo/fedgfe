from typing import Iterator
from torch import nn
import copy

class FLModel(nn.Module):
    def __init__(self,args, id) -> None:
        super(FLModel,self).__init__()
        self.id = id
        self.inner_model = copy.deepcopy(args.model)
        self.loss = None
        self.optimizer = None

    def to(self, device):
        self.device = device
        self.inner_model.to(device)
        return self

    def forward(self,x):
        return self.inner_model(x)
    
    def train(self):
        return self.inner_model.train()

    def eval(self):
        return self.inner_model.eval()
    
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        return self.inner_model.parameters(recurse)