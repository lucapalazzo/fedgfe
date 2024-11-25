from typing import Iterator
from torch import nn
from torch.optim import SGD
import copy

class FLModel(nn.Module):
    def __init__(self,args, model_id) -> None:
        super(FLModel,self).__init__()
        self.id = model_id
        self.inner_model = copy.deepcopy(args.model)
        self.loss = None
        # self.inner_model.optimizer = SGD(self.inner_model.parameters(), lr=0.01)
        print ( "Optimizer: ", hex(id(self.inner_model.optimizer)))
        self.optimizer = self.inner_model.optimizer
        self.pretext_task = None
        self.pretext_train = False

    def to(self, device):
        self.device = device
        self.inner_model.to(device)
        return self

    def forward(self,x):
        # self.inner_model.pretext_task = self.pretext_task
        # self.inner_model.pretext_train = self.pretext_train
        output = self.inner_model(x)
        return output
    
    def train(self):
        self.inner_model.pretext_train = self.pretext_train
        return self.inner_model.train()

    def eval(self):
        return self.inner_model.eval()
    
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        return self.inner_model.parameters(recurse)
    
    @property
    def pretext_task(self):
        return self.inner_model.pretext_task
    
    @pretext_task.setter
    def pretext_task(self, pretext_task):
        self.inner_model.pretext_task = pretext_task
    
    @property
    def pretext_train(self):
        return self.inner_model.pretext_train
    
    @pretext_train.setter
    def pretext_train(self, pretext_train):
        self.inner_model.pretext_train = pretext_train