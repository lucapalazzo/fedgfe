from typing import Iterator
from torch import nn
from torch.optim import SGD
import copy

class FLModel(nn.Module):
    def __init__(self, args, model_id, model = None) -> None:
        super(FLModel,self).__init__()
        self.id = model_id
        if model != None:
            self.inner_model = model
        else:
            self.inner_model = copy.deepcopy(args.model)

        self.loss = None
        # print ( "Optimizer: ", hex(id(self.inner_model.optimizer)))
        # self.optimizer = self.inner_model.optimizer
        # self.pretext_task = None
        self.pretext_train = False
        self.downstream_task = None

    def to(self, device):
        self.device = device
        self.inner_model.to(device)
        return self

    def forward(self,x):
        output = self.inner_model(x)
        return output
    
    def loss(self, x, Y = None):
        return self.inner_model.loss(x, Y)
    
    def accuracy(self, x, Y = None):
        return self.inner_model.accuracy(x, Y)
    
    def test_metrics_calculate(self, x, Y = None, round = None):
        return self.inner_model.test_metrics_calculate(x, Y, round=round)
    
    def test_metrics_log(self, round = None):
        return self.inner_model.test_metrics_log(round=round)
    
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
    

    @property
    def pretext_task_name(self):
        return self.inner_model.pretext_task_name
    
    @pretext_task_name.setter
    def pretext_task_name(self, pretext_task):
        self.inner_model.pretext_task_name = pretext_task
    
    @property
    def pretext_train(self):
        return self.inner_model.pretext_train
    
    @pretext_train.setter
    def pretext_train(self, pretext_train):
        self.inner_model.pretext_train = pretext_train

    def downstream_task_set(self, new_downstream_task):
        self.downstream_task = new_downstream_task
        self.inner_model.downstream_task = new_downstream_task

    @property
    def backbone(self):
        if self.inner_model != None and hasattr(self.inner_model, "backbone"):
            return self.inner_model.backbone
        else:
            return None
        
    @property
    def round(self):
        return self.inner_model.round
    
    @round.setter
    def round(self, round):
        self.inner_model.round = round