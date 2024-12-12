from torch import nn
import torch

class Downstream (nn.Module):
    def __init__(self, backbone):
        super(Downstream, self).__init__(  )

        self.downstream_head = nn.Identity()

        self.backbone = backbone
        self.backbone_enabled = True if backbone is not None else False

    def parameters(self, recurse = True):
        print ( "Warning: Downstream parameters not implemented" )
        return None
        
    def forward(self, x):
        if self.backbone is not None:
            x = self.backbone(x)
            # prende solo il cls token
            # x = x[:,0,:]
        return x
    
    def backbone_forward(self,x):
        return self.backbone.backbone_forward(x)
    

