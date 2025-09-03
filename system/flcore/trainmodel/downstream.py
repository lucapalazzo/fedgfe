from torch import nn
import torch

from timm.models.vision_transformer import VisionTransformer
from transformers import ViTModel



class Downstream (nn.Module):
    def __init__(self, backbone, cls_token_only = False, img_size = 224, patch_count = -1, patch_size = -1, wandb_log = False, device = None):
        super(Downstream, self).__init__()

        self.task_name = "downstream"
        self.downstream_head = nn.Identity()
        self.round = 0
        self.id = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == None else device


        self.img_size = img_size
        self.wandb_log = wandb_log

        if patch_count > 0:
            self.patch_count = int(patch_count)
            self.patch_size = int(img_size // (patch_count ** 0.5))
        elif patch_size > 0:
            self.patch_size = int(patch_size)
            self.patch_count = int((img_size // patch_size) ** 2)

        self.backbone = backbone
        self.backbone_enabled = True if backbone is not None else False
        self.test_running_round = -1
        self.cls_token_only = cls_token_only
        self.metrics_last_round = -1
        self.metrics_round_epochs_count = 0

        self.task_test_metrics = None
        self.task_test_metrics_aggregated = None
    
    def train(self, mode: bool = True) -> None:
        super(Downstream, self).train(mode)
        if self.backbone is not None:
            self.backbone.train(mode)
    
    def eval(self, mode: bool = True) -> None:
        super(Downstream, self).eval()

    def metrics_reset(self):
        self.test_running_round = -1
        self.init_metrics()

    def define_metrics(self, metrics_path=None):
        self.metrics_path = metrics_path

        return None
    
    def test_metrics_accuracy(self, logits, labels):
        return None
    
    def task_test_metrics(self, logits, labels):
        return None

    def downstream_loss(self, logits, labels):
        assert (False, "Downstream loss not implemented")
        return None
    
    def loss(self, logits, labels, samples = None):
        return self.downstream_loss(logits, labels, samples=samples) if self.downstream_loss is not None else None

    def parameters(self, recurse = True):
        print ( "Warning: Downstream parameters not implemented" )
        return None
        
    def forward(self, x):
        if self.backbone is not None:
            x = self.backbone(x)

        if self.cls_token_only:
            # prende solo il cls token
            if isinstance(self.backbone, VisionTransformer):
                x = x[:,0,:]
            elif isinstance(self.backbone, ViTModel):
                x = x.last_hidden_state[:,0,:]
        else:
            if isinstance(self.backbone, ViTModel):
                x = x.last_hidden_state[:,1:,:]
        return x
    
    def backbone_forward(self,x):
        return self.backbone.backbone_forward(x)
    

