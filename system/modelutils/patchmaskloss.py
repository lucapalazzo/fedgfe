import torch
from torch import nn

class PatchMaskLoss(nn.Module):
    def __init__(self, num_patches, hidden_dim):
        super(PatchMaskLoss, self).__init__()

        # Use Mean Squared Error Loss without reduction (we'll handle reduction manually)
        self.loss_fn = nn.MSELoss(reduction='none')
    
    def forward(self, preds, targets, mask):
        """
        imgs: original images
        preds: reconstructed patches
        mask: mask indicating which patches were masked
        """
        # Target: flatten the images to patches
        
        loss = self.loss_fn(preds, targets).mean(dim=-1)  # [B, num_patches]
        

        # Apply mask to compute the mean loss over masked patches only
        loss = (loss * mask).sum() / mask.sum()
        return loss