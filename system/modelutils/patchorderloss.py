import torch
from torch import nn

class PatchOrderLoss(nn.Module):
    def __init__(self, num_patches, hidden_dim):
        super(PatchOrderLoss, self).__init__()
        self.num_patches = num_patches
        self.order_predictor = nn.Linear(hidden_dim, num_patches).to("cuda")
        self.loss_fn = nn.CrossEntropyLoss()
    
    def forward(self, patch_embeddings):
        # patch_embeddings: [B, N, D]
        B, N, D = patch_embeddings.shape
        position_logits = self.order_predictor(patch_embeddings).to(patch_embeddings.device)  # [B, N, num_patches]
        target_positions = torch.arange(N, device=patch_embeddings.device).unsqueeze(0).expand(B, N)  # [B, N]
        # target_positions = target_positions.unsqueeze(0).repeat(B, 1)  # [B, N]

        loss = self.loss_fn(position_logits.view(-1, self.num_patches), target_positions.reshape(-1))
        # loss = self.loss_fn(position_logits.view(-1, self.num_patches), target_positions)
        return loss