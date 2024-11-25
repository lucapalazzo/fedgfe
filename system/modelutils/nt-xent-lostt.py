import torch
from torch import nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, num_patches, hidden_dim):
        super(NTXentLoss, self).__init__()

        # Use Mean Squared Error Loss without reduction (we'll handle reduction manually)
        self.loss_fn = nn.MSELoss(reduction='none')
    
    def forward(self, z_i, z_j, temperature=0.5):
        # Normalize embeddings to unit sphere
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate along batch dimension
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = torch.mm(representations, representations.T)

        # Create similarity mask
        batch_size = z_i.shape[0]
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = labels.to(device)
        mask = torch.eye(batch_size * 2, dtype=torch.bool).to(device)
        
        # Compute similarity scores and apply mask
        similarity_matrix = similarity_matrix / temperature
        similarity_matrix.masked_fill_(mask, -1e9)  # Mask diagonal elements

        # Calculate loss using cross-entropy
        positives = torch.diag(similarity_matrix, batch_size)
        negatives = similarity_matrix[~mask].view(batch_size * 2, -1)
        logits = torch.cat((positives.view(-1, 1), negatives), dim=1)
        
        loss = F.cross_entropy(logits, labels)
        return loss