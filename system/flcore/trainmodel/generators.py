import torch.nn as nn
import torch
import torch.nn.functional as F



class GANGenerator(nn.Module):
    def __init__(self, latent_dim, hidden_dim=512, output_dim=768):
        """
        Generator della GAN.
        Args:
            latent_dim (int): Dimensione del vettore di rumore in input.
            hidden_dim (int): Dimensione degli hidden layer.
            output_dim (int): Dimensione del prompt generato (es. 768).
        """
        super(GANGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, z):
        return self.model(z)


class GANDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=512):
        """
        Discriminatore della GAN.
        Args:
            input_dim (int): Dimensione del prompt in input.
            hidden_dim (int): Dimensione degli hidden layer.
        """
        super(GANDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),

        )
        
    def forward(self, x):
        return self.model(x)
    
def compute_novelty_loss(fake_prompts: torch.Tensor,
                         real_prompts: torch.Tensor,
                         dist_type: str = "l2"
                         ) -> torch.Tensor:
    """    
    Parametri:
      fake_prompts: [M, D] batch di prompt generati
      real_prompts: [N, D] prompt reali raccolti dai client
      dist_type: 'l2' (Euclidea) o 'cosine' (distanza coseno)
    Ritorna:
      novelty_loss (torch.Tensor) - un valore da SOMMARE alla loss del generatore.
    """
    # Controllo dimensioni
    assert fake_prompts.size(1) == real_prompts.size(1), \
        "fake_prompts e real_prompts devono avere la stessa dimensione di feature (D)."

    # Calcolo la distanza pairwise [M,N]
    if dist_type == "l2":
        dists = (fake_prompts.unsqueeze(1) - real_prompts.unsqueeze(0)).pow(2).sum(dim=2).sqrt()
    elif dist_type == "cosine":
        f_norm = F.normalize(fake_prompts, dim=1)  # [M, D]
        r_norm = F.normalize(real_prompts, dim=1)  # [N, D]
        cos_sim = torch.matmul(f_norm, r_norm.transpose(0, 1))  # [M, N]
        dists = 1.0 - cos_sim  # Distanza coseno = 1 - cos_sim
    else:
        raise ValueError(f"dist_type '{dist_type}' non supportato.")


    min_dists, _ = dists.min(dim=1)  # [M]


    novelty_loss = - min_dists.mean()
    return novelty_loss
    

class VAEGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, latent_dim=256, sequence_length=4):
        super(VAEGenerator, self).__init__()
        
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.output_dim = input_dim  # (512)
        self.total_output_dim = sequence_length * input_dim  # 4*512 = 2048
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.total_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, latent_dim * 2)  
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, self.total_output_dim)
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)  
        
        q = self.encoder(x_flat)
        mu, logvar = q[:, :q.size(1)//2], q[:, q.size(1)//2:]
        return mu, logvar
    
    def decode(self, z):
        batch_size = z.size(0)
        decoded_flat = self.decoder(z)
        return decoded_flat.view(batch_size, self.sequence_length, self.output_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        
        q = self.encoder(x_flat)
        mu, logvar = q[:, :q.size(1)//2], q[:, q.size(1)//2:]
        
        z = self.reparameterize(mu, logvar)
        
        decoded_flat = self.decoder(z)
        decoded = decoded_flat.view(batch_size, self.sequence_length, self.output_dim)
        
        return decoded, mu, logvar
    
class VAELoss(nn.Module):
    def __init__(self):
        super(VAELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, recon_x, x, mu, logvar, epoch):
        """
        MSE + KL Divergence + Similarity.
        """
        # Reconstruction loss (MSE)
        recon_loss = self.mse_loss(recon_x, x)

        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)  
        beta = min(1.0, (epoch+1) / 100)

        # Similarity loss (cosine similarity)
        cos_sim = nn.functional.cosine_similarity(recon_x, x, dim=-1).mean()
        similarity_loss = 1 - cos_sim  

        return recon_loss + beta * kl_loss + 0.1 * similarity_loss, recon_loss, kl_loss, similarity_loss
    
class ConditionedVAEGenerator(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=512, latent_dim=256, visual_dim=768, sequence_length=4):
        super(ConditionedVAEGenerator, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.visual_dim = visual_dim
        self.sequence_length = sequence_length
        self.total_input_dim = input_dim * sequence_length  # 4*512 = 2048
        self.total_visual_dim = visual_dim * sequence_length  # 4*768 = 3072
        
        # Encoder 
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim + self.total_visual_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder 
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + self.total_visual_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.total_input_dim),
            nn.Tanh()  # Normalizza output a [-1, 1]
        )
        
    def encode(self, x, visual_condition):
        batch_size = x.size(0)
        
        x_flat = x.view(batch_size, -1)
        
        visual_flat = visual_condition.view(1, -1).repeat(batch_size, 1)
        
        x_cond = torch.cat([x_flat, visual_flat], dim=1)
        h = self.encoder(x_cond)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z, visual_condition=None):
        batch_size = z.size(0)
        
        if visual_condition is not None:
            visual_flat = visual_condition.view(1, -1).repeat(batch_size, 1)
            
            z_cond = torch.cat([z, visual_flat], dim=1)
            decoded_flat = self.decoder(z_cond)
        else:
            zeros = torch.zeros(batch_size, self.total_visual_dim, device=z.device)
            z_cond = torch.cat([z, zeros], dim=1)
            decoded_flat = self.decoder(z_cond)
        

        output = decoded_flat.view(batch_size, self.sequence_length, self.input_dim)

        return output[0]
    
    def forward(self, x, visual_condition):
        mu, logvar = self.encode(x, visual_condition)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, visual_condition)
        return x_recon, mu, logvar
    
    def sample(self, num_samples, visual_condition, device='cuda'):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z, visual_condition)