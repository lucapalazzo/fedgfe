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
    def __init__(self, input_dim, hidden_dim=512, latent_dim=256, sequence_length=4, use_learned_upsampling=False, device = 'cpu'):
        super(VAEGenerator, self).__init__()

        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.output_dim = input_dim  # (768)
        self.total_output_dim = sequence_length * input_dim  # sequence_length * 768
        self.use_learned_upsampling = use_learned_upsampling
        self.device = device

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.total_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, latent_dim * 2)
        ).to(device)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, self.total_output_dim)
        ).to(device)

        # Initialize weights with small values to prevent explosion
        self._initialize_weights()

        # Optional: Learned upsampling network (more sophisticated than linear interpolation)
        if use_learned_upsampling:
            self.upsampler = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, input_dim, kernel_size=3, padding=1)
            ).to(device)
        else:
            self.upsampler = None

    def to(self, device):
        super(VAEGenerator, self).to(device)
        if self.upsampler is not None:
            self.upsampler.to(device)
        self.device = device
        return self
    
    def reparameterize(self, mu, logvar):
        # Clamp logvar to prevent explosion - use tighter bounds
        logvar = torch.clamp(logvar, min=-5, max=2)
        mu = torch.clamp(mu, min=-5, max=5)

        # Compute std with additional safety
        # With logvar max=2, std max = exp(1) ≈ 2.718
        std = torch.exp(0.5 * logvar)

        # Safety check - if std becomes too large or contains NaN, use fallback
        if torch.isnan(std).any() or torch.isinf(std).any() or (std > 10).any():
            print("Warning: Unstable std in reparameterize, using safer fallback")
            std = torch.clamp(std, min=1e-6, max=3.0)
            std = torch.nan_to_num(std, nan=1.0, posinf=3.0, neginf=1e-6)

        eps = torch.randn_like(std)
        z = mu + eps * std

        # Final safety check on output
        if torch.isnan(z).any() or torch.isinf(z).any():
            print("Warning: NaN/Inf in reparameterize output, clamping")
            z = torch.nan_to_num(z, nan=0.0, posinf=5.0, neginf=-5.0)

        # Clamp output to reasonable range
        z = torch.clamp(z, min=-5, max=5)

        return z
    
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

        # Input validation
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN/Inf in VAE input, cleaning")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        x_flat = x.view(batch_size, -1)

        # Encoder
        q = self.encoder(x_flat)

        # Check encoder output
        if torch.isnan(q).any() or torch.isinf(q).any():
            print("Warning: NaN/Inf in encoder output")
            q = torch.nan_to_num(q, nan=0.0, posinf=1e6, neginf=-1e6)

        mu, logvar = q[:, :q.size(1)//2], q[:, q.size(1)//2:]

        # Reparameterize (includes internal safety checks)
        z = self.reparameterize(mu, logvar)

        # Decoder
        decoded_flat = self.decoder(z)

        # Check decoder output
        if torch.isnan(decoded_flat).any() or torch.isinf(decoded_flat).any():
            print("Warning: NaN/Inf in decoder output")
            decoded_flat = torch.nan_to_num(decoded_flat, nan=0.0, posinf=1e6, neginf=-1e6)

        decoded = decoded_flat.view(batch_size, self.sequence_length, self.output_dim)

        return decoded, mu, logvar

    def sample(self, num_samples, device='cuda', target_sequence_length=None):
        """
        Generate new samples from the latent space without conditioning.

        Args:
            num_samples (int): Number of samples to generate
            device (str): Device to generate samples on
            target_sequence_length (int, optional): Target sequence length for output.
                If None, uses the model's default sequence_length.
                If specified (e.g., 1214 for full AST output), upsamples to that length.

        Returns:
            samples: [num_samples, target_sequence_length or sequence_length, output_dim]

        Upsampling Strategy:
            - If use_learned_upsampling=True: Uses Conv1D layers for better quality
            - Otherwise: Uses linear interpolation (fast but may lose details)
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        decoded = self.decode(z)  # [num_samples, sequence_length, output_dim]

        # If target_sequence_length is specified and different from default, upsample
        if target_sequence_length is not None and target_sequence_length != self.sequence_length:
            if self.use_learned_upsampling and self.upsampler is not None:
                # Learned upsampling: better quality but requires training
                # First do linear interpolation to target length
                decoded_transposed = decoded.transpose(1, 2)  # [batch, output_dim, seq_len]
                upsampled = torch.nn.functional.interpolate(
                    decoded_transposed,
                    size=target_sequence_length,
                    mode='linear',
                    align_corners=False
                )
                # Then refine with learned conv layers
                upsampled = self.upsampler(upsampled)
                decoded = upsampled.transpose(1, 2)  # [batch, target_seq_len, output_dim]
            else:
                # Standard linear interpolation: fast and parameter-free
                decoded_transposed = decoded.transpose(1, 2)  # [batch, output_dim, seq_len]
                upsampled = torch.nn.functional.interpolate(
                    decoded_transposed,
                    size=target_sequence_length,
                    mode='linear',
                    align_corners=False
                )
                decoded = upsampled.transpose(1, 2)  # [batch, target_seq_len, output_dim]

        return decoded

    def _initialize_weights(self):
        """Initialize weights with small values to prevent numerical explosion."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier/Glorot initialization with reasonable gain
                # Changed from 0.01 to 0.1 - 0.01 was too small and caused instability
                nn.init.xavier_normal_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class VAELoss(nn.Module):
    def __init__(self, total_epochs=100, beta_warmup_ratio=0.5):
        """
        VAE Loss with adaptive beta scheduling.

        Args:
            total_epochs: Total number of training epochs (default: 100)
            beta_warmup_ratio: Ratio of epochs for beta warmup (default: 0.5)
                              Beta reaches 1.0 at (total_epochs * beta_warmup_ratio)
        """
        super(VAELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.total_epochs = total_epochs
        self.beta_warmup_epochs = max(1, int(total_epochs * beta_warmup_ratio))

    def forward(self, recon_x, x, mu, logvar, epoch):
        # Input validation - check for NaN/Inf in inputs
        if torch.isnan(recon_x).any() or torch.isinf(recon_x).any():
            print("Warning: NaN/Inf detected in recon_x input to loss function")
            recon_x = torch.nan_to_num(recon_x, nan=0.0, posinf=1e6, neginf=-1e6)

        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Warning: NaN/Inf detected in x input to loss function")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)

        # DON'T clamp mu/logvar here - it breaks gradients!
        # Clamping is done in reparameterize() during forward pass
        # Here we compute KL loss with the actual values to allow gradients to flow

        # Reconstruction loss (MSE) with safety check
        recon_loss = self.mse_loss(recon_x, x)
        if torch.isnan(recon_loss) or torch.isinf(recon_loss):
            print("Warning: NaN/Inf in reconstruction loss, using fallback")
            recon_loss = torch.tensor(1.0, device=recon_x.device)

        # KL divergence with improved numerical stability
        # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # Compute per-dimension KL and then average (more stable than sum)

        # Clamp logvar ONLY for the exp() computation to prevent numerical overflow
        # but keep gradients flowing through the original logvar
        logvar_safe = torch.clamp(logvar, min=-10, max=10)  # Only for exp()

        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar_safe.exp())  # (B, latent_dim)
        kl_loss = kl_per_dim.mean()  # Mean over batch AND dimensions

        # Only check for NaN/Inf (don't hard clamp - let gradients flow)
        if torch.isnan(kl_loss) or torch.isinf(kl_loss):
            print("Warning: NaN/Inf in KL loss, using fallback")
            kl_loss = torch.tensor(0.0, device=recon_x.device, requires_grad=True)

        # Beta scheduling - balanced to prevent both posterior collapse and KL explosion
        # Start very small and gradually increase
        beta_max = 0.01  # Increased to give KL more influence
        beta = min(beta_max, beta_max * (epoch + 1) / self.beta_warmup_epochs)

        # No free bits - already using mean instead of sum makes KL values small enough

        # Similarity loss (cosine similarity) with safety checks
        recon_flat = recon_x.view(recon_x.size(0), -1)
        x_flat = x.view(x.size(0), -1)

        # Add small epsilon to prevent division by zero in cosine similarity
        recon_flat_norm = torch.nn.functional.normalize(recon_flat + 1e-8, p=2, dim=-1)
        x_flat_norm = torch.nn.functional.normalize(x_flat + 1e-8, p=2, dim=-1)
        cos_sim = (recon_flat_norm * x_flat_norm).sum(dim=-1).mean()
        cos_sim = torch.clamp(cos_sim, min=-1.0, max=1.0)  # Ensure valid range
        similarity_loss = 1 - cos_sim

        if torch.isnan(similarity_loss) or torch.isinf(similarity_loss):
            print("Warning: NaN/Inf in similarity loss, using fallback")
            similarity_loss = torch.tensor(1.0, device=recon_x.device)

        # Adaptive similarity weight: starts at 0.1, decays to 0.01
        sim_weight = max(0.01, 0.1 * (1 - epoch / self.total_epochs))

        # Combine losses
        total_loss = recon_loss + beta * kl_loss + sim_weight * similarity_loss

        # Final safety check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: NaN/Inf in total loss! recon={recon_loss.item():.4f}, kl={kl_loss.item():.4f}, sim={similarity_loss.item():.4f}")
            total_loss = recon_loss  # Fallback to just reconstruction loss

        return total_loss, recon_loss, kl_loss, similarity_loss
    
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
            nn.Linear(self.total_input_dim + self.total_visual_dim, hidden_dim),
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

        # Flatten visual_condition correctly: (seq_len, visual_dim) -> (1, seq_len*visual_dim)
        visual_flat = visual_condition.reshape(1, -1).repeat(batch_size, 1)

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

        return output

    def forward(self, x, visual_condition):
        mu, logvar = self.encode(x, visual_condition)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, visual_condition)
        return x_recon, mu, logvar

    def sample(self, num_samples, visual_condition, device='cuda'):
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z, visual_condition)


class MultiModalVAEGenerator(nn.Module):
    """
    VAE Generator condizionato da embeddings CLIP (4x768) e T5 (4x4096).
    Supporta l'uso di entrambi gli embeddings o solo uno dei due.
    """
    def __init__(self, input_dim=512, hidden_dim=512, latent_dim=256,
                 clip_dim=768, t5_dim=4096, fusion_dim=512, sequence_length=4):
        super(MultiModalVAEGenerator, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.total_input_dim = input_dim * sequence_length  # 4*512 = 2048
        self.clip_dim = clip_dim
        self.t5_dim = t5_dim
        self.fusion_dim = fusion_dim

        # Proiezioni per CLIP e T5
        self.clip_projection = nn.Sequential(
            nn.Linear(clip_dim * sequence_length, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        self.t5_projection = nn.Sequential(
            nn.Linear(t5_dim * sequence_length, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Fusion layer - supporta sia singolo che doppio condizionamento
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )

        # Single modality fallback (quando si usa solo CLIP o solo T5)
        self.single_fusion = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU()
        )

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.total_input_dim + fusion_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )

        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + fusion_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, self.total_input_dim),
            nn.Tanh()
        )

    def fuse_embeddings(self, clip_embedding=None, t5_embedding=None):
        """
        Fonde CLIP e T5 embeddings. Supporta:
        - Entrambi (clip_embedding AND t5_embedding)
        - Solo CLIP (clip_embedding, t5_embedding=None)
        - Solo T5 (clip_embedding=None, t5_embedding)

        Args:
            clip_embedding: [sequence_length, clip_dim] o [batch_size, sequence_length, clip_dim]
            t5_embedding: [sequence_length, t5_dim] o [batch_size, sequence_length, t5_dim]

        Returns:
            fused: [batch_size, fusion_dim] embedding fuso
        """
        if clip_embedding is None and t5_embedding is None:
            raise ValueError("Almeno uno tra clip_embedding e t5_embedding deve essere fornito")

        # Gestisci batch dimension
        if clip_embedding is not None:
            if len(clip_embedding.shape) == 2:  # [seq_len, dim]
                clip_embedding = clip_embedding.unsqueeze(0)  # [1, seq_len, dim]
            batch_size = clip_embedding.size(0)
            clip_flat = clip_embedding.view(batch_size, -1)
            clip_proj = self.clip_projection(clip_flat)
        else:
            clip_proj = None

        if t5_embedding is not None:
            if len(t5_embedding.shape) == 2:  # [seq_len, dim]
                t5_embedding = t5_embedding.unsqueeze(0)  # [1, seq_len, dim]
            batch_size = t5_embedding.size(0)
            t5_flat = t5_embedding.view(batch_size, -1)
            t5_proj = self.t5_projection(t5_flat)
        else:
            t5_proj = None

        # Fusione
        if clip_proj is not None and t5_proj is not None:
            # Entrambi disponibili - fusione completa
            fused = self.fusion(torch.cat([clip_proj, t5_proj], dim=1))
        elif clip_proj is not None:
            # Solo CLIP
            fused = self.single_fusion(clip_proj)
        else:
            # Solo T5
            fused = self.single_fusion(t5_proj)

        return fused

    def encode(self, x, clip_embedding=None, t5_embedding=None):
        """
        Args:
            x: [batch_size, sequence_length, input_dim] - prompt testuali
            clip_embedding: [sequence_length, clip_dim] - embedding CLIP (4x768) opzionale
            t5_embedding: [sequence_length, t5_dim] - embedding T5 (4x4096) opzionale
        """
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        # Fonde embeddings multi-modali
        condition = self.fuse_embeddings(clip_embedding, t5_embedding)
        if condition.size(0) == 1:
            condition = condition.repeat(batch_size, 1)

        x_cond = torch.cat([x_flat, condition], dim=1)
        h = self.encoder(x_cond)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, clip_embedding=None, t5_embedding=None):
        """
        Args:
            z: [batch_size, latent_dim]
            clip_embedding: [sequence_length, clip_dim] opzionale
            t5_embedding: [sequence_length, t5_dim] opzionale
        """
        batch_size = z.size(0)

        condition = self.fuse_embeddings(clip_embedding, t5_embedding)
        if condition.size(0) == 1:
            condition = condition.repeat(batch_size, 1)

        z_cond = torch.cat([z, condition], dim=1)
        decoded_flat = self.decoder(z_cond)
        return decoded_flat.view(batch_size, self.sequence_length, self.input_dim)

    def forward(self, x, clip_embedding=None, t5_embedding=None):
        """
        Forward pass completo

        Args:
            x: [batch_size, sequence_length, input_dim]
            clip_embedding: [sequence_length, clip_dim] opzionale
            t5_embedding: [sequence_length, t5_dim] opzionale

        Returns:
            x_recon: [batch_size, sequence_length, input_dim] - ricostruzione
            mu: [batch_size, latent_dim] - media latente
            logvar: [batch_size, latent_dim] - log varianza
        """
        mu, logvar = self.encode(x, clip_embedding, t5_embedding)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, clip_embedding, t5_embedding)
        return x_recon, mu, logvar

    def sample(self, num_samples, clip_embedding=None, t5_embedding=None, device='cuda'):
        """
        Genera nuovi prompt condizionati

        Args:
            num_samples: numero di campioni da generare
            clip_embedding: [sequence_length, clip_dim] opzionale
            t5_embedding: [sequence_length, t5_dim] opzionale
            device: dispositivo CUDA/CPU

        Returns:
            samples: [num_samples, sequence_length, input_dim]
        """
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z, clip_embedding, t5_embedding)