import torch
import torch.nn as nn

class CustomPatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768, custom_order=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # Proiezione delle patch
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Ordine personalizzato delle patch
        if custom_order is not None:
            assert len(custom_order) == num_patches, "La lunghezza di custom_order deve corrispondere al numero di patch."
            self.custom_order = torch.tensor(custom_order)
        else:
            self.custom_order = torch.arange(num_patches)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H', W']
        x = x.flatten(2)  # [B, embed_dim, N]
        x = x.transpose(1, 2)  # [B, N, embed_dim]

        # Riordina le patch secondo custom_order
        x = x[:, self.custom_order, :]

        return x