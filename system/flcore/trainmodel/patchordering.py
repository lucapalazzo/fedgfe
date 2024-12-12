from torch import nn
import torch
from flcore.trainmodel.pretexttask import PretextTask
from modelutils.patchorderloss import PatchOrderLoss
from modelutils.custompatchembedding import CustomPatchEmbed
from typing import Iterator


class PatchOrdering (PretextTask):
    def __init__(self, backbone=None, input_dim = 768, output_dim = 768, debug_images=False, img_size=224, patch_size=-1, patch_count = -1):
        super(PatchOrdering, self).__init__(backbone=backbone, input_dim = input_dim, output_dim = output_dim, debug_images=debug_images, img_size=img_size, patch_size=patch_size, patch_count = patch_count)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.backbone = backbone
        self.debug_images = debug_images
        self.head = None
        self.name = "patch_ordering"
        self.patch_ordering = None
        self.head_hidden_size = output_dim
        self.custom_order = self.patch_order_create()
        # self.loss = PatchOrderLoss(self.num_patches, self.head_hidden_size)
        self.pretext_loss = nn.CrossEntropyLoss()
        self.pretext_head = nn.Sequential( nn.Linear(output_dim, self.patch_count) ).to(self.device)

        self.custom_patch_embed = CustomPatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=3,
            embed_dim=self.embedding_size,
            custom_order=self.custom_order,
            device = self.device
        )


    def patch_embed ( self, x ):
        return self.custom_patch_embed(x)
    
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        modules = nn.ModuleList()
        modules.add_module("pretext_head", self.pretext_head)
        parameters = modules.parameters(recurse)
        return parameters

    def preprocess_sample(self, x):
        images, self.image_ordering_labels = self.shuffle_patches(x,self.custom_order)
        return images
    
    def loss(self, x):
        target = torch.tensor(self.custom_order).float().to(x.device)
        target = target.unsqueeze(0).expand(x.shape[0],-1)
        return self.pretext_loss(x, target)

    
    def forward(self, x):
        x = self.preprocess_sample(x)
        if self.backbone is not None:
            # self.backbone.logits_only = True
            x = self.backbone(x)


        return self.pretext_head(x)


        images, self.image_ordering_labels = self.shuffle_patches(x,self.custom_order)

        B, C, H, W = x.size()
        patches = self.vit.patch_embed(x)  # [B, num_patches, embed_dim]
        self.save_images(patches, images, max_saved=1)

        x = patches
        
        # Riordina le patch se custom_order è definito
        if self.custom_order is not None:
            self.custom_order_tensor = torch.tensor(self.custom_order).to(x.device)
            # x = x[:, self.custom_order_tensor, :]
            # Riordina anche gli embeddings posizionali
            pos_embed = self.vit.pos_embed[:, 1:, :][:, self.custom_order_tensor, :]
        else:
            pos_embed = self.vit.pos_embed[:, 1:, :]
        

        # Aggiungi l'embedding posizionale
        cls_token = self.vit.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = x + pos_embed
        x = torch.cat((cls_token, x), dim=1)  # [B, 1 + num_patches, embed_dim]
        
        # Passa attraverso i Transformer Encoder layers
        x = self.vit.pos_drop(x)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        
        self.patch_embedding = x[:, 1:].to(x.device)  # Ignora il token di classificazione
        # logits = self.vit.head(x[:, 0])
        return x
        output = self.vit(images)
        return output
    
    def patch_order_create ( self, random_order = True):
        if random_order:
            order = torch.randperm(self.num_patches)
        else:
            order = torch.arange(self.num_patches)
        return order
    
    def shuffle_patches(self, images, order):
        B, C, H, W = images.shape
        # num_patches_per_row = int(math.sqrt(N))
        # device = patches.device

        # shuffled_batches = []

        # shuffled_batches = patches[:, -1,:][:,order,:]
        num_patches_per_row = H // self.patch_size
        # num_patches_per_row = self.num_patches**0.5
        # self.patch_size = int(H // num_patches_per_row)

        patches = self.patchify(images)
        shuffled_batches = patches[::][:, order, :]
        reconstructed_imgs = []
        # self.save_images(images, shuffled_batches, max_saved=1)
        for b in range(shuffled_batches.shape[0]):
            reconstructed_img = self.reconstruct_image_from_patches(shuffled_batches[b], num_patches_per_row, self.patch_size)

            reconstructed_imgs.append(reconstructed_img)
        #     shuffled_patches = []
        #     batch_patches = patches[b]
        #     for i, patch in enumerate(batch_patches):
        #         shuffled_patches.append(batch_patches[order[i]])
        #     shuffled_patches = torch.stack(shuffled_patches) 
        #     # shuffled_patches = torch.Tensor(shuffled_patches).to(device)
        #     reconstructed_img = self.reconstruct_image_from_patches(shuffled_patches, num_patches_per_row, self.patch_size)
        #     save_grid_images([reconstructed_img], nrow=1, output_path=os.path.join(self.image_output_directory, f"shuffled_{b}.png"))
        #     shuffled_batches.append(shuffled_patches)
        reconstructed_imgs = torch.stack(reconstructed_imgs)
        self.save_images(images, reconstructed_imgs, max_saved=1)

        return reconstructed_imgs, order
    
    def patchify(self, x):
        """Divide l'immagine in patch."""
        
        p = self.patch_size
        batch_size, channels, img_h, img_w = x.shape
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(batch_size, channels, -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4)
        # patches = patches.permute(0, 2, 1, 3, 4).reshape(batch_size, -1, p * p * channels)
        return patches
