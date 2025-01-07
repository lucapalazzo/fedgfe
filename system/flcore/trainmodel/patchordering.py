from torch import nn
import torch
from flcore.trainmodel.patchpretexttask import PatchPretextTask
from modelutils.patchorderloss import PatchOrderLoss
from modelutils.custompatchembedding import CustomPatchEmbed
from typing import Iterator

from timm.models.vision_transformer import VisionTransformer
from transformers import ViTModel


class PatchOrdering (PatchPretextTask):
    def __init__(self, backbone=None, input_dim = 768, output_dim = 768, debug_images=False, img_size=224, patch_size=-1, patch_count = -1):
        super(PatchOrdering, self).__init__(backbone=backbone, input_dim = input_dim, output_dim = output_dim, debug_images=debug_images, img_size=img_size, patch_size=patch_size, patch_count = patch_count)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = "patch_ordering"
        self.patch_ordering = None
        self.head_hidden_size = output_dim
        self.custom_order = self.patch_order_create()
        # self.loss = PatchOrderLoss(self.num_patches, self.head_hidden_size)
        self.pretext_loss = nn.CrossEntropyLoss()
        # self.pretext_head = nn.Sequential( nn.Linear(output_dim, self.patch_count) ).to(self.device)
        self.pretext_head = nn.ModuleList(
            nn.Sequential(
                # nn.Linear(output_dim, 64).requires_grad_(False),
                # nn.ReLU(),
                nn.Linear(output_dim, self.patch_count)
            ) for _ in range(patch_count)
        ).to(self.device)

        self.heads_loss = [self.pretext_loss for _ in range(self.patch_count)]



        # self.custom_patch_embed = CustomPatchEmbed(
        #     img_size=self.img_size,
        #     patch_size=self.patch_size,
        #     in_channels=3,
        #     embed_dim=self.embedding_size,
        #     custom_order=self.custom_order,
        #     device = self.device
        # )


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
    
    def accuracy(self, x, y = None):
        # targets = torch.tensor(self.custom_order).long().to(self.device).unsqueeze(0).expand(x[0].shape[0],-1)
        targets = torch.tensor(self.custom_order).long().to(self.device)

        accuracy = 0
        patches = 0
        for patch_index,output in enumerate(x):
            target = targets[patch_index]
            for i,logits in enumerate(output):
                predicted = logits.argmax(dim=0)
                accuracy += (logits.argmax(dim=0) == target).float()
                # print (f"{i} Predicted: {predicted}, Target: {target}")
                patches += 1

        accuracy = accuracy / patches  
        return accuracy
        for output in x:
            accuracy = (output.argmax(dim=1) == targets).float()
        return (x.argmax(dim=1) == targets).float

    def loss(self, x, y = None):
        # targets = torch.tensor(self.custom_order).float().to(self.device)
        targets = torch.tensor(self.custom_order).to(self.device)

        targets = targets.unsqueeze(0).expand(x[0].shape[0],-1)

        # for criterion, output in zip(criterions, x):
        #     print(f"output: {output.shape}, target: {targets.shape}")
        #     loss = criterion(output, targets)
        #     print(f"loss: {loss}")
        patches = range(self.patch_count)
        losses = [criterion(output, targets[:,patch]) for patch, criterion, output in zip(patches,self.heads_loss, x)]

        loss = sum(losses)/len(losses)
        
        return loss
        return self.pretext_loss(x, targets)
    
    def forward(self, x):
        x = self.preprocess_sample(x)
        if self.backbone is not None:
            # self.backbone.logits_only = True
            x = self.backbone(x)

        if self.cls_token_only:
            if isinstance(self.backbone, VisionTransformer):
                x = x[:,0,:]
            elif isinstance(self.backbone, ViTModel):
                x = x.last_hidden_state[:,0,:]
        predictions = [classifier(x) for classifier in self.pretext_head]
        return predictions 
        
        return self.pretext_head(x)
    
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