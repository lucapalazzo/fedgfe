from torch import nn
import torch
from flcore.trainmodel.pretexttask import PretextTask
from modelutils.patchorderloss import PatchOrderLoss
from modelutils.custompatchembedding import CustomPatchEmbed
from typing import Iterator


class PatchPretextTask (PretextTask):
    def __init__(self, backbone=None, input_dim = 768, output_dim = 768, debug_images=False, img_size=224, patch_size=-1, patch_count = -1, cls_token_only=False):
        super(PatchPretextTask, self).__init__(backbone=backbone, input_dim = input_dim, output_dim = output_dim, debug_images=debug_images, img_size=img_size, patch_size=patch_size, patch_count = patch_count)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.backbone = backbone
        self.debug_images = debug_images
        self.head = None
        self.name = "patch_pretext"
        self.cls_token_only = cls_token_only

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        raise NotImplementedError("This method must be implemented in the subclass")

    # def loss(self, x):
    #     target = torch.tensor(self.custom_order).float().to(x.device)
    #     target = target.unsqueeze(0).expand(x.shape[0],-1)
    #     return self.pretext_loss(x, target)

    # def forward(self, x):
    #     raise NotImplementedError("This method must be implemented in the subclass")
   
    def reconstruct_image_from_patches(self, patches, num_patches_per_row, patch_size):
        """Ricostruisce l'immagine a partire dalle patch."""
        C = patches.shape[1]
        # image = patches.view(num_patches_per_row, num_patches_per_row, C, patch_size, patch_size)
        image = patches.reshape(num_patches_per_row, num_patches_per_row, C, patch_size, patch_size)
        image = image.permute(2, 0, 3, 1, 4).contiguous()
        image = image.view(C, num_patches_per_row * patch_size, num_patches_per_row * patch_size)
        return image
    
    def patchify(self, x, patch_count=-1):
        """Divide l'immagine in patch."""

        batch_size, channels, img_h, img_w = x.shape
        if patch_count != -1:
            # p = int(self.img_size // patch_count**0.5)
            p = int (img_h / patch_count**0.5)
        else:
            p = self.patch_size
        patches = x.unfold(2, p, p).unfold(3, p, p)
        patches = patches.contiguous().view(batch_size, channels, -1, p, p)
        patches = patches.permute(0, 2, 1, 3, 4)
        # patches = patches.permute(0, 2, 3, 1, 4, 5)
        # patches = patches.permute(0, 2, 1, 3, 4).reshape(batch_size, -1, p * p * channels)
        return patches
    
    def debug_output_images(self, samples, predictions, max_saved=1, output_filename=None, prefix=None, postfix=None, node_id=None):
        print("Calling debug_images in pretexttask")
        return
        if self.debug_images or (output_filename and os.path.exists("save_debug_images")):
            import matplotlib.pyplot as plt
            import numpy as np
            
            for i in range(min(max_saved, len(samples))):
                sample = samples[i]
                prediction = predictions[i]
                
                # Ricostruisci l'immagine dalle patch
                num_patches_per_row = int(prediction.shape[0]**0.5)
                patch_size = prediction.shape[1]
                reconstructed_image = self.reconstruct_image_from_patches(prediction, num_patches_per_row, patch_size)
                
                # Converti l'immagine in un formato visualizzabile
                reconstructed_image = reconstructed_image.cpu().numpy()
                reconstructed_image = np.transpose(reconstructed_image, (1, 2, 0))
                
                plt.figure(figsize=(8, 8))
                plt.imshow(reconstructed_image)
                plt.axis('off')
                
                if output_filename:
                    plt.savefig(f"{output_filename}_{i}.png")
                else:
                    plt.show()
