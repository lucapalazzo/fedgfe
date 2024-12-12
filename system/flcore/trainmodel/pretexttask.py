from torch import nn
import torch
import os
from utils.image_utils import plot_image, plot_grid_images, save_grid_images

class PretextTask (nn.Module):
    def __init__(self, backbone=None, input_dim = 768, output_dim = 768, debug_images=False, image_output_directory = 'output_images', embedding_size = 768, img_size=224, patch_size=-1, patch_count = -1):
        super(PretextTask, self).__init__()


        if ( patch_size == -1  and patch_count == -1 ):
            raise ValueError("At least one of patch_size or patch_count must be defined")
        elif (patch_size != -1):
            self.patch_size = patch_size
            self.num_patches = (img_size // self.patch_size) ** 2
            self.patch_count = self.num_patches
        elif (patch_count != -1):
            self.num_patches = patch_count
            self.patch_count = patch_count
            self.patch_size = img_size // int(self.num_patches ** 0.5)


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = backbone
        self.debug_images = debug_images
        self.image_output_directory = image_output_directory
        self.head = None
        self.img_size = img_size
        self.embedding_size = embedding_size
        self.custom_patch_embed = None
        self.pretext_head = nn.Identity()


    def patch_embed ( self, x ):
        print ( "Patch embed not implemented" )
        return x
    
    def forward(self, x):
        return x
    
    def loss(self, x):
        print ( "Pretext loss not implemented" )
        return x
    

    def reconstruct_image_from_patches(self, patches, num_patches_per_row, patch_size):
        C = patches.shape[1]
        image = patches.view(num_patches_per_row, num_patches_per_row, C, patch_size, patch_size)
        image = image.permute(2, 0, 3, 1, 4).contiguous()
        image = image.view(C, num_patches_per_row * patch_size, num_patches_per_row * patch_size)
        return image
    
    def save_images ( self, images, patches, max_saved = 0 ):

        if self.debug_images:

            B, C, H, W = images.shape
            # saved_order = np.random(0, ) 
            num_patches_per_row = H // self.patch_size
            recontructed_patches = []
            count = 0
            for batch_id in range(B):
                reconstructed = patches[batch_id]
                if ( patches[batch_id].shape[1] != images[batch_id].shape[1] ):
                    reconstructed = self.reconstruct_image_from_patches(patches[batch_id], num_patches_per_row, self.patch_size)
                output_filename = os.path.join(self.image_output_directory, f"reconstructed_{batch_id}.png")
                save_grid_images([images[batch_id], reconstructed], nrow=2, output_path=output_filename)
                recontructed_patches.append(reconstructed)
                count += 1
                if max_saved != 0 and count >= max_saved:
                    break 