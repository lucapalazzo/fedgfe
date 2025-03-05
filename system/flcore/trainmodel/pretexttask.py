from torch import nn
import torch
import os
from utils.image_utils import plot_image, plot_grid_images, save_grid_images

class PretextTask (nn.Module):
    def __init__(self, backbone=None, input_dim = 768, output_dim = 768, debug_images=False, image_output_directory = 'output_images', embedding_size = 768, img_size=224, patch_size=-1, patch_count = -1, cls_token_only = True):
        super(PretextTask, self).__init__()

        self.img_size = img_size
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

        self.patch_count_per_row = int(self.img_size / self.patch_size)

        self.cls_token_only = cls_token_only


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = backbone
        self.debug_images = debug_images
        self.image_output_directory = image_output_directory
        self.head = None
        self.img_size = img_size
        self.embedding_size = embedding_size
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.pretext_head = nn.Identity()


    def patch_embed ( self, x ):
        print ( "Patch embed not implemented" )
        return x
    
    def forward(self, x):
        return x
    
    def loss(self, x):
        print ( "Pretext loss not implemented" )
        return x

    def pretext_accuracy(self, x):
        print ( "Pretext accuracy not implemented" )
        return x 

    def reconstruct_image_from_patches(self, patches, num_patches_per_row, patch_size):
        C = patches.shape[1]
        image = patches.view(num_patches_per_row, num_patches_per_row, C, patch_size, patch_size)
        image = image.permute(2, 0, 3, 1, 4).contiguous()
        image = image.view(C, num_patches_per_row * patch_size, num_patches_per_row * patch_size)
        return image
    
    def save_images ( self, images, patches = None, max_saved = 0 ):

        B, C, H, W = images[0].shape
        # saved_order = np.random(0, ) 
        num_patches_per_row = H // self.patch_size
        reconstructed_patches = []
        count = 0
        for batch_id in range(B):
            output_filename = os.path.join(self.image_output_directory, f"reconstructed_{batch_id}.png")
            if patches is not None:
                for patch in patches:
                    reconstructed = self.reconstruct_image_from_patches(patches[batch_id], num_patches_per_row, self.patch_size)
                    reconstructed_patches.append(reconstructed)

            save_images = [image[batch_id] for image in images]
            save_images += [reconstructed_patches[i][batch_id] for i in range(len(reconstructed_patches))]
            save_grid_images(save_images, nrow=len(images), output_path=output_filename)
            count += 1
            if max_saved != 0 and count >= max_saved:
                break 