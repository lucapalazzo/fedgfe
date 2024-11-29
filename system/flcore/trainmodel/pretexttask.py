from torch import nn
import torch

class PretextTask (nn.Module):
    def __init__(self, backbone=None, input_dim, output_dim, debug_images=False):
        super(PretextTask, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = backbone
        self.debug_images = debug_images
        self.head = None

    def to(self, device):
        self.device = device
        self.fc.to(device)
        self.fc1.to(device)
        self.fc2.to(device)
        self.relu.to(device)

        return self

    def forward(self, x):

        images, self.image_rotation_labels = self.rotate_images(x)
        self.save_images(x, images, max_saved=1)

        output = self.vit(images)
        return output
    
    def rotate_images(self, imgs):
        B, C, H, W = imgs.shape


        device = imgs.device
        num_patches_per_row = H // self.patch_size
        num_patches = num_patches_per_row ** 2
        labels = torch.zeros(B, dtype=torch.long, device=device)
        rotated_imgs = torch.zeros_like(imgs).to(imgs.device)

        a = transforms
        for b in range(B):

            angle = np.random.choice(self.image_rotation_angles)
            angle_index = self.image_rotation_angles.index(angle)
            labels[b] = angle_index
            img = imgs[b]
            rotated = transforms.functional.rotate(img, angle.item()).to(device)
            rotated_imgs[b] = rotated
        return rotated_imgs, labels
