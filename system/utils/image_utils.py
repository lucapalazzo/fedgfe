
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import torchvision.transforms.functional as F
from torchvision.utils import save_image as save_image_from_tensor
import numpy as np

def plot_image(image_sample):
    plt.imshow(image_sample, cmap='gray')  # Puoi cambiare 'gray' se l'immagine è a colori
    plt.axis('off')  # Rimuovi gli assi per una visualizzazione pulita
    plt.show()

def save_image(image_sample, path):
    save_image_from_tensor(F.to_tensor(image_sample), path)

def save_grid_images(images, nrow=1, output_path="image.png"):
    grid = make_grid_images(images, nrow=nrow)
    save_image(grid, output_path)

def make_grid_images(images, nrow=1):
    # for i in range(len(images)):
        # if images[i].shape[0] > 10:
        #     images[i] = images[i].permute(2, 0, 1)
        # if len(images[i].shape) > 3:
        #     images[i] = images[i].squeeze(0)
    images = torch.stack(images)
    grid = make_grid(images, nrow=nrow)
    grid = grid.permute(1, 2, 0)
    grid = grid.detach().cpu().numpy()
    return grid

def plot_grid_images(images, nrow=1, output_path="image.png"):
    # for i in range(len(images)):
    #     if images[i].shape[0] > 3:
    #         images[i] = images[i].permute(2, 0, 1)
            
    grid = make_grid(images, nrow=nrow)
    grid = grid.permute(1, 2, 0)
    grid = grid.detach().cpu().numpy()
    save_image(grid, output_path)