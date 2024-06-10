import os
import torch, numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid

def sample(model, epoch, device, latent_dim, num_samples, grid_size=10, save_path='./results'):
    """
    Generates images from the VAE model and saves them in a grid.

    Parameters:
    - model : The VAE model to generate images from.
    - epoch : Current epoch number.
    - device : The device to run the model on.
    - latent_dim: Dimensionality of latent space.
    - num_samples : Number of images to generate, must be below 100.
    - grid_size : Dimensions of the grid to arrange the images.
    - save_path : Path to save the generated image grid.
    """

    model.eval()
    os.makedirs(save_path, exist_ok=True)
    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        generated_images = model.decoder(z).cpu()
        save_image(generated_images, f'{save_path}/epoch_{epoch+1}.png', nrow=grid_size, normalize=True)
    
    grid = make_grid(generated_images, nrow=grid_size, normalize=True)
    np_img = grid.numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(np_img, (1, 2, 0)))
    plt.title(f'Epoch {epoch+1}')
    plt.axis('off')
    plt.show()


def plot_reconstruction(img, recons):
    """
    Plot the original and reconstructed images during training.
    """
    _, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5))

    for j in range(5):
        # Original images - transposing to (height, width, channels)
        original_image = img[j].detach().cpu().numpy().transpose(1, 2, 0)
        axes[0][j].imshow(original_image)
        axes[0][j].axis('off')

    for j in range(5):
        # Reconstructed images - transposing to (height, width, channels)
        reconstructed_image = recons[j].detach().cpu().numpy().transpose(1, 2, 0)
        axes[1][j].imshow(reconstructed_image)
        axes[1][j].axis('off')

    plt.tight_layout(pad=0.)
    plt.show()
