import torch
import numpy as np
import matplotlib.pyplot as plt
import os 
import imageio
# Import the necessary libraries
import torch, torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR

# Import the VAE model and functions
import networks
from interpolations import interpolate
from datasets import create_datasets
from evaluation import evaluate
from sampling import plot_reconstruction
from plotting import plot_loss_lr, plot_loss_components

def interpolate(z1, z2, model, step_size, save_path='output/interpolation.mp4'):

    # Ensure model is in evaluation mode
    model.eval()

    # Generate interpolation steps
    interpolation_vectors = []
    for alpha in np.linspace(0, 1, step_size):
        interpolation_vector = alpha * z1 + (1 - alpha) * z2
        interpolation_vectors.append(interpolation_vector)

    # Decode interpolated vectors
    images = []
    with torch.no_grad():
        for vector in interpolation_vectors:
            reconstructed_image = model.decoder(vector.to(z1.device))
            reconstructed_image = torch.clamp(reconstructed_image, 0, 1)  # Clamp to valid pixel range
            # Save the plot to a temporary image file
            # Scale image to [0, 255] and convert to uint8
            img_np = (reconstructed_image.cpu().squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            images.append(img_np)

    # Plot and save images
    fig, axes = plt.subplots(nrows=1, ncols=step_size, figsize=(step_size * 2, 2))
    for i, image in enumerate(images):
        axes[i].imshow(image)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
    imageio.mimsave(save_path, images, fps=10)
    
def main():
    # Configuration
    input_channels = 3
    emb_dim = 64
    n_embed = 512
    commitment_cost = 0.25
    batch_size = 128  # Just for demonstration, usually larger for training
    image_size = 32  # CIFAR-10 image size
    
    # Create the VQ-VAE model
    model = VQVAE(input_channels, emb_dim, n_embed, commitment_cost).to('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = 'CIFAR10'
    train_loader, test_loader = create_datasets(dataset_name, batch_size)

    images, labels = next(iter(train_loader))  # Fetch the next batch
    
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images = images.to(device)
    model = model.to(device)
    
    reconstructed_images, loss, embedding_loss, commitment_loss = model(images)
    
if __name__ == "__main__":
    main()