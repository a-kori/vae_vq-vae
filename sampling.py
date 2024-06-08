import torch
import matplotlib.pyplot as plt
import numpy as np

def sample(model,latent_dim, device, num_samples=100, grid_size=(10, 10)):
    """
    Parameters:
    -----------
    model : The trained nn model used for generating images.
    num_samples : The number of images to generate.
    grid_size : The dimensions of the grid.
    img_size : The size of each generated image.
    """

    model.eval()
    generated_images = []

    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn(num_samples, latent_dim).to(device)
            # Generate image from noise
            generated_img = model.decoder(z)
            generated_images.append(generated_img)

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    idx = 0
    for i in range(grid_size):
        for j in range(grid_size):
            if idx >= num_samples:
                break
            image = generated_images[idx].permute(1, 2, 0).numpy()
            axes[i, j].imshow(image)
            axes[i, j].axis('off')
            idx += 1

    plt.tight_layout()
    plt.savefig('images/generated_grid.png')
    plt.show()
