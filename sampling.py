import torch
import matplotlib.pyplot as plt

def sample(model, num_samples=100, grid_size=(10, 10), img_size=(28, 28)):
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
            # Generate random noise
            noise = torch.randn(1, *img_size).to(next(model.parameters()).device)
            # Generate image from noise
            generated_img = model(noise).cpu().numpy().squeeze()
            generated_images.append(generated_img)

    fig, axes = plt.subplots(*grid_size, figsize=(10, 10))

    for ax, img in zip(axes.flatten(), generated_images):
        ax.imshow(img, cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('generated_grid.png')
    plt.show()
