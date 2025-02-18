import torch, numpy as np
import matplotlib.pyplot as plt

def plot_reconstruction(model, test_loader, device, num_samples=5):
    """
    Plots the reconstructions of images from the testing dataset 
    and shows them next to the original ones.

    Parameters:
    - model : The VAE model to generate images from.
    - test_loader : Loader for the testing dataset.
    - device : The device to run the model on.
    - num_samples : Number of images to generate, must be below 100.
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Get a batch of test data
    data, _ = next(iter(test_loader))
    data = data.to(device)
    
    # Ensure random sampling
    indices = np.random.choice(len(data), num_samples, replace=False)
    sampled_data = data[indices]

    # Pass the data through the VAE
    with torch.no_grad():
        recon_batch, _, _, _= model(sampled_data)
    
    # Plot the original and reconstructed images
    _, axes = plt.subplots(nrows=2, ncols=num_samples, figsize=(num_samples * 2, 4))
    
    for i in range(num_samples):
        # Original image
        original_img = sampled_data[i].cpu().numpy().transpose(1, 2, 0)
        axes[0, i].imshow(original_img.squeeze(), cmap='gray')
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Reconstructed image
        recon_img = recon_batch[i].cpu().numpy().transpose(1, 2, 0)
        axes[1, i].imshow(recon_img.squeeze(), cmap='gray')
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
