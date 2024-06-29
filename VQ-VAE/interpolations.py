import torch
import imageio
import numpy as np

def interpolate(z1, z2, model, step_size, save_path='output/interpolation.mp4'):
    '''
    Function to interpolate between two latent vectors z1 and z2.
    '''
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
            
            # Scale image to [0, 255] and convert to uint8
            img_np = (reconstructed_image.cpu().squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            images.append(img_np)
    
    # Save images as video
    imageio.mimsave(save_path, images, fps=10)
    print(f"Interpolation video saved as {save_path}.")
