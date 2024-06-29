import torch
import numpy as np
import moviepy.editor as mpy

def interpolate(z1, z2, model, steps, save_path='output/interpolation.mp4'):
    """
    Generates a video showing the interpolation between two random latent vectors z1 and z2.

    Parameters:
    - z1: Starting latent vector.
    - z2: Ending latent vector.
    - model: Trained VQ-VAE model.
    - steps: Number of interpolation steps (number of generated images).
    - save_path: Path of the output video file.
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Generate interpolation vectors
    interpolation_vectors = []
    for alpha in np.linspace(0, 1, steps):
        interpolation_vector = alpha * z1 + (1 - alpha) * z2
        interpolation_vectors.append(interpolation_vector)

    images = []
    with torch.no_grad():
        for vector in interpolation_vectors:
            # Decode interpolated vectors
            reconstructed_image = model.decoder(vector.to(z1.device))
            reconstructed_image = torch.clamp(reconstructed_image, 0, 1)
            
            # Scale image to [0, 255] and convert to uint8
            img_np = (reconstructed_image.cpu().squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            images.append(img_np)

    # Create a video clip from the list of numpy arrays
    clip = mpy.ImageSequenceClip(images, fps=1)  # fps=1 means each image will be shown for 1 second

    # Save the resulting video
    clip.write_videofile(save_path, codec="libx264")

    print(f"Interpolation video saved as {save_path}.")
