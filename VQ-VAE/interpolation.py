import numpy as np
import torch
import cv2

def interpolate(z1, z2, model, steps, save_path='output/interpolation.avi', fps=24):
    """
    Generates a video showing the interpolation between two random latent vectors z1 and z2.

    Parameters:
    - z1: Starting latent vector.
    - z2: Ending latent vector.
    - model: Trained VQ-VAE model.
    - steps: Number of interpolation steps (number of generated images).
    - save_path: Path of the output video file.
    - fps: Frames per second for the video.
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

    # Get the dimensions of the images
    height, width, _ = images[0].shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    for img in images:
        # Convert RGB (default in most libraries) to BGR (default in OpenCV)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(img_bgr)

    video.release()
    print(f"Interpolation video saved as {save_path}.")
