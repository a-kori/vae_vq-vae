import torch
import numpy as np
import imageio
from torchvision.utils import save_image

def interpolate(model, z1, z2, steps=10, output_path='output/interpolation.mp4'):
    interpolations = []
    step_size = 1.0 / (steps - 1)
    
    for step in range(steps):
        alpha = step * step_size
        z_interp = (1 - alpha) * z1 + alpha * z2
        with torch.no_grad():
            img = model.decoder(z_interp.unsqueeze(0)).squeeze(0).cpu()
        interpolations.append(img)
    
    # Save images
    images = [img.permute(1, 2, 0).numpy() for img in interpolations]
    imageio.mimsave(output_path, [np.uint8(img * 255) for img in images], fps=2)

    print(f'Interpolation video saved to {output_path}')
