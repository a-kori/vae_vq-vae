import torch


def evaluate(model, test_loader, device):
    """
    Evaluates the models using the testing dataset and calculates its mean loss.

    Parameters:
    - model : The nn model to be evaluated.
    - test_loader : DataLoader providing the test dataset.
    - loss_fun : The loss function used to compute the loss.
    - device: The device to run the model on.
    
    Returns the mean testing loss.
    """

    total_loss = 0

    # Disable gradient computation.
    with torch.no_grad():
        # Switch to evaluation mode.
        model.eval()
        # Iterate over the entire test dataset.
        for image, _ in test_loader:
            # Move image to the appropriate device
            image = image.to(device)

            # recon_batch - reconstructed output from the VAE
            # mu - mean of the latent space distribution
            # logvar - logarithm of the variance of the latent space distribution.
            x_reconst, loss,embedding_loss,commitment_loss = model(image)

            total_loss += loss.item()

    return total_loss / len(test_loader)
