import torch


def evaluate(model_option, model, test_loader, loss_fun, device):
    """
    Evaluates the model using the testing dataset and calculates its mean loss.

    Parameters:
    - model_option: String specifying if the vae or the vq-vae model is being trained.
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
            # logvar - logarithm of the variance of the latent space distribution
            # vq_loss - L2 error between the embedding space and the encoder outputs
            if model_option == 'vae':
                recon_batch, mu, logvar = model(image)
                loss, _, _ = loss_fun(recon_batch, image, mu, logvar)
            else:
                recon_batch, vq_loss = model(image)
                loss, _, _ = loss_fun(recon_batch, image, vq_loss)

            total_loss += loss.item()

    return total_loss / len(test_loader)
