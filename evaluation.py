import torch


def evaluate(model, test_loader, loss_fun, device):
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
        for images, _ in test_loader:
            # Move images to the appropriate device
            images = images.to(device)
            recon_batch, mu, logvar = model(images)
            loss = loss_fun(recon_batch, images, mu, logvar)
            total_loss += loss.item()

    return total_loss / len(test_loader)
