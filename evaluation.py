import torch

def evaluate(model, test_loader, loss_fun, device):
    """
    Parameters:
    -----------
    model : The nn model to be evaluated.
    test_loader : DataLoader providing the test dataset.
    loss_fun : The loss function used to compute the loss.
    Returns the mean testing loss.
    """

    total_loss = 0

    # Disable gradient computation.
    with torch.no_grad():
        # Switch to evaluation mode.
        model.eval()
        # Iterate over the entire test dataset.
        for images, labels in test_loader:
            # Move images to the appropriate device
            images = images.to(device)
            output = model(images)
            loss = loss_fun(output, labels)
            total_loss += loss.item()

    return total_loss / len(test_loader)
