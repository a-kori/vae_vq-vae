import torch

def evaluate(model, test_loader, loss_fun):
    """
    Parameters:
    -----------
    model : The nn model to be evaluated.
    test_loader : DataLoader providing the test dataset.
    loss_fun : The loss function used to compute the loss.
    """

    total_loss = 0

    # Disable gradient computation.
    with torch.no_grad():
        # Switch to evaluation mode.
        model.eval()
        # Iterate over the entire test dataset.
        for images, labels in test_loader:
            output = model(images)
            loss = loss_fun(output, labels)
            total_loss += loss.item()

    mean_loss = total_loss / len(test_loader)

    return mean_loss
