import matplotlib.pyplot as plt
import torch

def plot_loss_lr(num_epochs, epoch_markers, train_losses, test_losses, learning_rates):
    '''
    Plots the Train Loss, Test Loss and Learning Rate development.
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle('Train Loss & Test Loss & LR')

    # Plot Train & Test losses
    ax1.set_title('Train & Test losses')
    ax1.set_xlabel('Epoch')
    ax1.plot(train_losses, label='Train Loss', color='C0')
    ax1.scatter(epoch_markers, [train_losses[i] for i in epoch_markers], color='C0')
    ax1.plot(epoch_markers, test_losses, label='Test Loss', color='C1')
    ax1.legend()

    # Plot Learning Rate over time
    ax2.set_title('Learning Rate over time')
    ax2.set_xlabel('Epoch')
    ax2.plot(range(1, num_epochs + 1), learning_rates, label='Learning Rate', color='C2')

    fig.tight_layout()
    plt.show()


def plot_loss_components(reconstruction_losses, kl_losses):
    '''
    Plots the development of the 2 loss components: the reconstruction loss and the KL divergence loss.
    '''
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    fig.suptitle('Loss Components')

    # Convert lists to tensors
    reconstruction_losses = torch.tensor(reconstruction_losses)
    kl_losses = torch.tensor(kl_losses)

    # Transfer to CPU and convert to NumPy arrays
    reconstruction_losses = reconstruction_losses.cpu().numpy()
    kl_losses = kl_losses.cpu().numpy()

    # Plot reconstruction loss
    ax1.set_title('Reconstruction Loss')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Reconstruction Loss')
    ax1.plot(range(len(reconstruction_losses)), reconstruction_losses, label='Reconstruction Loss', color='C0')

    # Plot KL divergence loss
    ax2.set_title('KL Divergence Loss')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('KL Divergence Loss')
    ax2.plot(range(len(kl_losses)), kl_losses, label='KL Divergence Loss', color='C1')

    fig.tight_layout()
    plt.show()
