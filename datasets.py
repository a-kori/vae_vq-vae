import torch, itertools
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def visualize_dataset(dataset, name):
    '''
    Visualizes the given dataset by showing sample images in a grid.
    '''
    class_names = dataset.classes
    images, labels = zip(*tuple(itertools.islice(iter(dataset), 16)))
    images = torch.stack(images)
    grid = utils.make_grid(images, nrow=4)
    grid = (grid + 1) / 2
    np_grid = grid.permute(1, 2, 0).numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(np_grid)
    plt.title(f'Images from the {name} dataset')
    plt.axis('off')
    img_height = np_grid.shape[0] // 4
    img_width = np_grid.shape[1] // 4
    for i, label in enumerate(labels):
        row = i // 4
        col = i % 4
        x_pos = (col * img_width) + (img_width / 2)
        y_pos = (row + 1) * img_height - (img_height / 8)
        plt.text(x_pos, y_pos, class_names[label], ha='center', va='top', fontsize=10, color='white')


def create_datasets(dataset_name, batch_size):
    '''
    Creates data loaders for training and testing datasets with the given batch size.
    Also visualizes the training dataset.
    '''
    if str.upper(dataset_name) == 'MNIST':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor()
        ])
        train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
        visualize_dataset(train_dataset, 'MNIST training')

    elif str.upper(dataset_name) == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
        visualize_dataset(train_dataset, 'CIFAR10 training')
    else:
        return None, None
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader
