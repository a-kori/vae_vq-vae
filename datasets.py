import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def visualize_dataset(dataloader):
    images, labels = next(iter(dataloader))

    fig, axes = plt.subplots(1, len(images), figsize=(12, 2.5))

    for i in range(len(images)):
        ax = axes[i]
        img = images[i].numpy().transpose((1, 2, 0))
        ax.imshow(img)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def create_dataset(dataset_name):
    return DataLoader(dataset_name, batch_size=10, shuffle=False)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.Normalize((0.4, 0.4, 0.4), (0.4, 0.4, 0.4)),
])
training_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)

training_dataloader = create_dataset(training_dataset)
test_dataloader = create_dataset(test_dataset)

visualize_dataset(training_dataloader)
visualize_dataset(test_dataloader)
