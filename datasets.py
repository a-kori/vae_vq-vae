import torch, itertools
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt

def visualize_dataset(dataset, name):
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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

training_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)

visualize_dataset(training_dataset, 'training')
