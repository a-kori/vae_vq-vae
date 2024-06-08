import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class Encoder(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2_mean = nn.Linear(512, latent_dim)
        self.fc2_logvar = nn.Linear(512, latent_dim)

    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.relu(self.conv3(x))
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)  # Flatten the tensor
        h = F.relu(self.fc1(x))
        # print(h.shape)
        mean = self.fc2_mean(h)
        logvar = self.fc2_logvar(h)
        return mean, logvar
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 512)
        self.fc2 = nn.Linear(512, 128 * 3 * 3)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, z):
        #print(z.shape)
        z = F.relu(self.fc1(z))
        #print(z.shape)
        z = F.relu(self.fc2(z))
        #print(z.shape)
        z = z.view(z.size(0), 128, 3, 3)  # Reshape tensor
        #print(z.shape)
        z = F.relu(self.deconv1(z))
        #print(z.shape)
        z = F.relu(self.deconv2(z))
        #print(z.shape)
        z = torch.sigmoid(self.deconv3(z))
        #print(z.shape)
        return z

class VAE(nn.Module):
    def __init__(self, input_channels, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        #print(x.shape)
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        #print(z.shape)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mean, logvar

def loss_function(reconstructed_x, x, mean, logvar):
    #print(reconstructed_x.shape)
    #print(x.shape)
    BCE = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return BCE + KLD
