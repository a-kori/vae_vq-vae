import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import create_datasets

    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn1(self.conv2(out))
        out += identity
        return out    

class Encoder(nn.Module):
    def __init__(self, input_channels, emb_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.resblock1 = ResidualBlock(256,256)
        self.resblock2 = ResidualBlock(256,256)
        self.conv3 = nn.Conv2d(256, emb_dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.conv3(x)
        return x

class Decoder(nn.Module):
    def __init__(self, output_channels, emb_dim):
        super(Decoder, self).__init__()
        self.deconv1 = nn.Conv2d(emb_dim, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.resblock1 = ResidualBlock(256,256)
        self.resblock2 = ResidualBlock(256,256)
        self.deconv2 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.deconv1(x)))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = F.relu(self.bn2(self.deconv2(x)))
        x = torch.sigmoid(self.deconv3(x))
        return x    
    
class VectorQuantizer(nn.Module):
    def __init__(self, emb_dim, n_embed, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.emb_dim = emb_dim
        self.n_embed = n_embed
        self.embedding = nn.Embedding(self.n_embed, self.emb_dim)
        self.embedding.weight.data.uniform_(-1/self.n_embed, 1/self.n_embed)
        self.commitment_cost = commitment_cost
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        input_shape = z.shape
        flat_input = z.view(-1, self.emb_dim)

        # Compute distances using torch.cdist, cdist requires shape [n,m,b] so unsqueeze both vectors
        distances = torch.cdist(flat_input.unsqueeze(0), self.embedding.weight.unsqueeze(0),p = 2).squeeze(0)
        
        # chose closest embedding vector 
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.n_embed, device=self.device)
        
        # Scatter encodings
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        z_quant = torch.matmul(encodings, self.embedding.weight).view(input_shape)

        # Loss calculation
        z_quant_detached = z_quant.detach()
        codebook_loss = F.mse_loss(z_quant, z.detach())
        commitment_loss = F.mse_loss(z, z_quant_detached)
        loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator
        z_quant = z + (z_quant - z).detach()
        z_quant = z_quant.permute(0, 3, 1, 2).contiguous()
        return z_quant, loss, codebook_loss, commitment_loss
        
class VQVAE(nn.Module):
    def __init__(self, input_channels, emb_dim, n_embed, commitment_cost):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(input_channels,emb_dim)
        self.decoder = Decoder(input_channels,emb_dim)
        self.vectorquantizer = VectorQuantizer(emb_dim, n_embed, commitment_cost)

    def forward(self, x):
        z = self.encoder(x)
        z_quant, loss,embedding_loss,commitment_loss = self.vectorquantizer.forward(z)
        x_reconstructed = self.decoder(z_quant)
        return x_reconstructed, loss,embedding_loss,commitment_loss
#debugging purposes
'''
def main():
    # Configuration
    input_channels = 3
    emb_dim = 64
    n_embed = 512
    commitment_cost = 0.25
    batch_size = 128  # Just for demonstration, usually larger for training
    image_size = 32  # CIFAR-10 image size
    
    # Create the VQ-VAE model
    model = VQVAE(input_channels, emb_dim, n_embed, commitment_cost).to('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = 'CIFAR10'
    train_loader, test_loader = create_datasets(dataset_name, batch_size)

    images, labels = next(iter(train_loader))  # Fetch the next batch
    
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images = images.to(device)
    model = model.to(device)
    
    reconstructed_images, loss, embedding_loss, commitment_loss = model(images)
    
if __name__ == "__main__":
    main()
    
    '''