import torch
import torch.nn as nn
import torch.nn.functional as F


# Observation: Higher model complexity causes overfitting
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
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.resblock1 = ResidualBlock(256, 256)
        self.resblock2 = ResidualBlock(256, 256)
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
        self.resblock1 = ResidualBlock(256, 256)
        self.resblock2 = ResidualBlock(256, 256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.bn1(self.deconv1(x)))
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = F.relu(self.bn2(self.deconv2(x)))
        x = torch.sigmoid(self.deconv3(x))
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, emb_dim, n_embed, beta):
        super(VectorQuantizer, self).__init__()
        self.emb_dim = emb_dim
        self.n_embed = n_embed
        self.embedding = nn.Embedding(self.n_embed, self.emb_dim)
        self.embedding.weight.data.normal_(mean=0, std=1 / self.n_embed)
        self.beta = beta
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, encoderout):
        encoderout = encoderout.permute(0, 2, 3, 1).contiguous()
        flat_input = encoderout.view(-1, self.emb_dim)

        # Compute distances using torch.cdist, cdist requires shape [n,m,b] so unsqueeze both vectors
        distances = torch.cdist(flat_input.unsqueeze(0), self.embedding.weight.unsqueeze(0), p=2).squeeze(0)

        # chose closest embedding vector
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        closest_embedding = self.embedding(encoding_indices).view(encoderout.shape)

        # Loss calculation
        # codebook loss adapts embedding vectors to encoder output
        codebook_loss = F.mse_loss(closest_embedding, encoderout.detach())
        # commitment loss adapts encoder output to embeddings
        commitment_loss = F.mse_loss(encoderout, closest_embedding.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # Straight-through estimator
        # Observation: Straight through estimator is causing a gradual increase in vq loss
        closest_embedding = encoderout + (closest_embedding - encoderout).detach()
        closest_embedding = encoderout.permute(0, 3, 1, 2).contiguous()

        return closest_embedding, loss, codebook_loss, commitment_loss

    def get_embedding(self, indices):
        return self.embedding(indices)


class VQVAE(nn.Module):
    def __init__(self, input_channels, emb_dim, n_embed, beta):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(input_channels, emb_dim)
        self.decoder = Decoder(input_channels, emb_dim)
        self.vectorquantizer = VectorQuantizer(emb_dim, n_embed, beta)

    def forward(self, x):
        z = self.encoder(x)
        z_quant, loss, embedding_loss, commitment_loss = self.vectorquantizer.forward(z)
        x_reconstructed = self.decoder(z_quant)
        return x_reconstructed, loss, embedding_loss, commitment_loss

    def sample(self, num_samples):
        # Randomly sample indices from the embedding space
        indices = torch.randint(0, self.vectorquantizer.n_embed, (num_samples, 1)).to(self.vectorquantizer.device)
        embeddings = self.vectorquantizer.get_embedding(indices).view(num_samples, self.vectorquantizer.emb_dim, 1, 1)
        embeddings = embeddings.to(self.vectorquantizer.device)

        # Decode the embeddings to generate samples
        generated_samples = self.decoder(embeddings.to(self.vectorquantizer.device))
        return generated_samples