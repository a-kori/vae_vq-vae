import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from sklearn.cluster import KMeans
import numpy as np

class InitializeCodebook:
    def __init__(self, codebook_size, patch_size=4, sample_size=10000):
        self.codebook_size = codebook_size  # Anzahl der Vektoren im Codebuch
        self.patch_size = patch_size  # Größe der Patches, die aus den Bildern extrahiert werden
        self.embedding_dim = patch_size * patch_size * 3  # Dimension jedes Vektors im Codebuch (Patchgröße x Patchgröße x 3 Farbkanäle)
        self.sample_size = sample_size  # Anzahl der Patches, die für das K-Means-Clustering verwendet werden sollen
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Transformiere die Bilder in Tensoren
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalisiere die Bilder (optional)
        ])

    def load_data(self):
        # Lade das CIFAR-10-Dataset
        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=self.transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=10000, shuffle=True, num_workers=2)
        data_iter = iter(trainloader)  # Erstelle einen Iterator für den DataLoader
        images, _ = next(data_iter)  # Hole einen Batch von Bildern
        return images  # Rückgabe der Bilder

    def extract_patches(self, images):
        # Extrahiere zufällige Patches aus den Bildern
        patches = []  # Liste zur Speicherung der Patches
        _, _, h, w = images.shape  # Hole die Höhe und Breite der Bilder

        images = images.permute(0, 2, 3, 1)  # Ändere die Dimensionen von (N, C, H, W) zu (N, H, W, C)
        # Die Permutation der Dimensionen von (N, C, H, W) zu (N, H, W, C) ist in vielen Bildverarbeitungsanwendungen üblich, 
        # insbesondere wenn man von einem Framework wie PyTorch, das standardmäßig (N, C, H, W) verwendet, zu numpy wechselt, 
        # das oft (H, W, C) bevorzugt. Diese Transformation erleichtert die Integration mit numpy und anderen Bildverarbeitungsbibliotheken, 
        # die (H, W, C) erwarten. (s. https://discuss.pytorch.org/t/why-does-pytorch-prefer-using-nchw/83637 )
        
        for image in images:
            for _ in range(10):  # Extrahiere 10 Patches pro Bild
                top = np.random.randint(0, h - self.patch_size)  # Zufällige y-Koordinate für den Patch
                left = np.random.randint(0, w - self.patch_size)  # Zufällige x-Koordinate für den Patch
                patch = image[top:top + self.patch_size, left:left + self.patch_size].flatten()  # Extrahiere und flatten den Patch
                patches.append(patch.numpy())  # Füge den Patch zur Liste hinzu
        patches = np.array(patches)  # Konvertiere die Liste in ein numpy-Array
        return patches  # Rückgabe der Patches

    def initialize_codebook(self):
   
        data = self.load_data()
        patches = self.extract_patches(data)
        
        # Ziehe eine zufällige Stichprobe der Patches
        if len(patches) > self.sample_size:
            indices = np.random.choice(len(patches), self.sample_size, replace=False)  # Zufällige Auswahl von Patches
            patches_sample = patches[indices]
        else:
            patches_sample = patches
        
        # Wende K-Means auf die Patches an
        kmeans = KMeans(n_clusters=self.codebook_size, random_state=0)
        kmeans.fit(patches_sample)  # Fitte das K-Means-Modell auf die Patches
        
        # Initialisiere das Codebuch mit den Clusterzentren und füge zufällige Störung hinzu
        codebook = kmeans.cluster_centers_ + 0.01 * np.random.randn(self.codebook_size, self.embedding_dim)
        codebook = torch.tensor(codebook, dtype=torch.float32)  # Konvertiere das Codebuch in einen PyTorch-Tensor
        return codebook  

codebook_size = 512  # Anzahl der Vektoren im Codebuch
patch_size = 4  

initializer = InitializeCodebook(codebook_size, patch_size)  # Initialisiere die Klasse mit den gegebenen Parametern
codebook = initializer.initialize_codebook()  # Initialisiere das Codebuch
print(f"Codebook initialized! Shape: {list(codebook.shape)}")  
