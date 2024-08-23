import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, Dataset
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import timm
from timm import create_model
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from PIL import Image

dataset_root_path = '../Dataset_BUSI_with_GT/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define paths
dataset_path = dataset_root_path
folders = ["benign", "malignant", "normal"]

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# Load datasets
dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Load the pre-trained model
model = timm.create_model('convnextv2_tiny.fcmae_ft_in22k_in1k', pretrained=True)

# Remove the classifier head
model.reset_classifier(0)

# Add a reconstruction head (simple example)
class ImageReconstructionModel(nn.Module):
    def __init__(self, base_model):
        super(ImageReconstructionModel, self).__init__()
        self.encoder = base_model
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7x7 -> 14x14
            nn.ReLU(True),
            nn.ConvTranspose2d(384, 192, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.ReLU(True),
            nn.ConvTranspose2d(192, 96, kernel_size=3, stride=2, padding=1, output_padding=1),   # 28x28 -> 56x56
            nn.ReLU(True),
            nn.ConvTranspose2d(96, 48, kernel_size=3, stride=2, padding=1, output_padding=1),    # 56x56 -> 112x112
            nn.ReLU(True),
            nn.ConvTranspose2d(48, 24, kernel_size=3, stride=2, padding=1, output_padding=1),    # 112x112 -> 224x224
            nn.ReLU(True),
            nn.ConvTranspose2d(24, 1, kernel_size=1, stride=1),  # Output 224x224 with 1 channel (grayscale)
            nn.Tanh()
        )

    def forward(self, x):
        features = self.encoder.forward_features(x)
        reconstructed = self.decoder(features)
        return reconstructed

# Initialize the model
reconstruction_model = ImageReconstructionModel(model).cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(reconstruction_model.parameters(), lr=1e-4)

num_epochs = 100

# Re-run training and visualize
for epoch in range(num_epochs):
    reconstruction_model.train()
    running_loss = 0.0

    for inputs, _ in dataloader:
        inputs = inputs.cuda()

        # Forward pass
        outputs = reconstruction_model(inputs)
        loss = criterion(outputs, inputs)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

def show_images(original, reconstructed, n=5):
    fig, axes = plt.subplots(n, 2, figsize=(10, 15))
    for i in range(n):
        # Original image
        orig_img = original[i].cpu().numpy().transpose(1, 2, 0)
        orig_img = (orig_img * 0.5) + 0.5  # Denormalize
        axes[i, 0].imshow(orig_img.squeeze(), cmap='gray')
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')

        # Reconstructed image
        recon_img = reconstructed[i].cpu().detach().numpy().transpose(1, 2, 0)
        recon_img = (recon_img * 0.5) + 0.5  # Denormalize
        axes[i, 1].imshow(recon_img.squeeze(), cmap='gray')
        axes[i, 1].set_title("Reconstructed Image")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()

# Get some images from the dataloader
reconstruction_model.eval()
with torch.no_grad():
    for inputs, _ in dataloader:
        inputs = inputs.cuda()
        outputs = reconstruction_model(inputs)
        show_images(inputs, outputs)
        break

# Save the trained model
torch.save(model.state_dict(), './models/timm-convnext-v2-mae.pth')
print('Model saved to ./models/timm-convnext-v2-mae.pth')