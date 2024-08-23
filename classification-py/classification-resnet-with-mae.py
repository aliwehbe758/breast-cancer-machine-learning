import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the transformation that only converts images to tensor
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.ToTensor()
])


dataset = ImageFolder(root='../Dataset_BUSI_with_GT/', transform=transform)
loader = DataLoader(dataset, batch_size=50, num_workers=4, shuffle=False)

# Function to calculate mean and std
def get_mean_and_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5

    return mean, std

mean, std = get_mean_and_std(loader)
print(f'Mean: {mean}')
print(f'Std Dev: {std}')

# Define transformations for the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean, std)  # Normalize images
])

# Load the dataset with the new transformation
dataset = ImageFolder(root='../Dataset_BUSI_with_GT/', transform=transform)


# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class MaskedAutoEncoder(nn.Module):
    def __init__(self):
        super(MaskedAutoEncoder, self).__init__()

        # Encoder
        # Load pre-trained ResNet50 and remove the fully connected layer
        self.encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-2])  # Remove the fully connected layer and avgpool


        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=3, stride=2, padding=1, output_padding=1),  # (1024, 8, 8)
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # (512, 16, 16)
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # (256, 32, 32)
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # (128, 64, 64)
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # (64, 128, 128)
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),  # (3, 128, 128)
            nn.Tanh()
        )

    def forward(self, x, mask_ratio=0.75):
        # Apply grid masking
        batch_size, _, H, W = x.shape
        patch_size = 16  # Define the patch size
        num_patches = (H // patch_size) * (W // patch_size)
        num_masked_patches = int(mask_ratio * num_patches)

        mask = torch.ones(batch_size, H // patch_size, W // patch_size, device=x.device)
        for i in range(batch_size):
            mask_indices = torch.randperm(num_patches)[:num_masked_patches]
            mask[i].view(-1)[mask_indices] = 0

        mask = mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
        mask = mask.unsqueeze(1)  # Add channel dimension

        x_masked = x * mask

        # Encode
        encoded = self.encoder(x_masked)

        # Decode
        decoded = self.decoder(encoded)

        return decoded, mask

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=3):  # Assuming 3 classes, change as needed
        super(ResNetClassifier, self).__init__()
        # Load pre-trained ResNet50
        self.encoder = models.resnet50(weights=None)  # Load without pretrained weights initially
        # Replace the fully connected layer
        num_ftrs = self.encoder.fc.in_features
        self.encoder.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.encoder(x)

# Instantiate the model
classifier = ResNetClassifier(num_classes=3)  # Adjust the number of classes as needed
classifier = classifier.to(device)

# Load pre-trained encoder weights
pretrained_model_path = '../masked-auto-encoder/models/mae-grid-masking-resnet50-40.pth'
model = MaskedAutoEncoder()
model.load_state_dict(torch.load(pretrained_model_path, map_location=device))

# Load encoder state dict selectively
encoder_state_dict = model.encoder.state_dict()
model_state_dict = classifier.encoder.state_dict()

# Filter out unnecessary keys
filtered_encoder_state_dict = {k: v for k, v in encoder_state_dict.items() if k in model_state_dict}

# Load the state dictionary
model_state_dict.update(filtered_encoder_state_dict)
classifier.encoder.load_state_dict(model_state_dict)

# Optionally, freeze the encoder layers during initial training
for param in classifier.encoder.parameters():
    param.requires_grad = True

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, classifier.parameters()), lr=1e-3)

# Unfreeze the encoder layers for fine-tuning after some initial training epochs
unfreeze_epoch = 5

# Training the classifier
num_epochs = 50
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    if epoch == unfreeze_epoch:
        for param in classifier.encoder.parameters():
            param.requires_grad = True

    classifier.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = classifier(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    train_accuracy = 100 * correct_train / total_train
    train_accuracies.append(train_accuracy)

    classifier.eval()
    running_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = classifier(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Calculate validation accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_losses.append(val_loss)
    val_accuracy = 100 * correct_val / total_val
    val_accuracies.append(val_accuracy)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%")

# Save the trained classifier model
torch.save(classifier.state_dict(), './models/resnet_classifier_busi.pth')
print('Classifier model saved to ./models/resnet_classifier_busi.pth')