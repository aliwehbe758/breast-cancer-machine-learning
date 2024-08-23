import os
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset, Dataset, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from timm import create_model
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

dataset_root_path = '../Dataset_BUSI_with_GT/'

def get_train_and_validation_loader(transform_training_data, transform_validation_data, batch_size):
    # Load the dataset
    dataset = ImageFolder(root=dataset_root_path)

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Apply transforms separately
    train_dataset.dataset.transform = transform_training_data
    val_dataset.dataset.transform = transform_validation_data

    # Create data loaders
    batch_size = batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_loader, val_loader = get_train_and_validation_loader(transform, transform, batch_size=32)

# Parameters
batch_size = 32
num_epochs = 20
learning_rate = 1e-4
num_classes = 3
model_name = 'convnextv2_tiny.fcmae_ft_in22k_in1k'
val_split_ratio = 0.2

# Transformations
config = resolve_data_config({}, model=model_name)
train_transform = create_transform(**config, is_training=True)
val_transform = create_transform(**config, is_training=False)

# Dataset
full_dataset = datasets.ImageFolder(root=dataset_dir, transform=train_transform)

# Train-Validation Split
val_size = int(len(full_dataset) * val_split_ratio)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Apply separate transforms for train and val datasets
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Model
model = create_model(model_name, pretrained=True, num_classes=num_classes)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Validation Accuracy: {100 * correct / total:.2f}%')

# Save the trained model
torch.save(model.state_dict(), './models/timm-convnext-v2.pth')
print('Model saved to ./models/timm-convnext-v2.pth')
