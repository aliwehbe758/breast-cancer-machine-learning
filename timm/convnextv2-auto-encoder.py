import os
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchmetrics import MeanAbsoluteError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import timm
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ResizeWithPad(nn.Module):
    def __init__(self, new_shape, float_output=False):
        super(ResizeWithPad, self).__init__()
        self.new_shape = new_shape
        self.float_output = float_output

    def forward(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)

        image = self.resize_with_pad(image, self.new_shape, self.float_output)

        return Image.fromarray(image)

    def resize_with_pad(self, image, new_shape, float_output=False):
        original_shape = (image.shape[1], image.shape[0])
        ratio_width = float(new_shape[0]) / original_shape[0]
        ratio_height = float(new_shape[1]) / original_shape[1]
        ratio = min(ratio_width, ratio_height)
        new_size = tuple([round(x * ratio) for x in original_shape])

        if ratio_width > 1 and ratio_height > 1:
            result = image
            delta_w = new_shape[0] - original_shape[0]
            delta_h = new_shape[1] - original_shape[1]
        else:
            result = self.resize_channels(image, new_size, filter_name="bilinear")
            delta_w = new_shape[0] - new_size[0]
            delta_h = new_shape[1] - new_size[1]


        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        result_padding = np.pad(result, ((top, bottom), (left, right), (0, 0)), mode='constant')
        if not float_output:
            result_padding = np.uint8(np.rint(result_padding.clip(0., 255.)))

        return result_padding

    def resize_single_channel(self, x_np, output_size, filter_name="bicubic"):
        s1, s2 = output_size
        img = Image.fromarray(x_np.astype(np.float32), mode='F')
        img = img.resize(output_size, resample=Image.BICUBIC)
        return np.asarray(img).clip(0, 255).reshape(s2, s1, 1)

    def resize_channels(self, x, new_size, filter_name="bilinear"):
        x = [self.resize_single_channel(x[:, :, idx], new_size, filter_name=filter_name) for idx in range(x.shape[2])]
        x = np.concatenate(x, axis=2).astype(np.float32)
        return x


class CustomNormalize(nn.Module):
    def __init__(self):
        super(CustomNormalize, self).__init__()

    def forward(self, image):
        if not isinstance(image, torch.Tensor):
            raise TypeError("Expected input to be a torch.Tensor")

        normalized_image, mean, std = self.normalize(image)
        return normalized_image, mean, std

    def normalize(self, image):
        mean = image.mean()
        std = image.std()
        normalized_image = (image - mean) / std
        return normalized_image, mean, std

class ImageReconstructionModel(nn.Module):
    def __init__(self, base_model):
        super(ImageReconstructionModel, self).__init__()
        self.encoder = base_model
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7x7 -> 14x14
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),  # 14x14 -> 28x28
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),  # 28x28 -> 56x56
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),   # 56x56 -> 112x112
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),    # 112x112 -> 224x224
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 3, kernel_size=1, stride=1),  # Output 224x224 with 3 channels (RGB)
            nn.Tanh()
        )

    def forward(self, x):
        features = self.encoder.forward_features(x)
        reconstructed = self.decoder(features)
        return reconstructed

def show_images(original, means, stds, reconstructed, n):
    fig, axes = plt.subplots(n, 2, figsize=(8, n*3))
    for i in range(n):
        mean = means[i].item()
        std = stds[i].item()
        # Original image
        orig_img = original[i].cpu().numpy().transpose(1, 2, 0)
        orig_img = (orig_img * std) + mean  # Denormalize
        axes[i, 0].imshow(orig_img.squeeze(), cmap='gray')
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')

        # Reconstructed image
        recon_img = reconstructed[i].cpu().detach().numpy().transpose(1, 2, 0)
        recon_img = (recon_img * std) + mean  # Denormalize
        axes[i, 1].imshow(recon_img.squeeze(), cmap='gray')
        axes[i, 1].set_title("Reconstructed Image")
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig('./convnextv2-auto-encoder-reconstruction.png')

if __name__=='__main__':

    num_workers = 8
    batch_size = 16
    seed = 42
    input_size = 384
    num_epochs = 500
    lr = 1e-4

    seed_everything(42)

    # Path to the main directory
    main_dir = "./Dataset_BUSI_BrEaST_Training_Validation"

    # Define the subdirectories
    train_dir = os.path.join(main_dir, 'train')
    test_dir = os.path.join(main_dir, 'validation')

    # Define transformations
    train_transform = transforms.Compose([
        ResizeWithPad(new_shape=(input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),  # Add vertical flip
        transforms.RandomRotation(15),  # Increase rotation angle
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Slightly stronger jitter
        transforms.ToTensor(),
        CustomNormalize()
    ])
    test_transform = transforms.Compose([
        ResizeWithPad(new_shape=(input_size, input_size)),
        transforms.ToTensor(),
        CustomNormalize()
    ])

    # Load the datasets
    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = ImageFolder(root=test_dir, transform=test_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Check the dataset sizes
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(test_dataset)}")

    # Load the pre-trained model
    convnextv2 = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', pretrained=True)

    # Remove the classifier head
    convnextv2.reset_classifier(0)

    # Initialize the model
    reconstruction_model = ImageReconstructionModel(convnextv2).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.AdamW(reconstruction_model.parameters(), lr=lr)

    # Initialize metrics
    train_mae_metric = MeanAbsoluteError().to(device)
    train_psnr_metric = PeakSignalNoiseRatio().to(device)
    train_ssim_metric = StructuralSimilarityIndexMeasure().to(device)

    best_loss = float('inf')  # Initialize with infinity

    # Lists to store metrics for plotting
    train_losses = []
    train_mae = []
    train_psnr = []
    train_ssim = []

    # Training and validation loop
    for epoch in range(num_epochs):
        reconstruction_model.train()
        train_loss = 0.0
        # Reset metrics at the start of each epoch
        train_mae_metric.reset()
        train_psnr_metric.reset()
        train_ssim_metric.reset()

        for (inputs, means, stds), _ in train_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = reconstruction_model(inputs)
            loss = criterion(outputs, inputs)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # Update metrics
            inputs = inputs.float().contiguous()
            outputs = outputs.float().contiguous()

            train_mae_metric.update(outputs, inputs)
            train_psnr_metric.update(outputs, inputs)
            train_ssim_metric.update(outputs, inputs)

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        epoch_train_mae = train_mae_metric.compute().item()
        epoch_train_psnr = train_psnr_metric.compute().item()
        epoch_train_ssim = train_ssim_metric.compute().item()

        train_mae.append(epoch_train_mae)
        train_psnr.append(epoch_train_psnr)
        train_ssim.append(epoch_train_ssim)

        # Print metrics
        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(
            f'Loss: {train_loss:.4f}, MAE: {epoch_train_mae:.4f}, PSNR: {epoch_train_psnr:.4f}, SSIM: {epoch_train_ssim:.4f}')

        # Save the best model based on validation loss
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(reconstruction_model.state_dict(), './convnextv2-auto-encoder.pth')
            print(f'Best model saved with validation loss: {best_loss:.4f}')
            print('------------------------------------------------------------------------------------------------')
        else:
            print('------------------------------------------------------------------------------------------------')

    best_model = ImageReconstructionModel(convnextv2).to(device)
    best_model.load_state_dict(torch.load('./convnextv2-auto-encoder.pth'))
    print(f"Model loaded from {'./convnextv2-auto-encoder.pth'}.")
    best_model.eval()

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(11, 30))

    # Plot losses
    plt.subplot(4, 1, 1)  # 4 rows, 1 column, position 1
    plt.plot(epochs, train_losses, 'r-')
    plt.title('Loss', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Loss', fontsize=14)

    # Plot MAE
    plt.subplot(4, 1, 2)
    plt.plot(epochs, train_mae, 'r-')
    plt.title('MAE', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('MAE', fontsize=14)

    # Plot PSNR
    plt.subplot(4, 1, 3)
    plt.plot(epochs, train_psnr, 'r-')
    plt.title('PSNR', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('PSNR', fontsize=14)

    # Plot SSIM
    plt.subplot(4, 1, 4)
    plt.plot(epochs, train_ssim, 'r-')
    plt.title('SSIM', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('SSIM', fontsize=14)

    plt.tight_layout()
    plt.savefig('./convnextv2-auto-encoder.png')

    # Get some images from the dataloader
    with torch.no_grad():
        for (inputs, means, stds), _ in test_loader:
            inputs = inputs.to(device)
            outputs = best_model(inputs)
            show_images(inputs, means, stds, outputs, batch_size)
            break