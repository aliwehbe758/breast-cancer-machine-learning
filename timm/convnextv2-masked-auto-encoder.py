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

    def apply_grid_mask(self, x, mask_ratio=0.5):
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

        return x_masked, mask

    def forward(self, x):
        masked_x, mask = self.apply_grid_mask(x)
        features = self.encoder(masked_x)
        reconstructed = self.decoder(features)
        return reconstructed, mask, masked_x

def show_images(original, mask, masked_image, means, stds, reconstructed, n):
    fig, axes = plt.subplots(n, 4, figsize=(15, n*3))
    for i in range(n):
        mean = means[i].item()
        std = stds[i].item()

        # Original image
        orig_img = original[i].cpu().numpy().transpose(1, 2, 0)
        orig_img = (orig_img * std) + mean  # Denormalize
        axes[i, 0].imshow(orig_img.squeeze(), cmap='gray')
        axes[i, 0].set_title("Original Image")
        axes[i, 0].axis('off')

        # Mask
        mask_img = mask[i].cpu().numpy().transpose(1, 2, 0)
        axes[i, 1].imshow(mask_img.squeeze(), cmap='gray')
        axes[i, 1].set_title("Mask")
        axes[i, 1].axis('off')

        # Masked image
        masked_img = masked_image[i].cpu().numpy().transpose(1, 2, 0)
        masked_img = (masked_img * std) + mean  # Denormalize
        axes[i, 2].imshow(masked_img.squeeze(), cmap='gray')
        axes[i, 2].set_title("Masked Image")
        axes[i, 2].axis('off')

        # Reconstructed image
        recon_img = reconstructed[i].cpu().detach().numpy().transpose(1, 2, 0)
        recon_img = (recon_img * std) + mean  # Denormalize
        axes[i, 3].imshow(recon_img.squeeze(), cmap='gray')
        axes[i, 3].set_title("Reconstructed Image")
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig('./convnextv2-masked-auto-encoder-reconstruction.png')


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
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_transform = transforms.Compose([
        ResizeWithPad(new_shape=(input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
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
    model = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', pretrained=True)

    # Remove the classifier head
    model.reset_classifier(num_classes=0, global_pool='')

    # Initialize the model
    reconstruction_model = ImageReconstructionModel(model).to(device)

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
            outputs, _, _ = reconstruction_model(inputs)
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
            torch.save(reconstruction_model.state_dict(), './convnextv2-masked-auto-encoder.pth')

            pretrained_convnext_backbone = reconstruction_model.encoder
            state_dict = {"convnext_parameters": pretrained_convnext_backbone.state_dict()}
            torch.save(state_dict(), './convnextv2-masked-auto-encoder-backbone.pth')

            print(f'Best model saved with validation loss: {best_loss:.4f}')
            print('------------------------------------------------------------------------------------------------')
        else:
            print('------------------------------------------------------------------------------------------------')
    best_model = ImageReconstructionModel(model).to(device)
    best_model.load_state_dict(torch.load('./convnextv2-masked-auto-encoder.pth'))
    print(f"Model loaded from {'./convnextv2-masked-auto-encoder.pth'}.")
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
    plt.savefig('./convnextv2-masked-auto-encoder.png')

    # Get some images from the dataloader
    with torch.no_grad():
        for (inputs, means, stds), _ in test_loader:
            inputs = inputs.to(device)
            outputs, mask, masked_inputs = best_model(inputs)
            show_images(inputs, mask, masked_inputs, means, stds, outputs, batch_size)
            break
