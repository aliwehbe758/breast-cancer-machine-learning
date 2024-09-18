import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
from lightly.data import LightlyDataset
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform

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
        x0, x1 = image
        if not isinstance(x0, torch.Tensor) or not isinstance(x1, torch.Tensor):
            raise TypeError("Expected inputs to be a torch.Tensor")

        return self.normalize(x0), self.normalize(x1)

    def normalize(self, image):
        mean = image.mean()
        std = image.std()
        normalized_image = (image - mean) / std
        return normalized_image

class SimCLR(nn.Module):
    def __init__(self, backbone, projection_dim=128):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(147456, 512, projection_dim)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

def contrastive_accuracy(z0, z1):
    """Computes the accuracy of the model in finding the positive pairs."""
    batch_size = z0.shape[0]
    z0 = F.normalize(z0, dim=1)
    z1 = F.normalize(z1, dim=1)

    # Compute cosine similarity matrix between z0 and z1
    sim_matrix = torch.mm(z0, z1.T)  # shape: (batch_size, batch_size)

    # For each row, the diagonal element is the positive pair
    positive_sim = torch.diag(sim_matrix)  # shape: (batch_size,)

    # Contrastive accuracy is the percentage of times the highest similarity is the positive pair
    correct = (positive_sim == sim_matrix.max(dim=1)[0]).float().mean()

    return correct

def cosine_similarity(z0, z1):
    """Compute the cosine similarity between the embeddings."""
    z0 = F.normalize(z0, dim=1)
    z1 = F.normalize(z1, dim=1)
    return F.cosine_similarity(z0, z1).mean()

def alignment(z0, z1):
    """Measures alignment: how close the embeddings of positive pairs are."""
    return F.mse_loss(z0, z1)

def uniformity(z, t=2):
    """Measures uniformity: how uniformly the embeddings are spread on the unit hypersphere."""
    z = F.normalize(z, dim=1)
    dist = torch.pdist(z, p=2).pow(2)
    return dist.mul(-t).exp().mean().log()


if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_workers = 8
    batch_size = 16
    seed = 42
    input_size = 384
    num_epochs = 2000
    lr = 1e-4

    pl.seed_everything(seed)

    simclr_transform = SimCLRTransform(input_size=input_size, min_scale=1.0, vf_prob=0.5, rr_prob=0.5, normalize=None)
    transform = transforms.Compose([
        ResizeWithPad(new_shape=(input_size, input_size)),
        simclr_transform,
        CustomNormalize()
    ])
    dataset = LightlyDataset(input_dir='./Dataset_BUSI_BrEaST',
                             transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    convnextv2 = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', pretrained=True)
    backbone = nn.Sequential(*list(convnextv2.children())[:-1])
    model = SimCLR(backbone).to(device)
    criterion = NTXentLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    best_loss = float('inf')  # Initialize with infinity

    print("Starting Training")
    losses = []
    accuracies = []
    cosine_sims = []
    alignments = []
    uniformities = []

    for epoch in range(2):
        total_loss = 0
        total_accuracy = 0
        total_cosine_sim = 0
        total_alignment = 0
        total_uniformity = 0

        for batch in data_loader:
            x0, x1 = batch[0]
            x0 = x0.to(device)
            x1 = x1.to(device)

            z0 = model(x0)
            z1 = model(x1)

            loss = criterion(z0, z1)
            total_loss += loss.detach()

            # Calculate additional metrics
            acc = contrastive_accuracy(z0, z1)
            cos_sim = cosine_similarity(z0, z1)
            align = alignment(z0, z1)
            unif = uniformity(z0)

            total_accuracy += acc.detach()
            total_cosine_sim += cos_sim.detach()
            total_alignment += align.detach()
            total_uniformity += unif.detach()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(data_loader)
        avg_acc = total_accuracy / len(data_loader)
        avg_cosine_sim = total_cosine_sim / len(data_loader)
        avg_align = total_alignment / len(data_loader)
        avg_unif = total_uniformity / len(data_loader)

        losses.append(avg_loss.cpu())
        accuracies.append(avg_acc.cpu())
        cosine_sims.append(avg_cosine_sim.cpu())
        alignments.append(avg_align.cpu())
        uniformities.append(avg_unif.cpu())

        print(f"Epoch: [{epoch + 1}/{num_epochs}], loss: {avg_loss:.5f}, contrastive accuracy: {avg_acc:.5f}, "
              f"cosine similarity: {avg_cosine_sim:.5f}, alignment: {avg_align:.5f}, uniformity: {avg_unif:.5f}")

        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            pretrained_convnext_backbone = model.backbone
            state_dict = {"convnext_parameters": pretrained_convnext_backbone.state_dict()}
            torch.save(state_dict, "./covnextv2-simclr.pth")
            print(f'Best model saved with loss: {best_loss:.4f}')
    epochs = range(1, num_epochs + 1)

    # Increase the figure size drastically to make the plots significantly larger
    plt.figure(figsize=(20, 80))  # Larger width and height for bigger plots

    # Plot training losses
    plt.subplot(5, 1, 1)  # 5 rows, 1 column, position 1
    plt.plot(epochs, losses, 'r-', label='Train Loss')
    plt.title('Loss', fontsize=24)  # Larger title font size
    plt.xlabel('Epochs', fontsize=20)  # Larger x-axis label font size
    plt.ylabel('Loss', fontsize=20)  # Larger y-axis label font size
    plt.xticks(fontsize=16)  # Larger tick font size
    plt.yticks(fontsize=16)

    # Plot training accuracies
    plt.subplot(5, 1, 2)
    plt.plot(epochs, accuracies, 'r-')
    plt.title('Contrastive Accuracy', fontsize=24)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Accuracy', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Plot training cosine similarity scores
    plt.subplot(5, 1, 3)
    plt.plot(epochs, cosine_sims, 'r-')
    plt.title('Cosine Similarity Score', fontsize=24)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Cosine Similarity Score', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Plot training alignments
    plt.subplot(5, 1, 4)
    plt.plot(epochs, alignments, 'r-')
    plt.title('Alignment', fontsize=24)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Alignment', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    # Plot training uniformities
    plt.subplot(5, 1, 5)
    plt.plot(epochs, uniformities, 'r-')
    plt.title('Uniformity', fontsize=24)
    plt.xlabel('Epochs', fontsize=20)
    plt.ylabel('Uniformity', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.tight_layout()
    plt.savefig('./covnextv2-simclr.png')  # Save the plot as an image

    # load the model
    convnext_new = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', pretrained=True)
    backbone_new = nn.Sequential(*list(convnext_new.children())[:-1])
    ckpt = torch.load("./covnextv2-simclr.pth")
    backbone_new.load_state_dict(ckpt["convnext_parameters"])
