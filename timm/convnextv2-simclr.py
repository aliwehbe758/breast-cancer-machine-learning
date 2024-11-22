import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import timm
import pytorch_lightning as pl
from lightly.data import LightlyDataset
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        if(isinstance(image, list)):
          x0, x1 = image
          if not isinstance(x0, torch.Tensor) or not isinstance(x1, torch.Tensor):
            raise TypeError("Expected inputs to be a torch.Tensor")

          return self.normalize(x0), self.normalize(x1)
        else:
          if not isinstance(image, torch.Tensor):
            raise TypeError("Expected inputs to be a torch.Tensor")

          return self.normalize(image)

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

def generate_embeddings(model, dataloader):
    """Generates representations for all images in the dataloader with
    the given model
    """

    embeddings = []
    filenames = []
    with torch.no_grad():
        for img, _, fnames in dataloader:
            img = img.to(device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings.cpu())
    return embeddings, filenames

def get_image_as_np_array(filename: str):
    """Returns an image as an numpy array"""
    img = Image.open(filename)
    return np.asarray(img)

def plot_knn_examples(embeddings, filenames, num_examples, n_neighbors=3):
    """Plots multiple rows of random images with their nearest neighbors"""
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)

    samples_idx = np.random.choice(len(indices), size=num_examples, replace=False)

    fig, axes = plt.subplots(num_examples, 3, figsize=(12, num_examples*3))
    i = 0
    for idx in samples_idx:
        # loop through their nearest neighbors
        for plot_x_offset, neighbor_idx in enumerate(indices[idx]):
            fname = os.path.join(test_dir, filenames[neighbor_idx])
            axes[i, plot_x_offset].imshow(get_image_as_np_array(fname))
            axes[i, plot_x_offset].set_title(f"d={distances[idx][plot_x_offset]:.3f}")
            axes[i, plot_x_offset].axis('off')
        i += 1

    plt.tight_layout()
    plt.savefig('./covnextv2-simclr-knn.png')  # Save the plot as an image

if __name__=='__main__':

    num_workers = 8
    batch_size = 16
    seed = 42
    input_size = 384
    num_epochs = 500
    lr = 1e-4

    pl.seed_everything(seed)

    # Path to the main directory
    main_dir = "./Dataset_BUSI_BrEaST_Training_Validation"

    # Define the subdirectories
    train_dir = os.path.join(main_dir, 'train')
    test_dir = os.path.join(main_dir, 'validation')

    # Define transformations
    simclr_transform = SimCLRTransform(input_size=input_size, min_scale=1.0, vf_prob=0.5, rr_prob=0.5, normalize=None)
    train_transform = transforms.Compose([
        ResizeWithPad(new_shape=(input_size, input_size)),
        simclr_transform,
        CustomNormalize()
    ])
    test_transform = transforms.Compose([
        ResizeWithPad(new_shape=(input_size, input_size)),
        transforms.ToTensor(),
        CustomNormalize()
    ])

    # Load the datasets
    train_dataset = LightlyDataset(input_dir=train_dir, transform=train_transform)
    test_dataset = LightlyDataset(input_dir=test_dir, transform=test_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Check the dataset sizes
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(test_dataset)}")

    convnextv2 = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', pretrained=True)
    convnextv2.reset_classifier(num_classes=0, global_pool='')
    model = SimCLR(convnextv2).to(device)

    criterion = NTXentLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    best_loss = float('inf')  # Initialize with infinity

    losses = []
    accuracies = []
    cosine_sims = []
    alignments = []
    uniformities = []

    for epoch in range(num_epochs):
        total_loss = 0
        total_accuracy = 0
        total_cosine_sim = 0
        total_alignment = 0
        total_uniformity = 0

        for batch in train_loader:
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

        avg_loss = total_loss / len(train_loader)
        avg_acc = total_accuracy / len(train_loader)
        avg_cosine_sim = total_cosine_sim / len(train_loader)
        avg_align = total_alignment / len(train_loader)
        avg_unif = total_uniformity / len(train_loader)

        losses.append(avg_loss.cpu())
        accuracies.append(avg_acc.cpu())
        cosine_sims.append(avg_cosine_sim.cpu())
        alignments.append(avg_align.cpu())
        uniformities.append(avg_unif.cpu())

        print(f"Epoch: [{epoch + 1}/{num_epochs}]")
        print(
            f"Loss: {avg_loss:.4f}, Contrastive Accuracy: {avg_acc:.4f}, "f"Cosine Similarity: {avg_cosine_sim:.4f}, Alignment: {avg_align:.4f}, Uniformity: {avg_unif:.4f}")

        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            pretrained_convnext_backbone = model.backbone
            state_dict = {"convnext_parameters": pretrained_convnext_backbone.state_dict()}
            torch.save(state_dict, "./covnextv2-simclr.pth")
            print(f'Best model saved with loss: {best_loss:.4f}')
            print(
                '--------------------------------------------------------------------------------------------------------------')
        else:
            print(
                '--------------------------------------------------------------------------------------------------------------')

    # load the model
    convnext_new = timm.create_model('convnextv2_base.fcmae_ft_in22k_in1k_384', pretrained=True)
    backbone_new = nn.Sequential(*list(convnext_new.children())[:-1])
    ckpt = torch.load("./covnextv2-simclr.pth")
    backbone_new.load_state_dict(ckpt["convnext_parameters"])

    model.eval()
    embeddings, filenames = generate_embeddings(model, test_loader)

    plot_knn_examples(embeddings, filenames, batch_size)
