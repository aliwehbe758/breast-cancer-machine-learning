import sys
sys.path.append('../')
from utilities import *
import utilities

mean, std = get_mean_and_std(224)
print(f'Mean: {mean}')
print(f'Std Dev: {std}')

patch_size = 16         # Patch size (P) = 16
latent_size = 768       # Latent vector (D). ViT-Base uses 768
n_channels = 3          # Number of channels for input images
num_heads = 12          # ViT-Base uses 12 heads
num_encoders = 12       # ViT-Base uses 12 encoder layers
dropout = 0.1           # Dropout = 0.1 is used with ViT-Base & ImageNet-21k
num_classes = 3        # Number of classes in Busi dataset
size = 224              # Size used for training = 224

epochs = 2             # Number of epochs
base_lr = 1e-3         # Base LR
weight_decay = 0.03     # Weight decay for ViT-Base (on ImageNet-21k)
batch_size = 32

transform_training_data = Compose([
    RandomResizedCrop(size=224, scale=(0.8, 1.0)),
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    ToTensor(),
    Normalize(mean, std)
])

# Define the transformations for validation data (no augmentation, only necessary processing)
transform_validation_data = Compose([
    Resize(256),
    CenterCrop(224),
    ToTensor(),
    Normalize(mean, std)
])

train_loader, val_loader = get_train_and_validation_loader(transform_training_data, transform_validation_data, 32)

# Define model components and parameters
class InputEmbedding(nn.Module):
    def __init__(self, patch_size=16, n_channels=3, latent_size=768, device=device):
        super(InputEmbedding, self).__init__()
        self.latent_size = latent_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.device = device
        self.input_size = self.patch_size * self.patch_size * self.n_channels
        self.linearProjection = nn.Linear(self.input_size, self.latent_size)

    def forward(self, input_data):
        batch_size = input_data.size(0)
        input_data = input_data.to(self.device)
        patches = einops.rearrange(input_data, 'b c (h h1) (w w1) -> b (h w) (h1 w1 c)', h1=self.patch_size, w1=self.patch_size)
        linear_projection = self.linearProjection(patches).to(self.device)
        class_token = nn.Parameter(torch.randn(batch_size, 1, self.latent_size)).to(self.device)
        pos_embedding = nn.Parameter(torch.randn(batch_size, 1, self.latent_size)).to(self.device)
        linear_projection = torch.cat((class_token, linear_projection), dim=1)
        pos_embed = einops.repeat(pos_embedding, 'b 1 d -> b m d', m=linear_projection.size(1))
        linear_projection += pos_embed
        return linear_projection

class EncoderBlock(nn.Module):
    def __init__(self, latent_size=latent_size, num_heads=num_heads, device=device, dropout=dropout):
        super(EncoderBlock, self).__init__()

        self.latent_size = latent_size
        self.num_heads = num_heads
        self.device = device
        self.dropout = dropout

        # Normalization layer for both sublayers
        self.norm = nn.LayerNorm(self.latent_size)

        # Multi-Head Attention layer
        self.multihead = nn.MultiheadAttention(
            self.latent_size, self.num_heads, dropout=self.dropout)

        # MLP_head layer in the encoder. I use the same configuration as that
        # used in the original VitTransformer implementation. The ViT-Base
        # variant uses MLP_head size 3072, which is latent_size*4.
        self.enc_MLP = nn.Sequential(
            nn.Linear(self.latent_size, self.latent_size*4),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.latent_size*4, self.latent_size),
            nn.Dropout(self.dropout)
        )

    def forward(self, embedded_patches):

        # First sublayer: Norm + Multi-Head Attention + residual connection.
        # We take the first element ([0]) of the returned output from nn.MultiheadAttention()
        # because this module returns 'Tuple[attention_output, attention_output_weights]'.
        firstNorm_out = self.norm(embedded_patches)
        attention_output = self.multihead(firstNorm_out, firstNorm_out, firstNorm_out)[0]

        # First residual connection
        first_added_output = attention_output + embedded_patches

        # Second sublayer: Norm + enc_MLP (Feed forward)
        secondNorm_out = self.norm(first_added_output)
        ff_output = self.enc_MLP(secondNorm_out)

        # Return the output of the second residual connection
        return ff_output + first_added_output

class VitTransformer(nn.Module):
    def __init__(self, num_encoders=num_encoders, latent_size=latent_size, device=device, num_classes=num_classes, dropout=dropout):
        super(VitTransformer, self).__init__()
        self.num_encoders = num_encoders
        self.latent_size = latent_size
        self.device = device
        self.num_classes = num_classes
        self.dropout = dropout

        self.embedding = InputEmbedding()

        # Create a stack of encoder layers
        self.encStack = nn.ModuleList([EncoderBlock() for i in range(self.num_encoders)])

        # MLP_head at the classification stage has 'one hidden layer at pre-training time
        # and by a single linear layer at fine-tuning time'. For this implementation I will
        # use what was used for training, so I'll have a total of two layers, one hidden
        # layer and one output layer.
        self.MLP_head = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.Linear(self.latent_size, self.latent_size),
            nn.Linear(self.latent_size, self.num_classes)
        )

    def forward(self, test_input):

        # Apply input embedding (patchify + linear projection + position embeding)
        # to the input image passed to the model
        enc_output = self.embedding(test_input)

        # Loop through all the encoder layers
        for enc_layer in self.encStack:
            enc_output = enc_layer(enc_output)

        # Extract the output embedding information of the [class] token
        cls_token_embedding = enc_output[:, 0]

        # Finally, return the classification vector for all image in the batch
        return self.MLP_head(cls_token_embedding)

model = VitTransformer(num_encoders, latent_size, device, num_classes).to(device)

# Betas used for Adam in paper are 0.9 and 0.999, which are the default in PyTorch
optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores, train_recalls, val_recalls, train_precisions, val_precisions, train_conf_matrices, val_conf_matrices, all_train_labels, all_train_probs, all_val_labels, all_val_probs = vit_train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device)

save_model(model, './models/vit-impl-1.pth')

model = VitTransformer(num_encoders, latent_size, device, num_classes).to(device)
load_model(model, './models/vit-impl-1.pth')

vit_test(model, val_loader, train_loader.dataset.dataset.classes, mean, std)