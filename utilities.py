from libraries import *
import libraries

dataset_root_path = '../Dataset_BUSI_with_GT/'

# Function to calculate mean and std
def get_mean_and_std(image_size):
    # Define the transformation that only converts images to tensor
    transform = Compose([
        Resize((image_size, image_size)),  # Resize all images to (image_size x image_size)
        ToTensor()
    ])

    dataset = ImageFolder(root=dataset_root_path, transform=transform)
    loader = DataLoader(dataset, batch_size=50, num_workers=4, shuffle=False)

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std

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


# Training function
def vit_train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1_scores, val_f1_scores = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []
    train_conf_matrices, val_conf_matrices = [], []
    all_train_probs, all_val_probs = [], []
    all_train_labels, all_val_labels = [], []

    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        all_probs = []

        for images, labels in tqdm(train_loader, desc="Training Batch", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_probs.extend(outputs.detach().cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=1)
        train_f1_scores.append(train_f1)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)

        train_conf_matrix = confusion_matrix(all_labels, all_preds)
        train_conf_matrices.append(train_conf_matrix)

        all_train_probs.extend(all_probs)
        all_train_labels.extend(all_labels)

        # Validation
        model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        all_probs = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation Batch", leave=False):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

                all_probs.extend(outputs.detach().cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = running_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=1)
        val_f1_scores.append(val_f1)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)

        val_conf_matrix = confusion_matrix(all_labels, all_preds)
        val_conf_matrices.append(val_conf_matrix)

        all_val_probs.extend(all_probs)
        all_val_labels.extend(all_labels)

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Training: Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | F1: {train_f1:.2f} | Precision: {train_precision:.2f} | Recall: {train_recall:.2f}')
        print(f'Validation: Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | F1: {val_f1:.2f} | Precision: {val_precision:.2f} | Recall: {val_recall:.2f}')
        print('------------------------------------------------------------------------------------------------------------------')

        scheduler.step()

    return train_losses, val_losses, train_accuracies, val_accuracies, train_f1_scores, val_f1_scores, train_recalls, val_recalls, train_precisions, val_precisions, train_conf_matrices, val_conf_matrices, all_train_labels, all_train_probs, all_val_labels, all_val_probs

# Plotting function
def vit_plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies,
                 train_f1_scores, val_f1_scores, train_recalls, val_recalls,
                 train_precisions, val_precisions, train_conf_matrices, val_conf_matrices,
                 all_train_labels, all_train_probs, all_val_labels, all_val_probs, classes):
    
    epochs = range(1, len(train_losses) + 1)
    
    # Setting up a larger figure to accommodate all subplots
    plt.figure(figsize=(18, 18))  # Adjust size to fit all subplots comfortably
    
    # Plot training and validation losses
    plt.subplot(3, 3, 1)  # 3 rows, 3 columns, position 1
    plt.plot(epochs, train_losses, 'r-', label='Train Loss')
    plt.plot(epochs, val_losses, 'b-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation accuracies
    plt.subplot(3, 3, 2)
    plt.plot(epochs, train_accuracies, 'r-', label='Train Accuracy')
    plt.plot(epochs, val_accuracies, 'b-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training and validation F1 scores
    plt.subplot(3, 3, 4)
    plt.plot(epochs, train_f1_scores, 'r-', label='Train F1 Score')
    plt.plot(epochs, val_f1_scores, 'b-', label='Validation F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()

    # Plot training and validation recalls
    plt.subplot(3, 3, 5)
    plt.plot(epochs, train_recalls, 'r-', label='Train Recall')
    plt.plot(epochs, val_recalls, 'b-', label='Validation Recall')
    plt.title('Training and Validation Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.legend()

    # Plot training and validation precisions
    plt.subplot(3, 3, 6)
    plt.plot(epochs, train_precisions, 'r-', label='Train Precision')
    plt.plot(epochs, val_precisions, 'b-', label='Validation Precision')
    plt.title('Training and Validation Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.legend()

    # Plot confusion matrices using Seaborn
    plt.subplot(3, 3, 7)  # Adjust size to fit the matrix clearly
    sns.heatmap(train_conf_matrices[-1], annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Training Confusion Matrix - Last Epoch')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.subplot(3, 3, 8)  # Another plot for validation confusion matrix
    sns.heatmap(val_conf_matrices[-1], annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Validation Confusion Matrix - Last Epoch')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Binarize the labels for ROC curve calculation
    all_train_labels_bin = label_binarize(all_train_labels, classes=range(len(classes)))
    all_val_labels_bin = label_binarize(all_val_labels, classes=range(len(classes)))

    plt.subplot(3, 3, 9)  # Position for the ROC curve
    for i, class_name in enumerate(classes):
        fpr_train, tpr_train, _ = roc_curve(all_train_labels_bin[:, i], np.array(all_train_probs)[:, i])
        roc_auc_train = auc(fpr_train, tpr_train)

        fpr_val, tpr_val, _ = roc_curve(all_val_labels_bin[:, i], np.array(all_val_probs)[:, i])
        roc_auc_val = auc(fpr_val, tpr_val)

        plt.plot(fpr_train, tpr_train, label=f'Train {class_name} (area = {roc_auc_train:.2f})')
        plt.plot(fpr_val, tpr_val, label=f'Val {class_name} (area = {roc_auc_val:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()


# Save the model
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}.")


# Load the model
def load_model(model, path):
    # model = VitTransformer(num_encoders, latent_size, device, num_classes).to(device)
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}.")
    model.eval()

def imshow(img, title, mean, std):
    img = img.clone()  # Clone the tensor to avoid modifying the original image
    for i in range(img.shape[0]):  # Unnormalize each channel
        img[i] = img[i] * std[i] + mean[i]
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

def vit_test(model, val_loader, classes, mean, std):
  dataiter = iter(val_loader)
  images, labels = next(dataiter)
  images, labels = images.to(device), labels.to(device)

  # Print the actual labels
  print('Actual labels: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

  # Predict the labels
  outputs = model(images)
  _, predicted = torch.max(outputs, 1)

  # Print the predicted labels
  print('Predicted labels: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

  # Show images with predicted labels
  imshow(torchvision.utils.make_grid(images.cpu()), 'Predicted: ' + ' '.join('%5s' % classes[predicted[j]] for j in range(4)), mean, std)


def mae_train(model, train_loader, val_loader, epochs, criterion, optimizer):

    model.to(device)

    # Training and validation metrics
    train_losses = []
    val_losses = []
    train_mse_scores = []
    val_mse_scores = []
    train_mae_scores = []
    val_mae_scores = []
    train_r2_scores = []
    val_r2_scores = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        epoch_train_labels = []
        epoch_train_outputs = []

        for images, _ in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            outputs, mask, _ = model(images)

            # Broadcast the mask to match the number of channels
            bool_mask = mask.repeat(1, images.size(1), 1, 1).bool()

            # Calculate loss only for the masked parts
            loss = criterion(outputs[bool_mask], images[bool_mask])
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Collect labels and outputs for metrics
            with torch.no_grad():
                labels = images[bool_mask].cpu().numpy().flatten()
                outputs_np = outputs[bool_mask].cpu().numpy().flatten()
                epoch_train_labels.extend(labels)
                epoch_train_outputs.extend(outputs_np)

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        # Calculate regression metrics for training data
        train_mse = mean_squared_error(epoch_train_labels, epoch_train_outputs)
        train_mae = mean_absolute_error(epoch_train_labels, epoch_train_outputs)
        train_r2 = r2_score(epoch_train_labels, epoch_train_outputs)

        train_mse_scores.append(train_mse)
        train_mae_scores.append(train_mae)
        train_r2_scores.append(train_r2)

        model.eval()
        running_loss = 0.0

        epoch_val_labels = []
        epoch_val_outputs = []

        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                outputs, mask, _ = model(images)

                # Broadcast the mask to match the number of channels
                bool_mask = mask.repeat(1, images.size(1), 1, 1).bool()

                # Calculate loss only for the masked parts
                loss = criterion(outputs[bool_mask], images[bool_mask])
                running_loss += loss.item()

                # Collect labels and outputs for metrics
                labels = images[bool_mask].cpu().numpy().flatten()
                outputs_np = outputs[bool_mask].cpu().numpy().flatten()
                epoch_val_labels.extend(labels)
                epoch_val_outputs.extend(outputs_np)

        val_loss = running_loss / len(val_loader)
        val_losses.append(val_loss)

        # Calculate regression metrics for validation data
        val_mse = mean_squared_error(epoch_val_labels, epoch_val_outputs)
        val_mae = mean_absolute_error(epoch_val_labels, epoch_val_outputs)
        val_r2 = r2_score(epoch_val_labels, epoch_val_outputs)

        val_mse_scores.append(val_mse)
        val_mae_scores.append(val_mae)
        val_r2_scores.append(val_r2)

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Training: Loss: {train_loss:.4f} | MSE: {train_mse:.2f} | MAE: {train_mae:.2f} | R2: {train_r2:.2f}')
        print(f'Validation: Loss: {val_loss:.4f} | MSE: {val_mse:.2f} | MAE: {val_mae:.2f} | R2: {val_r2:.2f}')
        print('------------------------------------------------------------------------------------------------------------------')

    return (train_losses, val_losses, train_mse_scores, val_mse_scores, train_mae_scores, val_mae_scores, train_r2_scores, val_r2_scores)


def mae_plot_metrics(train_losses, val_losses, train_mse_scores, val_mse_scores, train_mae_scores, val_mae_scores,
                     train_r2_scores, val_r2_scores):
    epochs = range(1, len(train_losses) + 1)

    # Setting up a larger figure to accommodate all subplots
    plt.figure(figsize=(18, 12))  # Adjust size to fit all subplots comfortably

    # Plot training and validation losses
    plt.subplot(3, 3, 1)  # 3 rows, 3 columns, position 1
    plt.plot(epochs, train_losses, 'r-', label='Train Loss')
    plt.plot(epochs, val_losses, 'b-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation MSE Score
    plt.subplot(3, 3, 2)
    plt.plot(epochs, train_mse_scores, 'r-', label='Train MSE Score')
    plt.plot(epochs, val_mse_scores, 'b-', label='Validation MSE Score')
    plt.title('Training and Validation MSE Score')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()

    # Plot training and validation MAE Score
    plt.subplot(3, 3, 4)
    plt.plot(epochs, train_mae_scores, 'r-', label='Train MAE Score')
    plt.plot(epochs, val_mae_scores, 'b-', label='Validation MAE Score')
    plt.title('Training and Validation MAE Score')
    plt.xlabel('Epochs')
    plt.ylabel('MAE Score')
    plt.legend()

    # Plot training and validation R2 Score
    plt.subplot(3, 3, 5)
    plt.plot(epochs, train_r2_scores, 'r-', label='Train R2 Score')
    plt.plot(epochs, val_r2_scores, 'b-', label='Validation R2 Score')
    plt.title('Training and Validation R2 Score')
    plt.xlabel('Epochs')
    plt.ylabel('R2 Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

def mae_plot_masking_details(model, val_loader, num_images):
  model.to(device)
  for i in range(num_images):
    dataiter = iter(val_loader)
    images, _ = next(dataiter)
    images = images.to(device)
    image = images[i].unsqueeze(0)  # Get the first image and add a batch dimension

    # Pass the image through the autoencoder
    with torch.no_grad():
        reconstructed, mask, input_masked = model(image)

    # Compute absolute difference and sum
    absolute_difference = torch.abs(image - reconstructed)
    total_difference = torch.sum(absolute_difference).item()
    print(f"Total sum of absolute differences: {total_difference}")
    fig, axs = plt.subplots(1, 4, figsize=(18, 5))

    # Convert tensors to numpy for plotting
    original_np =image[0].cpu().permute(1, 2, 0).numpy()
    mask_np = mask[0].cpu().squeeze().numpy()
    input_masked_np = input_masked[0].cpu().permute(1, 2, 0).numpy()
    reconstructed_np = reconstructed[0].cpu().permute(1, 2, 0).numpy()

    axs[0].imshow(original_np)
    axs[0].set_title("Original Image")
    axs[0].axis('off')


    axs[1].imshow(mask_np, cmap='gray')
    axs[1].set_title("Mask")
    axs[1].axis('off')

    axs[2].imshow(input_masked_np)
    axs[2].set_title("Input with Mask")
    axs[2].axis('off')

    axs[3].imshow(reconstructed_np)
    axs[3].set_title("Reconstructed Image")
    axs[3].axis('off')

    plt.show()

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

        return self.normalize(image)

    def normalize(self, image):
        mean = image.mean()
        std = image.std()
        normalized_image = (image - mean) / std
        return normalized_image