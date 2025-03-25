import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
import seaborn as sns

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Hyperparameters
batch_size = 64
num_epochs = 10
learning_rate = 0.001

# Define transforms
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load datasets
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

# Define classes
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


# Function to visualize dataset
def visualize_dataset():
    # Get random samples
    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # Show images
    plt.figure(figsize=(10, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        # Denormalize images
        img = images[i] / 2 + 0.5
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.title(classes[labels[i]])
        plt.axis('off')
    plt.tight_layout()
    plt.suptitle("CIFAR-10 Sample Images", fontsize=16)
    plt.subplots_adjust(top=0.85)
    plt.show()


# Define CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        # Conv Layer 1: Input = 32x32x3, Output = 32x32x32
        # Kernel Size = 3x3, Stride = 1, Padding = 1 (same padding)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization
        self.relu1 = nn.ReLU()

        # Conv Layer 2: Input = 32x32x32, Output = 32x32x64
        # Kernel Size = 3x3, Stride = 1, Padding = 1 (same padding)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # Batch normalization
        self.relu2 = nn.ReLU()

        # Max Pooling: Input = 32x32x64, Output = 16x16x64
        # Kernel Size = 2x2, Stride = 2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout layer with 25% dropout rate
        self.dropout1 = nn.Dropout(0.25)

        # Conv Layer 3: Input = 16x16x64, Output = 16x16x128
        # Kernel Size = 3x3, Stride = 1, Padding = 1 (same padding)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # Batch normalization
        self.relu3 = nn.ReLU()

        # Conv Layer 4: Input = 16x16x128, Output = 16x16x128
        # Kernel Size = 3x3, Stride = 1, Padding = 1 (same padding)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)  # Batch normalization
        self.relu4 = nn.ReLU()

        # Max Pooling: Input = 16x16x128, Output = 8x8x128
        # Kernel Size = 2x2, Stride = 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout layer with 25% dropout rate
        self.dropout2 = nn.Dropout(0.25)

        # Fully Connected Layer 1: Input = 8*8*128 = 8192, Output = 512
        self.fc1 = nn.Linear(8 * 8 * 128, 512)
        self.bn5 = nn.BatchNorm1d(512)  # Batch normalization
        self.relu5 = nn.ReLU()

        # Dropout layer with 50% dropout rate
        self.dropout3 = nn.Dropout(0.5)
        

        # Fully Connected Layer 2: Input = 512, Output = 10 (number of classes)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Convolutional layers
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Flatten
        x = x.view(-1, 8 * 8 * 128)

        # Fully connected layers
        x = self.relu5(self.bn5(self.fc1(x)))
        x = self.dropout3(x)
        x = self.fc2(x)

        return x


# Function to train the model
def train_model(model, device, trainloader, testloader, criterion, optimizer, num_epochs):
    model.to(device)
    best_accuracy = 0.0

    # Lists to store metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    start_time = time.time()

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Calculate statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Print statistics every 100 mini-batches
            if (i + 1) % 100 == 0:
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}, '
                      f'Accuracy: {100 * correct / total:.2f}%')
                running_loss = 0.0

        # Calculate training accuracy for the epoch
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {running_loss:.3f}, Train Acc: {train_accuracy:.2f}%, '
              f'Val Loss: {val_loss:.3f}, Val Acc: {val_accuracy:.2f}%')

        val_accuracies.append(val_accuracy)

        # Save the best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')

    training_time = time.time() - start_time
    print(f'Training completed in {training_time:.2f} seconds')

    return model, training_time, train_accuracies, val_accuracies


# Function to evaluate the model
def evaluate_model(model, device, testloader):
    model.eval()

    # Lists to store predictions and true labels
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds) * 100
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Generate classification report
    report = classification_report(all_labels, all_preds, target_names=classes)

    return accuracy, precision, recall, f1, cm, report


# Function to visualize CNN kernels/filters
def visualize_kernels(model):
    # Get the weights from the first conv layer
    kernels = model.conv1.weight.detach().cpu().numpy()

    # Normalize for better visualization
    kernels = (kernels - kernels.min()) / (kernels.max() - kernels.min())

    # Plot the kernels
    fig, axes = plt.subplots(4, 8, figsize=(15, 8))
    fig.suptitle("First Convolutional Layer Filters", fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < kernels.shape[0]:  # We have 32 kernels in the first layer
            # Each kernel has 3 channels (RGB)
            kernel = np.transpose(kernels[i], (1, 2, 0))
            ax.imshow(kernel)
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


# Function to visualize feature maps
def visualize_feature_maps(model, device, testloader):
    # Get a sample image
    dataiter = iter(testloader)
    images, _ = next(dataiter)
    image = images[0:1].to(device)  # Use the first image

    # Create a model that outputs the feature maps after the first conv layer
    class FeatureExtractor(nn.Module):
        def __init__(self, model):
            super(FeatureExtractor, self).__init__()
            self.features = nn.Sequential(
                model.conv1,
                model.bn1,
                model.relu1
            )

        def forward(self, x):
            return self.features(x)

    feature_extractor = FeatureExtractor(model)
    feature_extractor.to(device)
    feature_extractor.eval()

    # Get the feature maps
    with torch.no_grad():
        feature_maps = feature_extractor(image)

    # Convert to numpy for visualization
    feature_maps = feature_maps.cpu().numpy()[0]

    # Plot the feature maps
    fig, axes = plt.subplots(4, 8, figsize=(15, 8))
    fig.suptitle("Feature Maps after First Convolutional Layer", fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < feature_maps.shape[0]:  # We have 32 feature maps
            ax.imshow(feature_maps[i], cmap='viridis')
            ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()


# Function to plot training history
def plot_training_history(train_accuracies, val_accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, len(val_accuracies) + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()


# Main execution flow
if __name__ == "__main__":
    # Visualize dataset
    print("Visualizing CIFAR-10 dataset samples:")
    visualize_dataset()

    # Create model, loss function, and optimizer
    model = CNNModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Print model architecture
    print("\nModel Architecture:")
    print(model)

    # Train on CPU first
    print("\n\nTraining model on CPU...")
    device_cpu = torch.device("cpu")
    model_cpu = CNNModel()
    optimizer_cpu = optim.Adam(model_cpu.parameters(), lr=learning_rate)
    model_cpu, training_time_cpu, train_acc_cpu, val_acc_cpu = train_model(
        model_cpu, device_cpu, trainloader, testloader, criterion, optimizer_cpu, num_epochs)

    # Evaluate on CPU
    print("\nEvaluating model on CPU...")
    accuracy_cpu, precision_cpu, recall_cpu, f1_cpu, cm_cpu, report_cpu = evaluate_model(
        model_cpu, device_cpu, testloader)

    print(f"\nCPU Results:")
    print(f"Training Time: {training_time_cpu:.2f} seconds")
    print(f"Test Accuracy: {accuracy_cpu:.2f}%")
    print(f"Precision: {precision_cpu:.4f}")
    print(f"Recall: {recall_cpu:.4f}")
    print(f"F1 Score: {f1_cpu:.4f}")
    print("\nClassification Report:")
    print(report_cpu)

    # Plot training history for CPU
    plot_training_history(train_acc_cpu, val_acc_cpu)

    # Plot confusion matrix for CPU
    plot_confusion_matrix(cm_cpu, classes)

    # Visualize kernels for CPU model
    visualize_kernels(model_cpu)

    # Visualize feature maps for CPU model
    visualize_feature_maps(model_cpu, device_cpu, testloader)

    # If GPU is available, train on GPU
    if torch.cuda.is_available():
        print("\n\nTraining model on GPU...")
        device_gpu = torch.device("cuda:0")
        model_gpu = CNNModel()
        optimizer_gpu = optim.Adam(model_gpu.parameters(), lr=learning_rate)
        model_gpu, training_time_gpu, train_acc_gpu, val_acc_gpu = train_model(
            model_gpu, device_gpu, trainloader, testloader, criterion, optimizer_gpu, num_epochs)

        # Evaluate on GPU
        print("\nEvaluating model on GPU...")
        accuracy_gpu, precision_gpu, recall_gpu, f1_gpu, cm_gpu, report_gpu = evaluate_model(
            model_gpu, device_gpu, testloader)

        print(f"\nGPU Results:")
        print(f"Training Time: {training_time_gpu:.2f} seconds")
        print(f"Test Accuracy: {accuracy_gpu:.2f}%")
        print(f"Precision: {precision_gpu:.4f}")
        print(f"Recall: {recall_gpu:.4f}")
        print(f"F1 Score: {f1_gpu:.4f}")
        print("\nClassification Report:")
        print(report_gpu)

        # Plot training history for GPU
        plot_training_history(train_acc_gpu, val_acc_gpu)

        # Plot confusion matrix for GPU
        plot_confusion_matrix(cm_gpu, classes)

        # Visualize kernels for GPU model
        visualize_kernels(model_gpu)

        # Visualize feature maps for GPU model
        visualize_feature_maps(model_gpu, device_gpu, testloader)

        # Compare CPU vs GPU
        speedup = training_time_cpu / training_time_gpu if training_time_gpu > 0 else float('inf')
        print(f"\nCPU vs GPU Comparison:")
        print(f"Training Time: CPU = {training_time_cpu:.2f}s, GPU = {training_time_gpu:.2f}s")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Accuracy: CPU = {accuracy_cpu:.2f}%, GPU = {accuracy_gpu:.2f}%")
    else:
        print("\nGPU is not available. Skipping GPU training.")