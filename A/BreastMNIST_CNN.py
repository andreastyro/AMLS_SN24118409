import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO
import matplotlib.pyplot as plt

# Function to get data loaders
def get_data_loaders(data_flag='breastmnist', batch_size=32, download=True):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    transform = transforms.Compose([ #Data augmenetation
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor()
    ])

    train_data = DataClass(split='train', transform=transform, download=download)
    val_data = DataClass(split='val', transform=transform, download=download)
    test_data = DataClass(split='test', transform=transform, download=download)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# CNN model class
class BreastMNISTCNN(nn.Module):
    def __init__(self):
        super(BreastMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1) #64x28x28
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) #128x28x28
        self.bn2 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2, 2)  # 128x14x14

        self.fc1 = nn.Linear(128 * 14 * 14, 256) #256x14x14
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 1) #Binary Classification

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))) #Convolution + Batch normalization + ReLU
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x) # Pooling

        x = x.view(x.size(0), -1) #Flatten

        x = F.relu(self.fc1(x)) # Fully Connected + ReLU
        x = self.dropout(x)
        x = self.fc2(x) #output
        return x

# Function for training and validation of the model
def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    #Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            labels = labels.float() #Adjust labels shape for BCEWithLogitsLoss
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Loss and accuracy calculations
            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.4f}')

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        total_val = 0

        with torch.no_grad(): #No weight adjustments for validation
            for images, labels in val_loader:
                labels = labels.float()
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                predictions = (torch.sigmoid(outputs) >= 0.5).float()
                val_correct += (predictions == labels).sum().item()
                total_val += labels.size(0)

        val_accuracy = val_correct / total_val
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}] - Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}")

    return train_losses, train_accuracies, val_losses, val_accuracies

# Function to test the model
def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad(): #No weight adjustments for testing
        for images, labels in test_loader:
            labels = labels.float()
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    test_accuracy = correct / total
    print(f'Test Loss: {test_loss/len(test_loader):.4f}, Accuracy: {test_accuracy:.4f}')
    return test_loss, test_accuracy

# Function to plot training and validation performance
def plot_performance(epochs, train_losses, val_losses, train_accuracies, val_accuracies):
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Adjusting the y-axis limits for Loss (first graph)
    axes[0].set_ylim(0, max(max(train_losses), max(val_losses)) * 1.1)

    # Adjusting the y-axis limits for Accuracy (second graph)
    axes[1].set_ylim(0.5, 1)

    # Plot Loss
    axes[0].plot(epochs, train_losses, label="Training Loss", marker='o')
    axes[0].plot(epochs, val_losses, label="Validation Loss", marker='o')
    axes[0].set_title("Loss vs Epochs")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    # Plot Accuracy
    axes[1].plot(epochs, train_accuracies, label="Training Accuracy", marker='o')
    axes[1].plot(epochs, val_accuracies, label="Validation Accuracy", marker='o')
    axes[1].set_title("Accuracy vs Epochs")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    # Adjust spacing between plots
    plt.tight_layout()
    plt.show()
