import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO
import matplotlib.pyplot as plt

# Function to load data
def get_data_loaders(data_flag='bloodmnist', batch_size=32, download=True):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    transform = transforms.Compose([transforms.ToTensor()]) #No data augmentation

    train_data = DataClass(split='train', transform=transform, download=download)
    test_data = DataClass(split='test', transform=transform, download=download)
    val_data = DataClass(split='val', transform=transform, download=download)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, len(info['label'])

# CNN Model
class BloodMNISTCNN(nn.Module):
    def __init__(self, n_classes):
        super(BloodMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1) #16x28x28
        self.pool1 = nn.MaxPool2d(2, 2) #16x14x14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1) #32x14x14
        self.pool2 = nn.MaxPool2d(2, 2) #32x7x7
        self.fc1 = nn.Linear(32 * 7 * 7, 128) #128x7x7
        self.dropout = nn.Dropout(p=0.5) #50% dropout
        self.fc2 = nn.Linear(128, n_classes) #8 classes

    def forward(self, x): #Forwards passs
        x = F.relu(self.conv1(x)) 
        x = self.pool1(x) # Convolution + ReLU + Pooling

        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = x.view(-1, 32 * 7 * 7) #Flatten
        x = F.relu(self.fc1(x)) #Fully connected + ReLU

        x = self.dropout(x)
        x = self.fc2(x) #Output
        return x

# Training and Validation
def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer):
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    #Training

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            labels = labels.squeeze().long() # Adjust labels shape for BCEWithLogitsLoss
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1) #No sigmoid, chooses the class with the highest score
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_losses[-1]:.4f}, Accuracy: {train_accuracy:.4f}')

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0 
        val_total = 0

        with torch.no_grad(): #No weights are adjusted for validation
            for images, labels in val_loader:
                labels = labels.squeeze().long()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / val_total
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)

        print(f"Validation - Loss: {val_losses[-1]:.4f}, Accuracy: {val_accuracy:.4f}")

    return train_losses, val_losses, train_accuracies, val_accuracies

# Testing
def test_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0 
    correct = 0 
    total = 0

    with torch.no_grad(): #No weights are adjusted for testing
        for images, labels in test_loader:
            labels = labels.squeeze().long()
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f'Test Loss: {test_loss/len(test_loader):.4f}, Accuracy: {accuracy:.4f}')
    return test_loss, accuracy

# Plotting
def plot_performance(epochs, train_losses, val_losses, train_accuracies, val_accuracies):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(epochs, train_losses, label="Training Loss", marker='o')
    axes[0].plot(epochs, val_losses, label="Validation Loss", marker='o')
    axes[0].set_title("Loss vs Epochs")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_accuracies, label="Training Accuracy", marker='o')
    axes[1].plot(epochs, val_accuracies, label="Validation Accuracy", marker='o')
    axes[1].set_title("Accuracy vs Epochs")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    plt.tight_layout()
    plt.show()
