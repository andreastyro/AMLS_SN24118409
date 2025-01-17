import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO, Evaluator
from torchviz import make_dot
import matplotlib.pyplot as plt

# Set parameters
data_flag = 'breastmnist'
download = True
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.00006

# Dataset info
info = INFO[data_flag]
n_channels = info['n_channels']  # Number of input channels (grayscale = 1)
n_classes = len(info['label'])   # Number of classes (2 for BreastMNIST)

# Dataset class
DataClass = getattr(medmnist, info['python_class'])

# Transform
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor()
])

# Load data

train_data = DataClass(split='train', transform=transform, download=download)
test_data = DataClass(split='test', transform=transform, download=download)
val_data = DataClass(split='val', transform=transform, download=download)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

"""#Visualize Images
montage_image = train_data.montage(length=20)

# Display the montage using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(montage_image)
plt.axis("off")
plt.title("Montage of Training Data")
plt.show()"""

# Define the CNN model
class BreastMNISTCNN(nn.Module):
    def __init__(self):
        super(BreastMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(2, 2)  # 64x14x14

        self.fc1 = nn.Linear(128 * 14 * 14, 256)  # Adjust for flattening
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = x.view(x.size(0), -1)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # Output logits
        return x

# Initialize the model
model = BreastMNISTCNN()
print(model)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Combines sigmoid + binary cross-entropy
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

# Lists to store loss and accuracy
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        labels = labels.float()  # Adjust labels shape for BCEWithLogitsLoss
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Track loss and accuracy
        train_loss += loss.item()
        predictions = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_accuracy = correct / total
    train_losses.append(train_loss / len(train_loader))
    train_accuracies.append(train_accuracy)

    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {train_loss/len(train_loader):.4f}, Accuracy: {train_accuracy:.4f}')

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            labels = labels.float()  # Adjust labels shape for BCEWithLogitsLoss
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Track validation loss and accuracy
            val_loss += loss.item()
            predictions = (torch.sigmoid(outputs) >= 0.5).float()
            val_correct += (predictions == labels).sum().item()
            total_val += labels.size(0)

    val_accuracy = val_correct / total_val
    val_losses.append(val_loss / len(val_loader))
    val_accuracies.append(val_accuracy)
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] - Val Loss: {val_loss/len(val_loader):.4f}, Val Accuracy: {val_accuracy:.4f}")

# Test loop
model.eval()
test_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        labels = labels.float()  # Adjust labels shape for BCEWithLogitsLoss
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        predictions = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

test_accuracy = correct / total
print(f'Test Loss: {test_loss/len(test_loader):.4f}, Accuracy: {test_accuracy:.4f}')

# Plotting Performance
epochs = range(1, NUM_EPOCHS + 1)

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Adjusting the y-axis limits for Loss (first graph)
axes[0].set_ylim(0, 1)

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