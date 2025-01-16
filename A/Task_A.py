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
NUM_EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Dataset info
info = INFO[data_flag]
n_channels = info['n_channels']  # Number of input channels (grayscale = 1)
n_classes = len(info['label'])   # Number of classes (2 for BreastMNIST)

# Dataset class
DataClass = getattr(medmnist, info['python_class'])

# Transform
transform = transforms.Compose([transforms.ToTensor()])

# Load data
train_data = DataClass(split='train', transform=transform, download=download)
test_data = DataClass(split='test', transform=transform, download=download)
val_data = DataClass(split='val', transform=transform, download=download)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

#Visualize Images
montage_image = train_data.montage(length=20)

# Display the montage using matplotlib
plt.figure(figsize=(10, 10))
plt.imshow(montage_image)
plt.axis("off")
plt.title("Montage of Training Data")
plt.show()

# Define the CNN model
class BreastMNISTCNN(nn.Module):
    def __init__(self):
        super(BreastMNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Output: 32x28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Output: 64x28x28
        self.pool = nn.MaxPool2d(2, 2)  # Halves dimensions: 64x14x14
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, 1)  # Output layer (1 neuron for binary classification)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Convolution + ReLU
        x = self.pool(F.relu(self.conv2(x)))  # Convolution + ReLU + Pooling
        x = x.view(-1, 64 * 14 * 14)  # Flatten
        x = F.relu(self.fc1(x))  # Fully connected + ReLU\
        x = self.fc2(x)  # Output logits
        return x

# Initialize the model
model = BreastMNISTCNN()
print(model)