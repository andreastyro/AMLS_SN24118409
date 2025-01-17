import sys
from pathlib import Path
import torch
import torch.nn as nn

# Add folder A to the Python path
sys.path.append(str(Path(__file__).parent / "A"))

# Import functions and class from BreastMNIST_CNN
from BreastMNIST_CNN import (
    get_data_loaders,
    BreastMNISTCNN,
    train_model,
    test_model,
    plot_performance,
)

# Set parameters
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.00006

# Load data
train_loader, val_loader, test_loader = get_data_loaders(batch_size=BATCH_SIZE)

# Initialize the model
model = BreastMNISTCNN()
print(model)

# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss() #Used for BINARY classification
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)

# Train the BreastMNIST model
train_losses, train_accuracies, val_losses, val_accuracies = train_model(
    model, train_loader, val_loader, NUM_EPOCHS, criterion, optimizer
)

# Plot the performance
plot_performance(
    range(1, NUM_EPOCHS + 1), train_losses, val_losses, train_accuracies, val_accuracies
)

# Testing the model
test_loss, test_accuracy = test_model(model, test_loader, criterion)

#Add folder B to the Python path
sys.path.append(str(Path(__file__).parent / "B"))

from BloodMNIST_CNN import (
    get_data_loaders,
    BloodMNISTCNN,
    train_model,
    test_model,
    plot_performance,
)

# Parameters
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# Load data
train_loader, val_loader, test_loader, n_classes = get_data_loaders(batch_size=BATCH_SIZE)

# Initialize model
model = BloodMNISTCNN(n_classes)
criterion = nn.CrossEntropyLoss() #Used for MULTI-CLASS clasification
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train BloodMNIST model
train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, val_loader, NUM_EPOCHS, criterion, optimizer
)

# Plot performance
plot_performance(
    range(1, NUM_EPOCHS + 1), train_losses, val_losses, train_accuracies, val_accuracies
)

# Test model
test_model(model, test_loader, criterion)