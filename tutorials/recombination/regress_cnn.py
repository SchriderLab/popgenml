# -*- coding: utf-8 -*-
from popgenml.data.simulators import MSPrimeSimulator
from popgenml.data.transforms import FastSeriate, PadCrop, Flip, Compose
from popgenml.data.datasets import LiveSimulationDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import torch
import numpy as np

from popgenml.models.torchvision import resnet18

# Use the included config (.ini) file which defines our prior
simulator = MSPrimeSimulator('recom.ini')
# specify a formatting pipeline for the binary popgen alignment

pipeline = Compose([Flip(), FastSeriate(), PadCrop(128)])

# let's compute the mean and std of our target, log-recombination rate
samples = simulator.r.rvs(size=10000)

log_samples = np.log(samples)

mu_y = np.mean(log_samples)
std_y = np.std(log_samples)

# we'll attempt to predict the log-recombination rate which is returned by our simulator
# as the 'r' entry in the dictionary
def parse_fn(result):
    x = result['x'] # numpy array
    pos = result['pos']
    r = result['r']
    
    r = (np.log(r) - mu_y) / std_y

    x, pos = pipeline(x, pos)

    # make into torch Tensors, expand a channel dimension s.t. the returned shape is (1, 50, 128)
    return torch.FloatTensor(x).unsqueeze(0), torch.FloatTensor(np.array([r]))
    
dataset = LiveSimulationDataset(simulator, parse_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using " + str(device) + " as device")

model = resnet18(in_channels = 1, num_classes = 1).to(device)

# ---------------------------------------------------------
# 1. Create the DataLoader
# ---------------------------------------------------------
batch_size = 32
# Adjust num_workers if your simulator supports multiprocessing
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0)

# ---------------------------------------------------------
# 2. Define Loss function and Optimizer
# ---------------------------------------------------------
# We are predicting a continuous value, so Mean Squared Error is appropriate
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 5
batches_per_epoch = 100  # Cap the number of batches since Live simulations can be infinite

train_losses = []

# ---------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------
print("\nStarting Training...")
model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    
    # Create an iterator from the DataLoader 
    data_iter = iter(dataloader)
    
    for batch_idx in range(batches_per_epoch):
        try:
            inputs, targets = next(data_iter)
        except StopIteration:
            # If the dataset is finite, it might run out before batches_per_epoch
            break
            
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, targets)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / batches_per_epoch
    train_losses.append(avg_loss)
    print(f"Epoch [{epoch+1}/{epochs}], Average Loss (MSE): {avg_loss:.4f}")

# ---------------------------------------------------------
# 4. Display the Results with Matplotlib
# ---------------------------------------------------------
# Plot 1: Training Loss Curve
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_losses, marker='o', linestyle='-', color='b')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.grid(True)

# Plot 2: Predicted vs. Actual on a validation batch
model.eval()
with torch.no_grad():
    # Grab one batch to visualize performance
    sample_inputs, sample_targets = next(iter(dataloader))
    sample_inputs = sample_inputs.to(device)
    
    preds = model(sample_inputs).cpu().numpy()
    actuals = sample_targets.numpy()
    
    # Inverse transform to get back to original log space distribution
    preds_scaled = (preds * std_y) + mu_y
    actuals_scaled = (actuals * std_y) + mu_y

    plt.subplot(1, 2, 2)
    plt.scatter(actuals_scaled, preds_scaled, alpha=0.6, color='r')
    
    # Plot an ideal 1:1 reference line
    min_val = min(actuals_scaled.min(), preds_scaled.min())
    max_val = max(actuals_scaled.max(), preds_scaled.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal')
    
    plt.title('Predicted vs. True log(r)')
    plt.xlabel('True log(r)')
    plt.ylabel('Predicted log(r)')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

