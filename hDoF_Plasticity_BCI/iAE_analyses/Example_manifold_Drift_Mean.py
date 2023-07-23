

import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate swiss roll dataset
n_samples = 15000
noise = 0.2
X, _ = make_swiss_roll(n_samples=n_samples, noise=noise)

# Scale the data to the range [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Scenario 1: Mean is (0, 0, 0)
mean_scenario1 = np.mean(X_scaled, axis=0)
X_scenario1 = X_scaled - mean_scenario1

# Scenario 2: Mean is non-zero, for example, (-2, 3, 1)
mean_scenario2 = np.array([-2, 3, 1])
X_scenario2 = X_scenario1 - mean_scenario2
X_scenario2_centered = X_scenario2 - np.mean(X_scenario2,axis=0)

# Convert data to tensors and create dataloaders
tensor_X_scenario1 = torch.Tensor(X_scenario1)
tensor_X_scenario2 = torch.Tensor(X_scenario2_centered)

dataset_scenario1 = TensorDataset(tensor_X_scenario1, tensor_X_scenario1)
dataset_scenario2 = TensorDataset(tensor_X_scenario2, tensor_X_scenario2)

dataloader_scenario1 = DataLoader(dataset_scenario1, batch_size=32, shuffle=True)
dataloader_scenario2 = DataLoader(dataset_scenario2, batch_size=32, shuffle=True)

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),            
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Define the training function
def train_autoencoder(model, dataloader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for data, _ in dataloader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * data.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# Train the autoencoder on Scenario 1
input_dim = X_scenario1.shape[1]
encoding_dim = 2

autoencoder_scenario1 = Autoencoder(input_dim, encoding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder_scenario1.parameters(), lr=0.001)
num_epochs = 50

train_autoencoder(autoencoder_scenario1, dataloader_scenario1, criterion, optimizer, num_epochs)

# Train the autoencoder on Scenario 2
autoencoder_scenario2 = Autoencoder(input_dim, encoding_dim)
optimizer = optim.Adam(autoencoder_scenario2.parameters(), lr=0.001)

train_autoencoder(autoencoder_scenario2, dataloader_scenario2, criterion, optimizer, num_epochs)

# Reconstruct the data points for Scenario 1
reconstructed_scenario1 = autoencoder_scenario1(tensor_X_scenario1).detach().numpy()

# Reconstruct the data points for Scenario 2
reconstructed_scenario2 = autoencoder_scenario2(tensor_X_scenario2).detach().numpy()
reconstructed_scenario2 = reconstructed_scenario2 - mean_scenario2 # adding mean back

# Plotting the results
fig = plt.figure(figsize=(15, 4))

# Scenario 1
ax1 = fig.add_subplot(141, projection='3d')
ax1.scatter(X_scenario1[:, 0], X_scenario1[:, 1], X_scenario1[:, 2], c='blue', edgecolor='k')
ax1.set_title('Scenario 1: Mean (0, 0, 0)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.view_init(10, -70)

# Reconstructed Scenario 1
ax2 = fig.add_subplot(142, projection='3d')
ax2.scatter(reconstructed_scenario1[:, 0], reconstructed_scenario1[:, 1], reconstructed_scenario1[:, 2], c='red', edgecolor='k')
ax2.set_title('Reconstructed Scenario 1')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.view_init(10, -70)

# Scenario 2
ax3 = fig.add_subplot(143, projection='3d')
ax3.scatter(X_scenario2[:, 0], X_scenario2[:, 1], X_scenario2[:, 2], c='blue', edgecolor='k')
ax3.set_title('Scenario 2: Mean (-2, 3, 1)')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.view_init(10, -70)

# Reconstructed Scenario 2
ax3 = fig.add_subplot(144, projection='3d')
ax3.scatter(reconstructed_scenario2[:, 0], reconstructed_scenario2[:, 1], reconstructed_scenario2[:, 2], c='red', edgecolor='k')
ax3.set_title('Reconstructed Scenario 2: Mean (-2, 3, 1)')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.view_init(10, -70)

plt.tight_layout()
plt.show()
