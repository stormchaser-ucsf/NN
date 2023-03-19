# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 12:02:10 2023

@author: nikic
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Define the spatiotemporal transformer model
class SpatiotemporalTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, dropout):
        super(SpatiotemporalTransformer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Temporal Encoder
        self.temporal_encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4)
        self.temporal_encoder = nn.TransformerEncoder(self.temporal_encoder_layer, num_layers=num_layers)

        # Spatial Encoder
        self.spatial_encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4)
        self.spatial_encoder = nn.TransformerEncoder(self.spatial_encoder_layer, num_layers=num_layers)

        # Linear Layers
        self.linear1 = nn.Linear(input_dim * 2, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim, num_channels)

        # Reshape for temporal transformer
        x = x.permute(1, 0, 2, 3)  # shape: (seq_len, batch_size, input_dim, num_channels)
        x = x.reshape(x.size(0), x.size(1), -1)  # shape: (seq_len, batch_size, input_dim * num_channels)

        # Temporal Encoder
        x = self.temporal_encoder(x)

        # Reshape for spatial transformer
        x = x.permute(1, 2, 0)  # shape: (batch_size, input_dim * num_channels, seq_len)

        # Spatial Encoder
        x = self.spatial_encoder(x)

        # Reshape for linear layers
        x = x.permute(0, 2, 1)  # shape: (batch_size, seq_len, input_dim * num_channels)
        x = x.mean(dim=1)  # shape: (batch_size, input_dim * num_channels)

        # Linear Layers
        x = self.linear1(x)
        x = self.dropout1(x)
        x = self.linear2(x)

        return x

# Define the training function
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    train_acc = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        train_acc += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader.dataset)
    train_acc /= len(train_loader.dataset)

    return train_loss, train_acc

# Define the testing function
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output