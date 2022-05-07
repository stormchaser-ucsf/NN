# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 22:47:39 2021

@author: Nikhlesh
"""
# testing variational autoencoders on the MNIST database using pytorch

# importing everything
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200

# setting up GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# creat encoder module with linear layers
class encoder(nn.Module):
    def __init__(self,latent_dims):
        super(encoder,self).__init__()
        self.linear1 = nn.Linear(784,256)
        self.linear2 = nn.Linear(256,latent_dims)
        
    def forward(self,x):
        x = torch.flatten(x,start_dim=1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x

class decoder(nn.Module):
    def __init__(self,latent_dims):
        super(decoder,self).__init__()
        self.linear1 = nn.Linear(latent_dims,256)
        self.linear2 = nn.Linear(256,784)
        
    def forward(self,z):        
        z = self.linear1(z)
        z = F.relu(z)
        z = self.linear2(z)
        z = z.reshape((-1, 1, 28, 28))     
        return z

class autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(autoencoder,self).__init__()
        self.encoder = encoder(latent_dims)
        self.decoder = decoder(latent_dims)
        
    def forward(self,x):
        z=self.encoder(x)
        z=self.decoder(z)
        return z
    
# training loop to train on MNIST dataset 
data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data',
               transform=torchvision.transforms.ToTensor(),
               download=True),
        batch_size=128,
        shuffle=True)

num_epochs=25
n_total_steps = len(data)
criterion = nn.MSELoss(reduction='sum')
def train(linearAE, data, epochs =num_epochs):
    opt  = torch.optim.Adam(linearAE.parameters())
    for epoch in range(epochs):
        for i,(x,y) in enumerate(data):
            x=x.to(device) # push it to GPU
            opt.zero_grad() # flush gradients
            xhat = linearAE(x)
            #loss = ((x - xhat)**2).sum()
            loss=criterion(x,xhat)
            loss.backward()
            opt.step()
            #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    return linearAE

latent_dims = 3
linearAE = autoencoder(latent_dims).to(device) # GPU

linearAE = train(linearAE, data)

# # pytorch engineer method of training 
# criterion = nn.MSELoss()
# opt  = torch.optim.Adam(linearAE.parameters())
# n_total_steps = len(data)
# for epoch in range(num_epochs):
#     for i,(x,y) in enumerate(data):
#         x=x.to(device)
        
#         # foward pass
#         xhat = linearAE(x)
#         #loss = ((x - xhat)**2).sum()
#         loss = criterion(x,xhat)
        
#         #backward pass and optimize
#         opt.zero_grad() # flush gradients
#         loss.backward()
#         opt.step()
#         if (i+1) % 100 == 0:
#             print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

#### end pytroch engineer method of training        





# plotting
def plot_latent(linearAE, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = linearAE.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1],z[:, 2], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
plot_latent(linearAE, data)



x=torch.randn(100,28,28)
z = linearAE.encoder(x.to(device))
z = z.to('cpu').detach().numpy()
plt.scatter(z[:, 0], z[:, 1],z[:, 2])

# ################## extra stuff ##################
# # plotting images using matplotlib
# fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1) # two axes on figure
# x=torch.randn(28,28)
# ax1.imshow(x)

# # understanding the flatten and the reshape functions
# x = torch.flatten(x)
# z=x
# z=z.reshape(28,28)
# ax2.imshow(z)

# x=np.random.randn(28,28)
# y=np.random.randn(28,28)
# z=((x-y)**2)
# print(z)



# nn.Linear












