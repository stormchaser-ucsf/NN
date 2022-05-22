# -*- coding: utf-8 -*-
"""
Created on Wed May 11 17:37:37 2022

@author: Nikhlesh
"""
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 11:31:37 2022

@author: Nikhlesh
"""
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
import math
plt.rcParams['figure.dpi'] = 200

# setting up GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'



#loading the matlab data
from scipy import io as sio
file_name = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\condn_data'
data=sio.loadmat(file_name, mdict=None, appendmat=True)
data=data.get('condn_data')

# get the data shape to be batch samples, features
condn_data = np.empty([0,32])
l=data.size
Y = np.empty([0])
for i in np.arange(l):
    tmp=data[:,i]
    tmp=tmp[0]
    condn_data = np.append(condn_data,tmp,axis=0)
    idx  = tmp.shape[0]
    Y = np.append(Y,i*np.ones([idx,1]))

data = condn_data
    

# changing the forward pass to be reflective of the VAE framework
class VariationalEncoder(nn.Module):
    def __init__(self,latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(32,16)
        self.linear2 = nn.Linear(16,latent_dims) # for the mean
        self.linear3 = nn.Linear(16,latent_dims) # for the std
        
        # gaussian samplimg
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda() #sampling on GPU loc-mean
        self.N.scale = self.N.scale.cuda() #sampling on GPU scale -std
        self.kl= 0 #KL loss
    
    def forward(self,x):
        #x=torch.flatten(x,start_dim=1) #first dim is batch 
        x=self.linear1(x)
        x=F.elu(x)
        mu = self.linear2(x)
        sigma = self.linear3(x)
        sigma = torch.exp(sigma) # exp of std
        z = mu + sigma*self.N.sample(mu.shape) # sampling from the normal distribution
        kl_loss = (sigma**2 + mu**2 - torch.log(sigma) -1/2).sum
        self.kl = kl_loss
        return z

class decoder(nn.Module):
    def __init__(self,latent_dims):
        super(decoder,self).__init__()
        self.linear1 = nn.Linear(latent_dims,16)
        self.linear2 = nn.Linear(16,32)
        
    def forward(self,z):        
        z = self.linear1(z)
        z = F.elu(z)
        z = self.linear2(z)
        #z = z.reshape((-1, 1, 28, 28))     
        return z


# combining both into one
class VariationalAutoencoder(nn.Module):
    def __init__(self,latent_dims):
        super(VariationalAutoencoder,self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = decoder(latent_dims)
    
    def forward(self,x):
        z=self.encoder(x)
        z=self.decoder(z)
        return z


# creating the model
latent_dims = 3
vae = VariationalAutoencoder(latent_dims).to(device) # GPU
criterion = nn.MSELoss(reduction='sum')
opt = torch.optim.Adam(vae.parameters())
beta_params=0;

# training loop
num_epochs=15
batch_size=64
num_batches = math.ceil(data.shape[0]/batch_size)

for epoch in range(num_epochs):
    # split the data into batches
    idx = np.random.permutation(data.shape[0])
    data1 = data[idx,:]
    
    for i in range(num_batches):
        k = i*batch_size
        k1 = k+batch_size
        
        if k1>data1.shape[0]:
            k1=data1.shape[0]
            
        x = data1[k:k1,:]
        x=torch.from_numpy(x)
        x=x.to(device) # push it to GPU
        x=x.to(torch.float32) # convert to single
        opt.zero_grad() # flush gradients
        xhat = vae(x)
        loss_kl = vae.encoder.kl()
        #print(loss_kl)
        #loss = ((x - xhat)**2).sum() + loss_kl
        loss=criterion(x,xhat) + beta_params*vae.encoder.kl()                  
        loss.backward()
        opt.step()
    print(epoch+1)
    
   
    
def plot_latent(ae, data, Y, num_samples):
    # randomly sample the number of samples 
    idx = np.random.choice(len(data),num_samples)
    y=Y[idx]
    z = data[idx,:]
    z = torch.from_numpy(z)
    z=z.to(device)
    z=z.to(torch.float32)
    z = ae.encoder(z)
    z = z.to('cpu').detach().numpy()
    plt.figure
    ax=plt.axes(projection="3d")
    p=ax.scatter3D(z[:, 0], z[:, 1],z[:,2], c=y, cmap='tab10')
    p=ax.scatter3D(z[:, 0], z[:, 1],z[:,2], c=y, cmap='tab10')
    plt.colorbar(p)
        
plot_latent(vae, data,Y,1000)


# now plot data from online thru the built autoencoder     
file_name = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\condn_data_online'
data_online=sio.loadmat(file_name, mdict=None, appendmat=True)
data_online=data_online.get('condn_data_online')

# get the data shape to be batch samples, features
condn_data = np.empty([0,32])
l=data_online.size
Y_online = np.empty([0])
for i in np.arange(l):
    tmp=data_online[:,i]
    tmp=tmp[0]
    condn_data = np.append(condn_data,tmp,axis=0)
    idx  = tmp.shape[0]
    Y_online = np.append(Y_online,i*np.ones([idx,1]))

data_online = condn_data

plot_latent(vae, data_online,Y_online,100)










