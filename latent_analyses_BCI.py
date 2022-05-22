
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
        self.linear1 = nn.Linear(32,8)        
        self.linear2 = nn.Linear(8,latent_dims) 
    
    def forward(self,x):        
        x=self.linear1(x)
        x=F.elu(x)
        x=self.linear2(x)
        x=F.elu(x)        
        return x

class decoder(nn.Module):
    def __init__(self,latent_dims):
        super(decoder,self).__init__()
        self.linear1 = nn.Linear(latent_dims,8)
        self.linear2 = nn.Linear(8,32)
        
        
    def forward(self,z):        
        z = self.linear1(z)
        z = F.elu(z)
        z = self.linear2(z)        
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
criterion = nn.MSELoss(reduction='mean')
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
        loss_kl =0
        #print(loss_kl)
        #loss = ((x - xhat)**2).sum() + loss_kl
        loss=criterion(x,xhat) + beta_params*loss_kl
        loss.backward()
        opt.step()    
    print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    
   
    
def plot_latent(ae, data, Y, num_samples,dim):
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
    if dim==3:
        ax=plt.axes(projection="3d")
        p=ax.scatter3D(z[:, 0], z[:, 1],z[:,2], c=y, cmap='tab10')
        p=ax.scatter3D(z[:, 0], z[:, 1],z[:,2], c=y, cmap='tab10')
        plt.colorbar(p)
    if dim==2:
        ax=plt.axes
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        plt.colorbar()
        
plot_latent(vae, data,Y,1000,3)


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


plot_latent(vae, data_online,Y_online,100,3)










