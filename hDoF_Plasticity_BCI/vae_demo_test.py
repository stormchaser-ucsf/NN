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
import numpy as np
import numpy.random as rnd
from utils import *
from models import *
import scipy.stats as stats

# setting up GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# getting the data 
condn_data_imagined = np.array(rnd.randn(30000,96))
Y=np.zeros((int(3e4),1))
idx = np.arange(0,1e4,dtype=int)
condn_data_imagined[idx,:] = condn_data_imagined[idx,:] + 0
idx = np.arange(1e4,2e4,dtype=int)
condn_data_imagined[idx,:] = condn_data_imagined[idx,:] - 0.5
Y[idx]=1
idx = np.arange(2e4,3e4,dtype=int)
condn_data_imagined[idx,:] = condn_data_imagined[idx,:] + 1.9
Y[idx]=2
plt.stem(np.mean(condn_data_imagined,axis=1))
plt.figure();plt.stem(Y)

# train and testing split
Y_mult = np.zeros((Y.shape[0],3))
for i in range(Y.shape[0]):
    tmp = int(Y[i])
    Y_mult[i,tmp]=1
Y = Y_mult


Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_imagined,Y,0.8)

# load the data from matlab
#condn_data_trial_Day1_online
file_name = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\condn_data_trial_day1.mat'
data_dict = mat73.loadmat(file_name)
trial_data = data_dict.get('trial_data')
Y = np.array(trial_data.get('targetID'))
data_imagined  = (trial_data.get('neural'))
Xtrain,Xtest,Ytrain,Ytest = training_test_split_trial_online(data_imagined,Y,0.8)


# load the data from matlab
file_name = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\condn_data_imagined_day4.mat'
data_dict = mat73.loadmat(file_name)
data_imagined = data_dict.get('condn_data')
#condn_data_imagined = np.array(condn_data_imagined)
condn_data_imagined = np.zeros((0,96))
Y = np.zeros(0)
for i in np.arange(7):
    tmp = np.array(data_imagined[i])
    condn_data_imagined = np.concatenate((condn_data_imagined,tmp),axis=0)
    idx = i*np.ones((tmp.shape[0],1))[:,0]
    Y = np.concatenate((Y,idx),axis=0)

Y_mult = np.zeros((Y.shape[0],7))
for i in range(Y.shape[0]):
    tmp = round(Y[i])
    Y_mult[i,tmp]=1
Y = Y_mult

# repeatedly sample and get a new data point for at least 2k data points
iterations=2500
append_data = np.empty((iterations,96))
append_y = np.empty((iterations,7))
for iters in np.arange(iterations):
    idx = rnd.choice(Y.shape[0],5,replace=True)
    tmp_data = np.mean(condn_data_imagined[idx,:],axis=0)[None,:]
    tmp_y = Y[idx[0],:]
    append_data[iters,:] = tmp_data
    append_y[iters,:] = tmp_y
    
condn_data_imagined = np.concatenate((condn_data_imagined,append_data),axis=0)
Y = np.concatenate((Y,append_y),axis=0)

Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_imagined,Y,0.9)



for i in np.arange(Xtrain.shape[0]):
    tmp = stats.zscore(Xtrain[i,:])
    Xtrain[i,:] = tmp
    
for i in np.arange(Xtest.shape[0]):
    tmp = stats.zscore(Xtest[i,:])
    Xtest[i,:] = tmp
    


# parameters of the model
input_size=96
hidden_dim=32
latent_dim=2
num_classes=3
dropout=0.3
model = vae(input_size,hidden_dim,latent_dim,num_classes,dropout).to(device)
input_data = torch.randn(128,96).to(device)
xhat,vae_loss = model(input_data)


# training params
num_epochs=30
batch_size=128
learning_rate = 5e-5
batch_val=512
patience=5
gradient_clipping=1
filename='vae_test.pth'
opt = torch.optim.Adam(model.parameters(),lr=learning_rate)
model_goat = training_loop_VAE(model,num_epochs,batch_size,opt,batch_val,
                      patience,gradient_clipping,filename,
                      Xtrain,Ytrain,Xtest,Ytest,
                      vae,input_size,hidden_dim,latent_dim,num_classes,dropout)


# plotting in latent space for just 1 action 
def plot_latent_1action(model, data, Y, num_samples,dim):
    # randomly sample the number of samples 
    idx = rnd.choice(data.shape[0],num_samples)
    data=torch.from_numpy(data).to(device).float()
    Y=torch.from_numpy(Y).to(device).float()
    Y=convert_to_ClassNumbers(Y)
    y=Y[idx]    
    z = data[idx,:]
    model = model.eval()
    z = model.encoder(z)
    z = z.to('cpu').detach().numpy()
    y = y.to('cpu').detach().numpy()    
    #D = sil(z,y)
    D=1
    plt.figure()
    if dim==3:
        ax=plt.axes(projection="3d")
        p=ax.scatter3D(z[:, 0], z[:, 1],z[:,2])        
        plt.colorbar(p)
    if dim==2:
        ax=plt.axes        
        plt.scatter(z[:, 0], z[:, 1])
        plt.colorbar()
    
    return D


        
D = plot_latent_1action(model_goat, Xtrain,Ytrain,Xtrain.shape[0],2)

Xtest_out = model_goat(torch.from_numpy(Xtest[0,:]).to(device).float())[0]
Xtest_out = Xtest_out.to('cpu').detach().numpy()
plt.figure()
plt.stem(Xtest_out)
plt.figure()
plt.stem(Xtest[0,:])
plt.show()


# # playing around with Kl stuff in pytorch 
# p = torch.distributions.Normal(0,1)
# q = torch.distributions.Normal(0,2)


# prob_p=np.array([])
# prob_q=np.array([])
# k_values=np.array([])

# for i in range(100):
#     z = q.rsample()   
#     log_pz = p.log_prob(z)
#     log_qz = q.log_prob(z)
#     prob_p =  np.append(prob_p,torch.exp(log_pz).numpy())
#     prob_q =  np.append(prob_q,torch.exp(log_qz).numpy())
#     d = log_qz - log_pz
#     k_values =  np.append(k_values,d.numpy())
    
# plt.figure()
# plt.hist(k_values)
# plt.figure()
# plt.hist(prob_p)
# plt.figure()
# plt.hist(prob_q)

