

"""
Created on Fri Jul 30 22:47:39 2021

@author: Nikhlesh
"""

import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import math
import mat73
import numpy.random as rnd
import numpy.linalg as lin
from sklearn.model_selection  import train_test_split
import os
plt.rcParams['figure.dpi'] = 200
from utils import *
from models import *
import sklearn as skl
from sklearn.metrics import silhouette_score as sil
from sklearn.metrics import silhouette_samples as sil_samples


# setting up GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""
OVERALL PLAN FOR THE ANALYSES
@author: Nikhlesh
STEP 1: Build iAE seperately for imagined and online, using held out samples, 
        then track mean diff and variance across days. Compare cross projected
        variance and mean diff statistics, and silhoutte index
STEP 2: Repeat Step 1 but for regular AE, VAE, iVAE etc. Choose which gives best
STEP 3: Repeat Steps 1 and 2, but now for an AE validated on held out trial data.
        Basically saying that within a common manifold, the online data achieves greater 
        separation in latent space than imagined data that it is built for. 
STEP 4: Look at activity in the final decoder layer (before or after sofmax) after
        training on imagined data; is change in neural activity reflected in the 
        decoder space for classifying each action? Get better at the emission of 
        actions in sequence for the mode filter? 

"""


# load the data from matlab
#condn_data_trial_Day1_online
file_name = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\condn_data_trial_Day1_online.mat'
data_dict = mat73.loadmat(file_name)
trial_data = data_dict.get('trial_data')
Y = np.array(trial_data.get('targetID'))
data_imagined  = (trial_data.get('neural'))

Xtrain,Xtest,Ytrain,Ytest = training_test_split_trial_online(data_imagined,Y,0.8)

tmp_data = np.zeros((0,96))
y_temp = np.zeros((0,1))
for i in range(len(data_imagined)):
    tmp = np.array(data_imagined[i])
    tmp_data = np.concatenate((tmp_data,tmp.T))
    tmp1 = round(Y[i])
    tmp1 = np.tile(tmp1,(tmp.shape[1],1))
    y_temp = np.concatenate((y_temp,tmp1),axis=0)
    
condn_data_online = tmp_data
Y = y_temp

Y_mult = np.zeros((Y.shape[0],7))
for i in range(Y.shape[0]):
    tmp = round((Y[i]-1)[0])
    Y_mult[i,tmp]=1
Yonline = Y_mult

# random split
Xtrain,Xtest,Ytrain,Ytest = training_test_split_trial(data_imagined,Y,0.8)


#condn_data_imagined = scale_zero_one(condn_data_imagined)            
#Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_imagined,Y,0.8)
Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_online,Yonline,0.8)


# testing it out
input_size=96
hidden_size = 32
latent_dims = 3
num_classes = 7 
model = get_model(input_size,hidden_size,latent_dims,num_classes)
input = torch.randn(64,96).to(device)
(recon,decodes) = model(input)


init_loss,init_acc,init_recon = validation_loss(model,Xtest,Ytest,512,1)
print('Initial loss, acc,  recon error')
print(init_loss,init_acc,init_recon)


# TRAINING LOOP
# minimize the recon loss as well as the classification loss
# return the model with lowest validation loss 
num_epochs=200
batch_size=32
learning_rate = 5e-4
opt = torch.optim.Adam(model.parameters(),lr=learning_rate)
batch_val=512
patience=6
gradient_clipping=10
filename='autoencoder.pth'
model_goat = training_loop_iAE(model,num_epochs,batch_size,opt,batch_val,patience,
                           gradient_clipping,filename)


  
condn_data_imagined = np.concatenate((Xtrain,Xtest),axis=0)     
Y =  np.concatenate((Ytrain,Ytest),axis=0)     
        
D = plot_latent(model_goat, Xtest,Ytest,Xtest.shape[0],3)

D = plot_latent(model_goat, condn_data_imagined,Y,2100,3)



# X=rnd.randn(10,12)
# X[[0,3,4,7],:] = X[[0,3,4,7],:] + 25
# X[[1,5,9],:] = X[[1,5,9],:] + 10
# X[[2,6,8],:] = X[[2,6,8],:] - 40
# labels = np.array([1,2,3,1,1,2,3,1,3,2])
# D = sil(X,labels)
# print(1-D)

# D = sil_samples(X,labels)
# print(np.mean(1-D))

D = monte_carlo_mahab(condn_data_imagined,Y,model_goat,0)
D = D[np.triu_indices(D.shape[0])]
D = D[D>0]
imagined_data_D = np.mean(D)
print(imagined_data_D)

D = monte_carlo_mahab_full(condn_data_imagined,Y,0)
plt.imshow(D)
plt.colorbar()
D = D[np.triu_indices(D.shape[0])]
D = D[D>0]
imagined_data_Dfull = np.mean(D)
print(imagined_data_Dfull)


# now load the testing data 
file_name = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\condn_data_online.mat'
data_online=mat73.loadmat(file_name)
data_online = data_online.get('condn_data')
condn_data_online = np.zeros((0,96))
Yonline = np.zeros(0)
for i in np.arange(7):
    tmp = np.array(data_online[i])
    condn_data_online = np.concatenate((condn_data_online,tmp),axis=0)
    idx = i*np.ones((tmp.shape[0],1))[:,0]
    Yonline = np.concatenate((Yonline,idx),axis=0)

Y_mult = np.zeros((Yonline.shape[0],7))
for i in range(Yonline.shape[0]):
    tmp = round(Yonline[i])
    Y_mult[i,tmp]=1
Yonline = Y_mult

    
D = plot_latent(model_goat, condn_data_online,Yonline,696,3)
D = monte_carlo_mahab(condn_data_online,Yonline,model_goat,0)
D = D[np.triu_indices(D.shape[0])]
D = D[D>0]
online_data_D = np.mean(D)
print(online_data_D)


D = monte_carlo_mahab_full(condn_data_online,Yonline,0)
plt.imshow(D)
plt.colorbar()
D = D[np.triu_indices(D.shape[0])]
D = D[D>0]
online_data_Dfull = np.mean(D)
print(online_data_Dfull)

# # now plot data from online thru the built autoencoder     

# # get the data shape to be batch samples, features
# condn_data = np.empty([0,32])
# l=data_online.size
# Y_online = np.empty([0])
# for i in np.arange(l):
#     tmp=data_online[:,i]
#     tmp=tmp[0]
#     condn_data = np.append(condn_data,tmp,axis=0)
#     idx  = tmp.shape[0]
#     Y_online = np.append(Y_online,i*np.ones([idx,1]))

# data_online = condn_data


# plot_latent(vae, data_online,Y_online,100,3)










