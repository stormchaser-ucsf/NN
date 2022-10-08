# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 22:29:06 2022

@author: nikic
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
from sklearn.metrics import silhouette_score as sil
from sklearn.metrics import silhouette_samples as sil_samples



# setting up GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# scale each data sample to be within 0 and 1
def scale_zero_one(indata):
    for i in range(indata.shape[0]):
        a = indata[i,:]
        a = (a-a.min())/(a.max()-a.min())
        indata[i,:] = a
    
    return(indata)


# function to get mahalanobis distance
def get_mahab_distance(x,y):
    C1 = np.cov(x,rowvar=False) +  1e-8*np.eye(x.shape[1])
    C2 = np.cov(y,rowvar=False) +  1e-8*np.eye(y.shape[1])
    C = (C1+C2)/2
    m1 = np.mean(x,0)
    m2 = np.mean(y,0)
    D = (m2-m1) @ lin.inv(C) @ np.transpose(m2-m1)     
    return D
    

# split into training and validation class 
def training_test_split(condn_data,Y,prop):
    len = np.arange(Y.shape[0])
    len_cutoff = round(prop*len[-1])
    idx = np.random.permutation(Y.shape[0])
    train_idx, test_idx = idx[:len_cutoff] , idx[len_cutoff:]
    Xtrain, Xtest = condn_data[train_idx,:] , condn_data[test_idx,:] 
    Ytrain, Ytest = Y[train_idx,:] , Y[test_idx,:]
    return Xtrain,Xtest,Ytrain,Ytest


# function to convert one-hot representation back to class numbers
def convert_to_ClassNumbers(indata):
    with torch.no_grad():
        outdata = torch.max(indata,1).indices
    
    return outdata


# function to validate model 
def validation_loss(model,X_test,Y_test,batch_val,val_type):    
    crit_classif_val = nn.CrossEntropyLoss(reduction='sum') #if mean, it is over all samples
    crit_recon_val = nn.MSELoss(reduction='sum') # if mean, it is over all elements 
    loss_val=0    
    accuracy=0
    recon_error=0
    if batch_val > X_test.shape[0]:
        batch_val = X_test.shape[0]
    
    idx=np.arange(0,X_test.shape[0],batch_val)    
    if idx[-1]<X_test.shape[0]:
        idx=np.append(idx,X_test.shape[0])
    else:
        print('something wrong here')
    
    iters=(idx.shape[0]-1)
    
    for i in np.arange(iters):
        x=X_test[idx[i]:idx[i+1],:]
        y=Y_test[idx[i]:idx[i+1],:]     
        with torch.no_grad():                
            if val_type==1: #validation
                x=torch.from_numpy(x).to(device).float()
                y=torch.from_numpy(y).to(device).float()
                model.eval()
                out,ypred = model(x) 
                loss1 = crit_recon_val(out,x)
                loss2 = crit_classif_val(ypred,y)
                loss_val += loss1.item() + loss2.item()
                model.train()
            else:
                out,ypred = model(x) 
                loss1 = crit_recon_val(out,x)
                loss2 = crit_classif_val(ypred,y)
                loss_val += loss1.item() + loss2.item()
            
            ylabels = convert_to_ClassNumbers(y)        
            ypred_labels = convert_to_ClassNumbers(ypred)     
            accuracy += torch.sum(ylabels == ypred_labels).item()
            recon_error += (torch.sum(torch.square(out-x))).item()   
            
    loss_val=loss_val/X_test.shape[0]
    accuracy = accuracy/X_test.shape[0]
    recon_error = (recon_error/X_test.shape[0])#.cpu().numpy()
    torch.cuda.empty_cache()
    return loss_val,accuracy,recon_error



# function to validate model 
def validation_loss_cluster(model,X_test,Y_test,batch_val,val_type):       
    crit_recon_val = nn.MSELoss(reduction='sum') # if mean, it is over all elements 
    loss_val=0    
    cluster_fitness=0
    recon_error=0
    if batch_val > X_test.shape[0]:
        batch_val = X_test.shape[0]
    
    idx=np.arange(0,X_test.shape[0],batch_val)    
    if idx[-1]<X_test.shape[0]:
        idx=np.append(idx,X_test.shape[0])
    else:
        print('something wrong here')
    
    iters=(idx.shape[0]-1)
    
    for i in np.arange(iters):
        x=X_test[idx[i]:idx[i+1],:]
        y=Y_test[idx[i]:idx[i+1],:]     
        with torch.no_grad():                
            if val_type==1: #validation
                x=torch.from_numpy(x).to(device).float()
                y=torch.from_numpy(y).to(device).float()
                model.eval()
                out = model(x) 
                out_latent = model.encoder(x)
                loss1 = crit_recon_val(out,x)
                loss2 = 1 - sil_samples(out_latent.detach().cpu().numpy(),
                        convert_to_ClassNumbers(y).detach().cpu().numpy())
                loss2 = torch.tensor(loss2).to(device).float()                
                loss_val += loss1.item() + loss2.sum().item()
                model.train()
            else: #do nothing
                loss1=0
                loss2 = 0                
                loss_val += loss1.item() + loss2.sum().item()         
            
            cluster_fitness += loss2.sum().item()            
            recon_error += (torch.sum(torch.square(out-x))).item()   
            
    loss_val=loss_val/X_test.shape[0]
    cluster_fitness = cluster_fitness/X_test.shape[0]
    recon_error = (recon_error/X_test.shape[0])#.cpu().numpy()
    torch.cuda.empty_cache()
    return loss_val,cluster_fitness,recon_error


# plotting in latent space
def plot_latent(model, data, Y, num_samples,dim):
    # randomly sample the number of samples 
    idx = rnd.choice(data.shape[0],num_samples)
    data=torch.from_numpy(data).to(device).float()
    Y=torch.from_numpy(Y).to(device).float()
    Y=convert_to_ClassNumbers(Y)
    y=Y[idx]    
    z = data[idx,:]
    z = model.encoder(z)
    z = z.to('cpu').detach().numpy()
    y = y.to('cpu').detach().numpy()    
    D = sil(z,y)
    plt.figure()
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
    
    return D

