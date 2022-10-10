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
    C1 = np.cov(x,rowvar=False) +  1e-3*np.eye(x.shape[1])
    C2 = np.cov(y,rowvar=False) +  1e-3*np.eye(y.shape[1])
    C = (C1+C2)/2
    m1 = np.mean(x,0)
    m2 = np.mean(y,0)
    D = (m2-m1) @ lin.inv(C) @ np.transpose(m2-m1)     
    return D


# get monte carlo estimate of the mahab distance pairwise in data in latent space
def monte_carlo_mahab(data,labels,model,num_samples):
    D=np.zeros((labels.shape[1],labels.shape[1]))
    labels = np.argmax(labels,axis=1)
    data = torch.from_numpy(data).to(device).float()
    z = model.encoder(data)
    for i in np.arange(np.max(labels)+1):
        idxA = (labels==i).nonzero()[0]
        A = z[idxA,:].detach().cpu().numpy()        
        for j in np.arange(i+1,np.max(labels)+1):
            idxB = (labels==j).nonzero()[0]
            B = z[idxB,:].detach().cpu().numpy()      
            D[i,j] = get_mahab_distance(A,B)
            D[j,i] = D[i,j]
    
    return(D)

# get monte carlo estimate of the mahab distance pairwise in data in full space
def monte_carlo_mahab_full(data,labels,num_samples):
    D=np.zeros((labels.shape[1],labels.shape[1]))
    labels = np.argmax(labels,axis=1)    
    z = data
    for i in np.arange(np.max(labels)+1):
        idxA = (labels==i).nonzero()[0]
        A = z[idxA,:]
        for j in np.arange(i+1,np.max(labels)+1):
            idxB = (labels==j).nonzero()[0]
            B = z[idxB,:]
            D[i,j] = get_mahab_distance(A,B)
            D[j,i] = D[i,j]
    
    return(D)


# split into training and validation class trial level
def training_test_split_trial(condn_data,Y,prop):
    len = np.arange(Y.shape[0])
    len_cutoff = round(prop*len[-1])
    idx = np.random.permutation(Y.shape[0])
    train_idx, test_idx = idx[:len_cutoff] , idx[len_cutoff:]
    Xtrain, Xtest = condn_data[train_idx,:,:] , condn_data[test_idx,:,:] 
    Ytrain, Ytest = Y[train_idx,:] , Y[test_idx,:]
    # training data     
    tmp_data=np.zeros((0,96))
    tmp_y = np.zeros((0,7))
    for i in np.arange(Xtrain.shape[0]):
        tmp = np.squeeze(Xtrain[i,:,:])
        tmp_data = np.concatenate((tmp_data,tmp.T),axis=0)
        tmp1 = Ytrain[i,:]
        tmp1 = np.tile(tmp1,(tmp.shape[1],1))
        tmp_y = np.concatenate((tmp_y,tmp1),axis=0)
    Xtrain = tmp_data
    Ytrain = tmp_y
    # shuffle samples
    idx  = np.random.permutation(Ytrain.shape[0])
    Ytrain = Ytrain[idx,:]
    Xtrain = Xtrain[idx,:]
    
    # testing data 
    tmp_data=np.zeros((0,96))
    tmp_y = np.zeros((0,7))
    for i in np.arange(Xtest.shape[0]):
        tmp = np.squeeze(Xtest[i,:,:])
        tmp_data = np.concatenate((tmp_data,tmp.T),axis=0)
        tmp1 = Ytest[i,:]
        tmp1 = np.tile(tmp1,(tmp.shape[1],1))
        tmp_y = np.concatenate((tmp_y,tmp1),axis=0)
    Xtest = tmp_data
    Ytest = tmp_y
    # shuffle samples
    idx  = np.random.permutation(Ytest.shape[0])
    Ytest = Ytest[idx,:]
    Xtest = Xtest[idx,:]    
    
    return Xtrain,Xtest,Ytrain,Ytest    



# split into training and validation class trial level for online data 
def training_test_split_trial_online(condn_data,Y,prop):
    length = np.arange(Y.shape[0])
    len_cutoff = round(prop*length[-1])
    idx = np.random.permutation(Y.shape[0])
    train_idx, test_idx = idx[:len_cutoff] , idx[len_cutoff:]
    
    # training data 
    Xtrain = np.zeros((0,96))
    Ytrain = np.zeros((0,7))
    for i in range(len(train_idx)):
        tmp = condn_data[train_idx[i]]
        Xtrain = np.concatenate((Xtrain,tmp.T),axis=0)
        tmp_idx = round(Y[train_idx[i]]-1)
        tmp_y = np.zeros((1,7))
        tmp_y[:,tmp_idx] = 1
        tmp_y = np.tile(tmp_y,(tmp.shape[1],1))
        Ytrain = np.concatenate((Ytrain,tmp_y),axis=0)
    # shuffle samples
    idx  = np.random.permutation(Ytrain.shape[0])
    Ytrain = Ytrain[idx,:]
    Xtrain = Xtrain[idx,:]
    
    # testing data 
    Xtest = np.zeros((0,96))
    Ytest = np.zeros((0,7))
    for i in range(len(test_idx)):
        tmp = condn_data[test_idx[i]]
        Xtest = np.concatenate((Xtest,tmp.T),axis=0)
        tmp_idx = round(Y[test_idx[i]]-1)
        tmp_y = np.zeros((1,7))
        tmp_y[:,tmp_idx] = 1
        tmp_y = np.tile(tmp_y,(tmp.shape[1],1))
        Ytest = np.concatenate((Ytest,tmp_y),axis=0)
    # shuffle samples
    idx  = np.random.permutation(Ytest.shape[0])
    Ytest = Ytest[idx,:]
    Xtest = Xtest[idx,:]
    
    return Xtrain,Xtest,Ytrain,Ytest    

# split into training and validation class 
def training_test_split(condn_data,Y,prop):
    len = np.arange(Y.shape[0])
    len_cutoff = round(prop*len[-1])
    idx = np.random.permutation(Y.shape[0])
    train_idx, test_idx = idx[:len_cutoff] , idx[len_cutoff:]
    Xtrain, Xtest = condn_data[train_idx,:] , condn_data[test_idx,:] 
    Ytrain, Ytest = Y[train_idx,:] , Y[test_idx,:]
    return Xtrain,Xtest,Ytrain,Ytest


    