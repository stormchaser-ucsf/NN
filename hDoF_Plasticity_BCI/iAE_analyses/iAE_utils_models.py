# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 23:22:57 2022

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
import os
plt.rcParams['figure.dpi'] = 200
import sklearn as skl
from sklearn.metrics import silhouette_score as sil
from sklearn.metrics import silhouette_samples as sil_samples
import scipy.stats as stats


# setting up GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#### UTILS SECTION


# get the data 
def get_data(filename,num_classes):
    data_dict = mat73.loadmat(filename)
    data_imagined = data_dict.get('condn_data')
    condn_data_imagined = np.zeros((0,data_imagined[0].shape[1]))
    Y = np.zeros(0)
    for i in np.arange(num_classes):
        tmp = np.array(data_imagined[i])
        condn_data_imagined = np.concatenate((condn_data_imagined,tmp),axis=0)
        idx = i*np.ones((tmp.shape[0],1))[:,0]
        Y = np.concatenate((Y,idx),axis=0)

    Y_mult = np.zeros((Y.shape[0],num_classes))
    for i in range(Y.shape[0]):
        tmp = round(Y[i])
        Y_mult[i,tmp]=1
    Y = Y_mult
    return condn_data_imagined, Y

# get the data 
def get_data_B2(filename):
    data_dict = mat73.loadmat(filename)
    data_imagined = data_dict.get('condn_data')
    condn_data_imagined = np.zeros((0,96))
    Y = np.zeros(0)
    for i in np.arange(4):
        tmp = np.array(data_imagined[i])
        condn_data_imagined = np.concatenate((condn_data_imagined,tmp),axis=0)
        idx = i*np.ones((tmp.shape[0],1))[:,0]
        Y = np.concatenate((Y,idx),axis=0)

    Y_mult = np.zeros((Y.shape[0],4))
    for i in range(Y.shape[0]):
        tmp = round(Y[i])
        Y_mult[i,tmp]=1
    Y = Y_mult
    return condn_data_imagined, Y


# median bootstrap
def median_bootstrap(indata,iters):
    out_boot=  np.zeros([iters,indata.shape[1]])  
    for cols in np.arange(indata.shape[1]):
        xx = indata[:,cols]
        for i in np.arange(iters):
            idx = rnd.choice(indata.shape[0],indata.shape[0])
            xx_tmp = np.median(xx[idx])
            out_boot[i,cols] = xx_tmp
    out_boot = np.sort(out_boot,axis=0)
    out_boot_std = np.std(out_boot,axis=0)
    return out_boot, out_boot_std

# scale each data sample to be within 0 and 1
def scale_zero_one(indata):
    for i in range(indata.shape[0]):
        a = indata[i,:]
        a = (a-a.min())/(a.max()-a.min())
        indata[i,:] = a
    
    return(indata)


def get_distance_means(z,idx,num_classes):        
    dist_means=np.array([])
    for i in np.arange(num_classes):
        idxA = (idx==i).nonzero()[0]
        A = z[idxA,:]        
        for j in np.arange(i+1,num_classes):
            idxB = (idx==j).nonzero()[0]
            B = z[idxB,:]
            d = np.mean(A,axis=0)-np.mean(B,axis=0)
            d = (d @ d.T) ** (0.5)
            dist_means = np.append(dist_means,d)
    return dist_means

def get_distance_means_B2(z,idx):        
    dist_means=np.array([])
    for i in np.arange(4):
        idxA = (idx==i).nonzero()[0]
        A = z[idxA,:]        
        for j in np.arange(i+1,4):
            idxB = (idx==j).nonzero()[0]
            B = z[idxB,:]
            d = np.mean(A,axis=0)-np.mean(B,axis=0)
            d = (d @ d.T) ** (0.5)
            dist_means = np.append(dist_means,d)
    return dist_means

def get_variances(z,idx,num_classes):
    dist_var = np.empty((num_classes,))
    for i in np.arange(len(np.unique(idx))):
        idxA = (idx==i).nonzero()[0]
        A = z[idxA,:]
       # A = stats.zscore(A,axis=0)
        C = np.cov(A,rowvar=False)
        if len(C.shape) > 0:
            C = C + 1e-12*np.identity(C.shape[0])
            A = lin.det(C)
        elif len(C.shape) == 0:
            A = C
        dist_var[i] = A
    return dist_var

def get_variances_B2(z,idx):
    dist_var = np.empty((4,))
    for i in np.arange(len(np.unique(idx))):
        idxA = (idx==i).nonzero()[0]
        A = z[idxA,:]
       # A = stats.zscore(A,axis=0)
        C = np.cov(A,rowvar=False)
        if len(C.shape) > 0:
            C = C + 1e-5*np.identity(C.shape[0])
            A = lin.det(C)
        elif len(C.shape) == 0:
            A = C
        dist_var[i] = A
    return dist_var

def get_variance_overall(z):
    C = np.cov(z,rowvar=False)
    if lin.matrix_rank(C) == C.shape[0]:
        A = lin.det(C)
    else:
        C = C + 1e-12*np.identity(C.shape[0])
        A = lin.det(C)
    return A


# function to get mahalanobis distance
def get_mahab_distance(x,y):
    C1 = np.cov(x,rowvar=False) +  1e-9*np.eye(x.shape[1])
    C2 = np.cov(y,rowvar=False) +  1e-9*np.eye(y.shape[1])
    C = (C1+C2)/2
    m1 = np.mean(x,0)
    m2 = np.mean(y,0)
    D = (m2-m1) @ lin.inv(C) @ np.transpose(m2-m1)     
    return D

# function to get mahalanobis distance
def get_mahab_distance_latent(z,idx,num_classes):
    mdist =  np.zeros([num_classes,num_classes])
    for i in np.arange(len(np.unique(idx))):
        idxA = (idx==i).nonzero()[0]
        A = z[idxA,:]
        for j in np.arange(i+1,len(np.unique(idx))):
            idxB = (idx==j).nonzero()[0]
            B = z[idxB,:]
            mdist[i,j] = get_mahab_distance(A, B)    
            mdist[j,i] = mdist[i,j]
    return mdist

# function to get mahalanobis distance
def get_mahab_distance_latent_B2(z,idx):
    mdist =  np.zeros([4,4])
    for i in np.arange(len(np.unique(idx))):
        idxA = (idx==i).nonzero()[0]
        A = z[idxA,:]
        for j in np.arange(i+1,len(np.unique(idx))):
            idxB = (idx==j).nonzero()[0]
            B = z[idxB,:]
            mdist[i,j] = get_mahab_distance(A, B)    
            mdist[j,i] = mdist[i,j]
    return mdist


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
    len1 = np.arange(Y.shape[0])
    len_cutoff = round(prop*len1[-1])
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
    len1 = np.arange(Y.shape[0])
    len_cutoff = round(prop*len1[-1])
    idx = np.random.permutation(Y.shape[0])
    train_idx, test_idx = idx[:len_cutoff] , idx[len_cutoff:]
    Xtrain, Xtest = condn_data[train_idx,:] , condn_data[test_idx,:] 
    Ytrain, Ytest = Y[train_idx,:] , Y[test_idx,:]
    return Xtrain,Xtest,Ytrain,Ytest

# split into training, testing and validation class 
def training_test_split_val(condn_data,Y,prop):
    # prop training, (1-prop)/2 each for val and testing
    len1 = np.arange(Y.shape[0])
    len_cutoff = round(prop*len1[-1])
    idx = np.random.permutation(Y.shape[0])
    train_idx, leftover_idx = idx[:len_cutoff] , idx[len_cutoff:]
    Xtrain, Xleftover = condn_data[train_idx,:] , condn_data[leftover_idx,:] 
    Ytrain, Yleftover = Y[train_idx,:] , Y[leftover_idx,:]
    # now split left over data in half
    len2_cutoff = round(Yleftover.shape[0]/2)
    Xval,Xtest = Xleftover[:len2_cutoff,:], Xleftover[len2_cutoff:,:]
    Yval,Ytest = Yleftover[:len2_cutoff,:], Yleftover[len2_cutoff:,:]    
    return Xtrain,Xtest,Xval,Ytrain,Ytest,Yval

####### MODELS SECTION
# function to convert one-hot representation back to class numbers
def convert_to_ClassNumbers(indata):
    with torch.no_grad():
        outdata = torch.max(indata,1).indices
    
    return outdata


# create a autoencoder with a classifier layer for separation in latent space
class encoder(nn.Module):
    def __init__(self,input_size,hidden_size,latent_dims,num_classes):
        super(encoder,self).__init__()
        self.hidden_size2 = round(hidden_size/3)
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,self.hidden_size2)
        self.linear3 = nn.Linear(self.hidden_size2,latent_dims)
        self.gelu = nn.ELU()
        self.tanh = nn.Tanh()
        self.dropout =  nn.Dropout(p=0.3)
        #self.lnorm1 = nn.LayerNorm(latent_dims,elementwise_affine=False)
        
    def forward(self,x):
        x=self.linear1(x)
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.linear2(x)        
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.linear3(x)
        #x=self.lnorm1(x)
        #x=self.tanh(x)
        return x
    
# create a autoencoder for b3 with a classifier layer for separation in latent space
class encoder_b3(nn.Module):
    def __init__(self,input_size,hidden_size,latent_dims,num_classes):
        super(encoder_b3,self).__init__()
        self.hidden_size2 = round(hidden_size/2)
        self.hidden_size3 = round(hidden_size/4)
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,self.hidden_size2)
        self.linear3 = nn.Linear(self.hidden_size2,self.hidden_size3)        
        self.linear4 = nn.Linear(self.hidden_size3,latent_dims)
        self.gelu = nn.ELU()
        self.tanh = nn.Tanh()
        self.dropout =  nn.Dropout(p=0.3)
        #self.lnorm1 = nn.LayerNorm(latent_dims,elementwise_affine=False)
        
    def forward(self,x):
        x=self.linear1(x)
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.linear2(x)        
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.linear3(x)
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.linear4(x)
        #x=self.lnorm1(x)
        #x=self.tanh(x)
        return x

class latent_classifier(nn.Module):
    def __init__(self,latent_dims,num_classes):
        super(latent_classifier,self).__init__()
        self.linear1 = nn.Linear(latent_dims,num_classes)
        self.weights = torch.randn(latent_dims,num_classes).to(device)        
    
    def forward(self,x):
        x=self.linear1(x)        
        #x=torch.matmul(x,self.weights)
        return x


# class latent_classifier(nn.Module):
#     def __init__(self,latent_dims,num_classes):
#         super(latent_classifier,self).__init__()
#         self.linear1 = nn.Linear(latent_dims,latent_dims*3)
#         self.linear2 = nn.Linear(latent_dims*3,latent_dims*4)
#         self.linear3 = nn.Linear(latent_dims*4,num_classes)
#         #self.weights = torch.randn(latent_dims,num_classes).to(device)        
#         self.gelu =  nn.GELU()
    
#     def forward(self,x):
#         x=self.linear1(x)        
#         x=self.gelu(x)
#         x=self.linear2(x)        
#         x=self.gelu(x)
#         x=self.linear3(x)                
#         #x=torch.matmul(x,self.weights)
#         return x
    
class recon_classifier(nn.Module):
    def __init__(self,input_size,num_classes):
        super(recon_classifier,self).__init__()
        self.linear1 = nn.Linear(input_size,num_classes)
        self.weights = torch.randn(input_size,num_classes)
        self.dropout =  nn.Dropout(p=0.3)
    
    def forward(self,x):
        x=self.linear1(x)
        return x

class decoder(nn.Module):
    def __init__(self,input_size,hidden_size,latent_dims,num_classes):
        super(decoder,self).__init__()
        self.hidden_size2 = round(hidden_size/3)
        self.linear1 = nn.Linear(latent_dims,self.hidden_size2)
        self.linear2 = nn.Linear(self.hidden_size2,hidden_size)
        self.linear3 = nn.Linear(hidden_size,input_size)
        self.gelu = nn.ELU()
        self.relu = nn.ReLU()
        self.dropout =  nn.Dropout(p=0.3)
        
        
    def forward(self,x):
        x=self.linear1(x)
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.linear2(x)
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.linear3(x)        
        return x

class decoder_b3(nn.Module):
    def __init__(self,input_size,hidden_size,latent_dims,num_classes):
        super(decoder_b3,self).__init__()
        self.hidden_size2 = round(hidden_size/2)
        self.hidden_size3 = round(hidden_size/4)
        self.linear1 = nn.Linear(latent_dims,self.hidden_size3)
        self.linear2 = nn.Linear(self.hidden_size3,self.hidden_size2)
        self.linear3 = nn.Linear(self.hidden_size2,hidden_size)
        self.linear4 = nn.Linear(hidden_size,input_size)
        self.gelu = nn.ELU()
        self.relu = nn.ReLU()
        self.dropout =  nn.Dropout(p=0.3)
        
        
    def forward(self,x):
        x=self.linear1(x)
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.linear2(x)
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.linear3(x)        
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.linear4(x)        
        return x

# combining all into 
class iAutoencoder(nn.Module):
    def __init__(self,input_size,hidden_size,latent_dims,num_classes):
        super(iAutoencoder,self).__init__()
        self.encoder = encoder(input_size,hidden_size,latent_dims,num_classes)
        self.decoder = decoder(input_size,hidden_size,latent_dims,num_classes)
        self.latent_classifier = latent_classifier(latent_dims,num_classes)
        #self.recon_classifier = recon_classifier(input_size,num_classes)
    
    def forward(self,x):
        z=self.encoder(x)
        y=self.latent_classifier(z)
        z=self.decoder(z)
        #y=self.recon_classifier(z)
        return z,y
    
# combining all into 
class Autoencoder(nn.Module):
    def __init__(self,input_size,hidden_size,latent_dims,num_classes):
        super(iAutoencoder,self).__init__()
        self.encoder = encoder(input_size,hidden_size,latent_dims,num_classes)
        self.decoder = decoder(input_size,hidden_size,latent_dims,num_classes)
        #self.recon_classifier = recon_classifier(input_size,num_classes)
    
    def forward(self,x):
        z=self.encoder(x)        
        z=self.decoder(z)
        #y=self.recon_classifier(z)
        return z   

# combining all into 
class iAutoencoder_B3(nn.Module):
    def __init__(self,input_size,hidden_size,latent_dims,num_classes):
        super(iAutoencoder_B3,self).__init__()
        self.encoder = encoder_b3(input_size,hidden_size,latent_dims,num_classes)
        self.decoder = decoder_b3(input_size,hidden_size,latent_dims,num_classes)
        self.latent_classifier = latent_classifier(latent_dims,num_classes)
        #self.recon_classifier = recon_classifier(input_size,num_classes)
    
    def forward(self,x):
        z=self.encoder(x)
        y=self.latent_classifier(z)
        z=self.decoder(z)
        #y=self.recon_classifier(z)
        return z,y
    
# combining all into 
class Autoencoder(nn.Module):
    def __init__(self,input_size,hidden_size,latent_dims,num_classes):
        super(iAutoencoder,self).__init__()
        self.encoder = encoder(input_size,hidden_size,latent_dims,num_classes)
        self.decoder = decoder(input_size,hidden_size,latent_dims,num_classes)
        #self.recon_classifier = recon_classifier(input_size,num_classes)
    
    def forward(self,x):
        z=self.encoder(x)        
        z=self.decoder(z)
        #y=self.recon_classifier(z)
        return z  


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

# TRAINING LOOP
def training_loop_iAE(model,num_epochs,batch_size,learning_rate,batch_val,
                      patience,gradient_clipping,filename,
                      Xtrain,Ytrain,Xtest,Ytest,
                      input_size,hidden_size,latent_dims,num_classes):
    
   
    num_batches = math.ceil(Xtrain.shape[0]/batch_size)
    recon_criterion = nn.MSELoss(reduction='sum')
    classif_criterion = nn.CrossEntropyLoss(reduction='sum')    
    opt = torch.optim.Adam(model.parameters(),lr=learning_rate)
    print('Starting training')
    goat_loss=99999
    counter=0
    model.train()
    for epoch in range(num_epochs):
      #shuffle the data    
      #shuffle the data    
      idx = rnd.permutation(Xtrain.shape[0]) 
      idx_split = np.array_split(idx,num_batches)
      
      
      if epoch>round(num_epochs*0.6):
          for g in opt.param_groups:
              g['lr']=1e-4
        
      for batch in range(num_batches):
          # get the batch 
          samples = idx_split[batch]
          Xtrain_batch = Xtrain[samples,:]
          Ytrain_batch = Ytrain[samples,:]        
          
          #push to gpu
          Xtrain_batch = torch.from_numpy(Xtrain_batch).to(device).float()
          Ytrain_batch = torch.from_numpy(Ytrain_batch).to(device).float()          
          
          # forward pass thru network
          opt.zero_grad() 
          recon,decodes = model(Xtrain_batch)
          latent_activity = model.encoder(Xtrain_batch)      
          
          # get loss      
          recon_loss = (recon_criterion(recon,Xtrain_batch))/Xtrain_batch.shape[0]
          classif_loss = (classif_criterion(decodes,Ytrain_batch))/Xtrain_batch.shape[0]      
          loss = recon_loss + classif_loss
          total_loss = loss.item()
          #print(classif_loss.item())
          
          # compute accuracy
          ylabels = convert_to_ClassNumbers(Ytrain_batch)        
          ypred_labels = convert_to_ClassNumbers(decodes)     
          accuracy = (torch.sum(ylabels == ypred_labels).item())/ylabels.shape[0]
          
          # backpropagate thru network 
          loss.backward()
          nn.utils.clip_grad_value_(model.parameters(), clip_value=gradient_clipping)
          opt.step()
      
      # get validation losses
      val_loss,val_acc,val_recon=validation_loss(model,Xtest,Ytest,batch_val,1)    
      #val_loss,val_recon=validation_loss_regression(model,Xtest,Ytest,batch_val,1)    
      
      
      print(f'Epoch [{epoch}/{num_epochs}], Val. Loss {val_loss:.2f}, Train Loss {total_loss:.2f}, Val. Acc {val_acc*100:.2f}, Train Acc {accuracy*100:.2f}')
      #print(f'Epoch [{epoch}/{num_epochs}], Val. Loss {val_loss:.4f}, Train Loss {total_loss:.4f}')
      
      if val_loss<goat_loss:
          goat_loss = val_loss
          goat_acc = val_acc*100      
          counter = 0
          print('Goat loss, saving model')      
          torch.save(model.state_dict(), filename)
      else:
          counter += 1
    
      if counter>=patience:
          print('Early stoppping point reached')
          print('Best val loss and val acc  are')
          print(goat_loss,goat_acc)
          break
    model_goat = iAutoencoder(input_size,hidden_size,latent_dims,num_classes)  
    #model_goat = iAutoencoder_B3(input_size,hidden_size,latent_dims,num_classes)
    model_goat.load_state_dict(torch.load(filename))
    model_goat=model_goat.to(device)
    model_goat.eval()
    return model_goat, goat_acc

# plotting and returning latent activations
def plot_latent(model, data, Y, num_samples,dim):
    # randomly sample the number of samples 
    #idx1 = rnd.choice(data.shape[0],num_samples)
    data=torch.from_numpy(data).to(device).float()
    Y=torch.from_numpy(Y).to(device).float()
    Y=convert_to_ClassNumbers(Y)
    y=Y#y=Y[idx1]    
    z = data#[idx1,:]
    model.eval()
    z = model.encoder(z)
    z = z.to('cpu').detach().numpy()
    y = y.to('cpu').detach().numpy()        
    #D = sil(z,y)
    D=0
    # scale between 0 and 1
    #z=  (z-np.min(z))/(np.max(z)-np.min(z))    
    fig=plt.figure()
    if dim==3:
        ax=plt.axes(projection="3d")
        p=ax.scatter3D(z[:, 0], z[:, 1],z[:,2], c=y, cmap='tab10')        
        plt.colorbar(p)
    if dim==2:
        ax=plt.axes        
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        plt.colorbar()
    model.train()    
    return D,z,y,fig

# plotting and returning latent activations and give acuracy
def plot_latent_acc(model, data, Y,dim):
    num_samples = data.shape[0]
    data=torch.from_numpy(data).to(device).float()
    Y=torch.from_numpy(Y).to(device).float()
    
    model.eval()
    
    z = model.encoder(data)
    recon,decodes = model(data)    
    ylabels = convert_to_ClassNumbers(Y)        
    ypred_labels = convert_to_ClassNumbers(decodes)     
    accuracy = (torch.sum(ylabels == ypred_labels).item())/ylabels.shape[0]
    
    y=ylabels    
    z = z.to('cpu').detach().numpy()
    y = y.to('cpu').detach().numpy()        
    D = sil(z,y)
        
    fig=plt.figure()
    if dim>2:
        ax=plt.axes(projection="3d")
        p=ax.scatter3D(z[:, 0], z[:, 1],z[:,2], c=y, cmap='tab10')        
        plt.colorbar(p)
    if dim==2:
        ax=plt.axes        
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        plt.colorbar()
    model.train()    
    return D,z,y,fig,accuracy


# plotting and returning latent activations and give acuracy
def plot_latent_select(model, data, Y,dim,ch):
    num_samples = data.shape[0]
    data=torch.from_numpy(data).to(device).float()
    Y=torch.from_numpy(Y).to(device).float()
    
    model.eval()
    
    z = model.encoder(data)
    recon,decodes = model(data)    
    ylabels = convert_to_ClassNumbers(Y)        
    ypred_labels = convert_to_ClassNumbers(decodes)    
        
    y=ylabels    
    z = z.to('cpu').detach().numpy()
    y = y.to('cpu').detach().numpy()   
    idx=np.array([],dtype=int)
    for i in np.arange(len(ch)):
        idx = np.append(idx,np.where(y==ch[i])[0])
               
    z = z[idx,:]
    y = y[idx]    
        
    fig=plt.figure()
    if dim>2:
        ax=plt.axes(projection="3d")
        p=ax.scatter3D(z[:, 0], z[:, 1],z[:,2], c=y, cmap='tab10')        
        plt.colorbar(p)
    if dim==2:
        ax=plt.axes        
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        plt.colorbar()
    model.train()    
    return fig


def return_recon(model,data,Y,num_classes):
    data = torch.from_numpy(data).to(device).float()
    Y=torch.from_numpy(Y).to(device).float()
    Y=convert_to_ClassNumbers(Y).to('cpu').detach().numpy()
    model.eval()
    hg_recon = []
    beta_recon=[]
    delta_recon=[]
    for query in np.arange(num_classes):
        idx = np.where(Y==query)[0]
        data_tmp = data[idx,:]        
        
        with torch.no_grad():
            recon_data,class_outputs = model(data_tmp)        
                
        recon_data = recon_data.to('cpu').detach().numpy()    
        # hg
        idx = np.arange(2,96,3)
        hg_recon_tmp = recon_data[:,idx]  
        hg_recon.append(hg_recon_tmp)
        # delta
        idx = np.arange(0,96,3)
        delta_recon_tmp = recon_data[:,idx]        
        delta_recon.append(delta_recon_tmp)
        #beta
        idx = np.arange(1,96,3)
        beta_recon_tmp = recon_data[:,idx]  
        beta_recon.append(beta_recon_tmp)
    
    model.train()
    return delta_recon,beta_recon,hg_recon

def return_recon_B2(model,data,Y):
    data = torch.from_numpy(data).to(device).float()
    Y=torch.from_numpy(Y).to(device).float()
    Y=convert_to_ClassNumbers(Y).to('cpu').detach().numpy()
    model.eval()
    hg_recon = []
    beta_recon=[]
    delta_recon=[]
    for query in np.arange(4):
        idx = np.where(Y==query)[0]
        data_tmp = data[idx,:]        
        
        with torch.no_grad():
            recon_data,class_outputs = model(data_tmp)        
                
        recon_data = recon_data.to('cpu').detach().numpy()    
        # hg
        idx = np.arange(2,96,3)
        hg_recon_tmp = recon_data[:,idx]  
        hg_recon.append(hg_recon_tmp)
        # delta
        idx = np.arange(0,96,3)
        delta_recon_tmp = recon_data[:,idx]        
        delta_recon.append(delta_recon_tmp)
        #beta
        idx = np.arange(1,96,3)
        beta_recon_tmp = recon_data[:,idx]  
        beta_recon.append(beta_recon_tmp)
    
    model.train()
    return delta_recon,beta_recon,hg_recon
    
def get_recon_channel_variances(recon_data):
    l = len(recon_data)
    variances = np.array([])
    for query in np.arange(l):
        tmp = recon_data[query]
        a = np.std(tmp,axis=0)
        variances = np.append(variances,a)
    
    return variances

def get_spatial_correlation(data1,data2,data3):
    corr_coef = []
    for query in np.arange(len(data1)):
        tmp1 = np.mean(data1[query],axis=0)
        tmp2 = np.mean(data2[query],axis=0)
        tmp3 = np.mean(data3[query],axis=0)
        a = np.corrcoef(tmp1,tmp2)[0,1]
        b = np.corrcoef(tmp1,tmp3)[0,1]
        c = np.corrcoef(tmp2,tmp3)[0,1]
        corr_coef.append([a,b,c])
    corr_coef = np.array(corr_coef).flatten()
    return corr_coef
    

def data_aug_mlp(indata,labels,data_size):
    N = (data_size/indata.shape[0]) #data aug factor
    labels_idx = np.argmax(labels,axis=1)
    num_labels = len(np.unique(labels_idx))
    condn_data_aug = []   
    labels_aug=[]
    for query in np.arange(num_labels):
        idx = np.where(labels_idx==query)[0]
        idx_len_aug = round(N*len(idx)) - len(idx)
        
        for i in np.arange(idx_len_aug):
            # randomly get 4 samples and average 
            a = rnd.choice(idx,5,replace=True)
            tmp_data = np.mean(indata[a,:],axis=0)
            b = 0.01 * rnd.randn(96)
            tmp_data = tmp_data + b
            tmp_data = tmp_data/lin.norm(tmp_data)
            condn_data_aug.append(tmp_data)
            labels_aug.append(labels[a,:][0,:])
    
    condn_data_aug = np.array(condn_data_aug)
    labels_aug = np.array(labels_aug)
    outdata = np.concatenate((indata,condn_data_aug),axis=0)
    outdata_labels = np.concatenate((labels,labels_aug),axis=0)
    return outdata, outdata_labels

def data_aug_mlp_chol(indata,labels,data_size):
    N = (data_size/indata.shape[0]) #data aug factor
    labels_idx = np.argmax(labels,axis=1)
    num_labels = len(np.unique(labels_idx))
    condn_data_aug = np.empty([0,96]) 
    labels_aug=np.empty([0,num_labels])
    # sample from random gaussian with known mean and cov matrix
    for query in np.arange(num_labels):
        idx = np.where(labels_idx==query)[0]
        idx_len_aug = round(N*len(idx)) - len(idx)
        tmp_data = indata[idx,:]
        C = np.cov(tmp_data,rowvar=False) + 1e-12*np.eye(tmp_data.shape[1])
        C12 = lin.cholesky(C)
        m = np.mean(tmp_data,axis=0)
        X = rnd.randn(idx_len_aug,96)
        new_data = X @ C12 + m
        condn_data_aug = np.concatenate((condn_data_aug,new_data),axis=0)    
        tmp_labels = np.zeros([idx_len_aug,num_labels])
        tmp_labels[:,query]=1        
        labels_aug = np.concatenate((labels_aug,tmp_labels))
       
    outdata = np.concatenate((indata,condn_data_aug),axis=0)
    outdata_labels = np.concatenate((labels,labels_aug),axis=0)
    return outdata, outdata_labels

def data_aug_mlp_chol_feature(indata,labels,data_size):
    N = (data_size/indata.shape[0]) #data aug factor
    labels_idx = np.argmax(labels,axis=1)
    num_labels = len(np.unique(labels_idx))
    condn_data_aug = np.empty([0,96]) 
    labels_aug=np.empty([0,7])
    # sample from random gaussian with known mean and cov matrix
    for query in np.arange(num_labels):
        idx = np.where(labels_idx==query)[0]
        idx_len_aug = round(N*len(idx)) - len(idx)
        tmp_data = indata[idx,:]
        
        # doing hg
        hg = tmp_data[:,np.arange(2,96,3)]
        C = np.cov(hg,rowvar=False)
        if lin.matrix_rank(C)<32:
            C = np.cov(hg,rowvar=False) +  1e-12*np.eye(hg.shape[1])
        C12 = lin.cholesky(C)
        m = np.mean(hg,axis=0)
        X = rnd.randn(idx_len_aug,hg.shape[1])
        new_hg = X @ C12 + m
        
        # doing delta
        delta = tmp_data[:,np.arange(0,96,3)]
        C = np.cov(delta,rowvar=False)
        if lin.matrix_rank(C)<32:
            C = np.cov(delta,rowvar=False) +  1e-12*np.eye(delta.shape[1])
        C12 = lin.cholesky(C)
        m = np.mean(delta,axis=0)
        X = rnd.randn(idx_len_aug,delta.shape[1])
        new_delta = X @ C12 + m
        
        # doing beta
        beta = tmp_data[:,np.arange(1,96,3)]
        C = np.cov(beta,rowvar=False)
        if lin.matrix_rank(C)<32:
            C = np.cov(beta,rowvar=False) +  1e-12*np.eye(beta.shape[1])
        C12 = lin.cholesky(C)
        m = np.mean(beta,axis=0)
        X = rnd.randn(idx_len_aug,beta.shape[1])
        new_beta = X @ C12 + m
        
        new_data = np.zeros([idx_len_aug,96])
        new_data[:,np.arange(0,96,3)] = new_delta
        new_data[:,np.arange(1,96,3)] = new_beta
        new_data[:,np.arange(2,96,3)] = new_hg
        
        condn_data_aug = np.concatenate((condn_data_aug,new_data),axis=0)    
        tmp_labels = np.zeros([idx_len_aug,7])
        tmp_labels[:,query]=1        
        labels_aug = np.concatenate((labels_aug,tmp_labels))
       
    outdata = np.concatenate((indata,condn_data_aug),axis=0)
    outdata_labels = np.concatenate((labels,labels_aug),axis=0)
    return outdata, outdata_labels


def data_aug_mlp_chol_feature_equalSize(indata,labels,data_size):    
    labels_idx = np.argmax(labels,axis=1)
    num_labels = len(np.unique(labels_idx))
    N = (data_size/num_labels) #data aug factor
    condn_data_aug = np.empty([0,96]) 
    labels_aug=np.empty([0,num_labels])
    # sample from random gaussian with known mean and cov matrix
    for query in np.arange(num_labels):
        idx = np.where(labels_idx==query)[0]
        idx_len_aug = round(N-len(idx))
        tmp_data = indata[idx,:]
        
        # doing hg
        hg = tmp_data[:,np.arange(2,96,3)]
        C = np.cov(hg,rowvar=False)
        if lin.matrix_rank(C)<32:
            C = np.cov(hg,rowvar=False) +  1e-12*np.eye(hg.shape[1])
        C12 = lin.cholesky(C)
        m = np.mean(hg,axis=0)
        X = rnd.randn(idx_len_aug,hg.shape[1])
        new_hg = X @ C12 + m
        
        # doing delta
        delta = tmp_data[:,np.arange(0,96,3)]
        C = np.cov(delta,rowvar=False)
        if lin.matrix_rank(C)<32:
            C = np.cov(delta,rowvar=False) +  1e-12*np.eye(delta.shape[1])
        C12 = lin.cholesky(C)
        m = np.mean(delta,axis=0)
        X = rnd.randn(idx_len_aug,delta.shape[1])
        new_delta = X @ C12 + m
        
        # doing beta
        beta = tmp_data[:,np.arange(1,96,3)]
        C = np.cov(beta,rowvar=False)
        if lin.matrix_rank(C)<32:
            C = np.cov(beta,rowvar=False) +  1e-12*np.eye(beta.shape[1])
        C12 = lin.cholesky(C)
        m = np.mean(beta,axis=0)
        X = rnd.randn(idx_len_aug,beta.shape[1])
        new_beta = X @ C12 + m
        
        new_data = np.zeros([idx_len_aug,96]) 
        new_data[:,np.arange(0,96,3)] = new_delta
        new_data[:,np.arange(1,96,3)] = new_beta
        new_data[:,np.arange(2,96,3)] = new_hg
        #add some noise
        new_data = new_data + 0.02*rnd.randn(new_data.shape[0],new_data.shape[1])
        
        # make it unit norm
        for i in np.arange(new_data.shape[0]):
            new_data[i,:] = new_data[i,:]/lin.norm(new_data[i,:])
        
        condn_data_aug = np.concatenate((condn_data_aug,new_data),axis=0)    
        tmp_labels = np.zeros([idx_len_aug,num_labels])
        tmp_labels[:,query]=1        
        labels_aug = np.concatenate((labels_aug,tmp_labels))
       
    outdata = np.concatenate((indata,condn_data_aug),axis=0)
    outdata_labels = np.concatenate((labels,labels_aug),axis=0)
    return outdata, outdata_labels

def data_aug_mlp_chol_feature_equalSize_B3_NoPooling(indata,labels,data_size):    
    labels_idx = np.argmax(labels,axis=1)
    num_labels = len(np.unique(labels_idx))
    N = (data_size/num_labels) #data aug factor
    feat_size = indata.shape[1]
    condn_data_aug = np.empty([0,feat_size]) 
    labels_aug=np.empty([0,num_labels])
    # sample from random gaussian with known mean and cov matrix
    for query in np.arange(num_labels):
        idx = np.where(labels_idx==query)[0]
        idx_len_aug = round(N-len(idx))
        tmp_data = indata[idx,:]
        
        # no pooling -> first 253 features are delta, 2nd 253 are beta and last are hG
        new_data = np.zeros([idx_len_aug,feat_size]) 
        for i in np.arange(0,indata.shape[1],253):
            hg = tmp_data[:,i:i+253] # just keeping the name hG
            C = np.cov(hg,rowvar=False)
            if lin.matrix_rank(C)<253:
                C = np.cov(hg,rowvar=False) +  1e-12*np.eye(hg.shape[1])
            C12 = lin.cholesky(C)
            m = np.mean(hg,axis=0)
            X = rnd.randn(idx_len_aug,hg.shape[1])
            new_hg = X @ C12 + m
            new_data[:,i:i+253] = new_hg
        
        #add some noise
        new_data = new_data + 0.02*rnd.randn(new_data.shape[0],new_data.shape[1])
        
        # make it unit norm
        for i in np.arange(new_data.shape[0]):
            new_data[i,:] = new_data[i,:]/lin.norm(new_data[i,:])
        
        condn_data_aug = np.concatenate((condn_data_aug,new_data),axis=0)    
        tmp_labels = np.zeros([idx_len_aug,num_labels])
        tmp_labels[:,query]=1        
        labels_aug = np.concatenate((labels_aug,tmp_labels))
       
    outdata = np.concatenate((indata,condn_data_aug),axis=0)
    outdata_labels = np.concatenate((labels,labels_aug),axis=0)
    return outdata, outdata_labels

def get_raw_channnel_variances(indata,labels):       
    idx = np.argmax(labels,axis=1)
    num_labels = len(np.unique(idx))
    hg_variances=[]
    delta_variances =[]
    beta_variances = []
    for query in np.arange(num_labels):
        
        idx1 = np.where(idx==query)[0]
        indata_tmp = indata[idx1,:]
            
        # hg
        idx2 = np.arange(2,96,3)
        hg = indata_tmp[:,idx2]      
        hg_variances_tmp = np.std(hg,axis=0)
        hg_variances.append(hg_variances_tmp)
        # delta
        idx2 = np.arange(0,96,3)
        delta = indata_tmp[:,idx2]            
        delta_variances_tmp = np.std(delta,axis=0)
        delta_variances.append(delta_variances_tmp)
        #beta
        idx2 = np.arange(1,96,3)
        beta = indata_tmp[:,idx2]  
        beta_variances_tmp = np.std(beta,axis=0)
        beta_variances.append(beta_variances_tmp)
    
    delta_variances = np.array(delta_variances).flatten()
    beta_variances = np.array(beta_variances).flatten()
    hg_variances = np.array(hg_variances).flatten()
    return delta_variances,beta_variances,hg_variances
    
    

### CENTERED LINEAR KERNAL ALIGNMENT TO COMPARE TWO FIXED LAYERED AUTOENCODERS
def linear_cka_dist(input_data,model,model1):
    # prelims 
    model.eval()
    model1.eval()
    elu=nn.ELU()
    input_data = torch.from_numpy(input_data).float().to(device)
    # getting the layers of first manifold
    layer = []
    layer.append(model.encoder.linear1.state_dict()['weight'])
    layer.append(model.encoder.linear2.state_dict()['weight'])
    layer.append(model.encoder.linear3.state_dict()['weight'])
    layer.append(model.decoder.linear1.state_dict()['weight'])
    layer.append(model.decoder.linear2.state_dict()['weight'])
    layer.append(model.decoder.linear3.state_dict()['weight'])
    bias = []
    bias.append(model.encoder.linear1.state_dict()['bias'])
    bias.append(model.encoder.linear2.state_dict()['bias'])
    bias.append(model.encoder.linear3.state_dict()['bias'])
    bias.append(model.decoder.linear1.state_dict()['bias'])
    bias.append(model.decoder.linear2.state_dict()['bias'])
    bias.append(model.decoder.linear3.state_dict()['bias'])
    # getting the layers of second manifold
    layer1 = []
    layer1.append(model1.encoder.linear1.state_dict()['weight'])
    layer1.append(model1.encoder.linear2.state_dict()['weight'])
    layer1.append(model1.encoder.linear3.state_dict()['weight'])
    layer1.append(model1.decoder.linear1.state_dict()['weight'])
    layer1.append(model1.decoder.linear2.state_dict()['weight'])
    layer1.append(model1.decoder.linear3.state_dict()['weight'])
    bias1 = []
    bias1.append(model1.encoder.linear1.state_dict()['bias'])
    bias1.append(model1.encoder.linear2.state_dict()['bias'])
    bias1.append(model1.encoder.linear3.state_dict()['bias'])
    bias1.append(model1.decoder.linear1.state_dict()['bias'])
    bias1.append(model1.decoder.linear2.state_dict()['bias'])
    bias1.append(model1.decoder.linear3.state_dict()['bias'])            
    
    # running the pairwise comparisons 
    D = np.zeros((6,6))
    for m in np.arange(len(layer)):
        out = input_data
        k=0
        while k<=m:                        
            w = torch.transpose(layer[k],0,1)
            b = bias[k]
            #shufle
            idx = torch.randperm(w.nelement())
            w = w.reshape(-1)[idx].reshape(w.size())
            idx = torch.randperm(b.nelement())
            b = b[idx]
            # compute
            out = torch.matmul(out, w) + b
            if k==2 or k==5:
                out=out
            else:
                out = elu(out)
            k=k+1
        
        for n in np.arange(len(layer1)):
            out1 = input_data
            k=0
            while k<=n:                        
                w = torch.transpose(layer1[k],0,1)
                b = bias1[k]
                #shufle
                idx = torch.randperm(w.nelement())
                w = w.reshape(-1)[idx].reshape(w.size())
                idx = torch.randperm(b.nelement())
                b = b[idx]
                # compute                        
                out1 = torch.matmul(out1, w) + b
                if k==2 or k==5:
                    out1=out1
                else:
                    out1 = elu(out1)
                k=k+1
            
            ### now get the CKA between out and out1
            X = out.to('cpu').detach().numpy()
            Y = out1.to('cpu').detach().numpy()
            X = X - np.mean(X,axis=0)
            Y = Y - np.mean(Y,axis=0)    
            ###cka
            a=lin.norm((X.T@X),'fro')
            b=lin.norm((Y.T@Y),'fro')
            c=(lin.norm((Y.T @ X),'fro'))**2
            d = c/(a*b)
            ### orthogonal procrustus 1
            #R, sca = op(X, Y)
            #d = (lin.norm(X@R - Y))/(lin.norm(Y))
            ### orthogonal procrustus 2
            # X = X/lin.norm(X,'fro')
            # Y = Y/lin.norm(Y,'fro')
            # a=lin.norm(X,'fro')**2
            # b=lin.norm(Y,'fro')**2
            # c=lin.norm(X.T@Y,'nuc')
            # d = a+b-2*c
            
            D[m,n] = d
            
    return(D)


def linear_cka_dist2(input_data,input_data1,model,model1):
    # prelims 
    model.eval()
    model1.eval()
    elu=nn.ELU()
    input_data = torch.from_numpy(input_data).float().to(device)
    input_data1 = torch.from_numpy(input_data1).float().to(device)
    # getting the layers of first manifold
    layer = []
    layer.append(model.encoder.linear1.state_dict()['weight'])
    layer.append(model.encoder.linear2.state_dict()['weight'])
    layer.append(model.encoder.linear3.state_dict()['weight'])
    layer.append(model.decoder.linear1.state_dict()['weight'])
    layer.append(model.decoder.linear2.state_dict()['weight'])
    layer.append(model.decoder.linear3.state_dict()['weight'])
    bias = []
    bias.append(model.encoder.linear1.state_dict()['bias'])
    bias.append(model.encoder.linear2.state_dict()['bias'])
    bias.append(model.encoder.linear3.state_dict()['bias'])
    bias.append(model.decoder.linear1.state_dict()['bias'])
    bias.append(model.decoder.linear2.state_dict()['bias'])
    bias.append(model.decoder.linear3.state_dict()['bias'])
    # getting the layers of second manifold
    layer1 = []
    layer1.append(model1.encoder.linear1.state_dict()['weight'])
    layer1.append(model1.encoder.linear2.state_dict()['weight'])
    layer1.append(model1.encoder.linear3.state_dict()['weight'])
    layer1.append(model1.decoder.linear1.state_dict()['weight'])
    layer1.append(model1.decoder.linear2.state_dict()['weight'])
    layer1.append(model1.decoder.linear3.state_dict()['weight'])
    bias1 = []
    bias1.append(model1.encoder.linear1.state_dict()['bias'])
    bias1.append(model1.encoder.linear2.state_dict()['bias'])
    bias1.append(model1.encoder.linear3.state_dict()['bias'])
    bias1.append(model1.decoder.linear1.state_dict()['bias'])
    bias1.append(model1.decoder.linear2.state_dict()['bias'])
    bias1.append(model1.decoder.linear3.state_dict()['bias'])
    
    # compute the activations
    # act_l1
    # act_l1
    # act_l1
    # act_l1
    # act_l1
    # act_l1
    
    # running the pairwise comparisons 
    D = np.zeros((6,6))
    for m in np.arange(len(layer)):
        out = input_data
        k=0
        while k<=m:                        
            w = torch.transpose(layer[k],0,1)
            b = bias[k]
            out = torch.matmul(out, w) + b
            if k==2 or k==5:
                out=out
            else:
                out = elu(out)
            k=k+1
        
        for n in np.arange(len(layer1)):
            out1 = input_data1
            k=0
            while k<=n:                        
                w = torch.transpose(layer1[k],0,1)
                b = bias1[k]
                out1 = torch.matmul(out1, w) + b
                if k==2 or k==5:
                    out1=out1
                else:
                    out1 = elu(out1)
                k=k+1
            
            ### now get the CKA between out and out1
            X = out.to('cpu').detach().numpy()
            Y = out1.to('cpu').detach().numpy()
            X = X - np.mean(X,axis=0)
            Y = Y - np.mean(Y,axis=0)    
            ###cka
            a=lin.norm((X.T@X),'fro')
            b=lin.norm((Y.T@Y),'fro')
            c=(lin.norm((Y.T @ X),'fro'))**2
            d = c/(a*b)
            ### orthogonal procrustus 1
            #R, sca = op(X, Y)
            #d = (lin.norm(X@R - Y))/(lin.norm(Y))
            ### orthogonal procrustus 2
            # X = X/lin.norm(X,'fro')
            # Y = Y/lin.norm(Y,'fro')
            # a=lin.norm(X,'fro')**2
            # b=lin.norm(Y,'fro')**2
            # c=lin.norm(X.T@Y,'nuc')
            # d = a+b-2*c
            
            D[m,n] = d
            
    return(D)



 
    

    
    
    
    
    
    
