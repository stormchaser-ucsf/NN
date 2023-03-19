# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:06:34 2023

@author: nikic
"""

"""
MAIN BLOCK FOR ANALYZING ACROSS-DAY MANIFOLD DRIFT. GOAL IS TO COMPARE NEURAL
NETWORK ACROSS DAYS I.E. THE AUTOENCODER. CONTRAST PAIRWISE THE MANIFOLDS
AFTER CROSS-PROJECTING DATA AND USING THE LINEAR CKA METRIC FOR EACH LAYER
"""

#%% SETTING THINGS UP
import os
os.chdir('C:/Users/nikic/Documents/GitHub/NN/hDoF_Plasticity_BCI/iAE_analyses')
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt
import math
import mat73
import numpy.random as rnd
import numpy.linalg as lin
plt.rcParams['figure.dpi'] = 200
from iAE_utils_models import *
import sklearn as skl
from sklearn.metrics import silhouette_score as sil
from sklearn.metrics import silhouette_samples as sil_samples
from tempfile import TemporaryFile
from scipy.ndimage import gaussian_filter1d
import scipy as scipy
import scipy.stats as stats
# GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# training params 
latent_dims=2
num_epochs=150
batch_size=32
learning_rate = 1e-3
batch_val=512
patience=5
gradient_clipping=10

# model params
input_size=96
hidden_size=48
latent_dims=2
num_classes = 7

# file location
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker'
root_imag_filename = '\condn_data_Imagined_Day'
root_online_filename = '\condn_data_Online_Day'
root_batch_filename = '\condn_data_Batch_Day'

#%% MAIN LOOP TO GET THE DATA

num_days=10
for i in np.arange(num_days)+1: #ROOT DAYS
    # load the data
    print('Processing Day ' + str(i) + ' data')
    imagined_file_name = root_path + root_imag_filename +  str(i) + '.mat'
    condn_data_imagined,Yimagined = get_data(imagined_file_name,num_classes)
    online_file_name = root_path + root_online_filename +  str(i) + '.mat'
    condn_data_online,Yonline = get_data(online_file_name,num_classes)
    batch_file_name = root_path + root_batch_filename +  str(i) + '.mat'
    condn_data_batch,Ybatch = get_data(batch_file_name,num_classes)    
    nn_filename = 'iAE_' + str(i) + '.pth'   
    
    # data augment
    condn_data_online,Yonline =   data_aug_mlp_chol_feature_equalSize(condn_data_online,Yonline,condn_data_imagined.shape[0])
    condn_data_batch,Ybatch =   data_aug_mlp_chol_feature_equalSize(condn_data_batch,Ybatch,condn_data_imagined.shape[0])
    
    # stack everything together 
    condn_data_total = np.concatenate((condn_data_imagined,condn_data_online,condn_data_batch),axis=0)    
    Ytotal = np.concatenate((Yimagined,Yonline,Ybatch),axis=0)     

    # demean
    #condn_data_total = condn_data_total - np.mean(condn_data_total,axis=0)                   
    
    #### train the model 
    Ytest = np.zeros((2,2))
    while len(np.unique(np.argmax(Ytest,axis=1)))<num_classes:
        Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_total,Ytotal,0.8)                        
        
    if 'model' in locals():
        del model    
    model = iAutoencoder(input_size,hidden_size,latent_dims,num_classes).to(device)        
    model,acc = training_loop_iAE(model,num_epochs,batch_size,learning_rate,batch_val,
                          patience,gradient_clipping,nn_filename,
                          Xtrain,Ytrain,Xtest,Ytest,
                          input_size,hidden_size,latent_dims,num_classes) 
    
    # how similar are the layers of this AE to itself?
    
    
    for j in np.arange(i+1,10): #COMPARISON TO OTHER DAYS
        # load the data
        print('Processing Day ' + str(j) + ' data')
        imagined_file_name = root_path + root_imag_filename +  str(j) + '.mat'
        condn_data_imagined1,Yimagined1 = get_data(imagined_file_name,num_classes)
        online_file_name = root_path + root_online_filename +  str(j) + '.mat'
        condn_data_online1,Yonline1 = get_data(online_file_name,num_classes)
        batch_file_name = root_path + root_batch_filename +  str(j) + '.mat'
        condn_data_batch1,Ybatch1 = get_data(batch_file_name,num_classes)    
        nn_filename = 'iAE_' + str(j) + '.pth'   
        
        # data augment
        condn_data_online1,Yonline1 =   data_aug_mlp_chol_feature_equalSize(condn_data_online1,Yonline1,condn_data_imagined1.shape[0])
        condn_data_batch1,Ybatch1 =   data_aug_mlp_chol_feature_equalSize(condn_data_batch1,Ybatch1,condn_data_imagined1.shape[0])
        
        # stack everything together 
        condn_data_total1 = np.concatenate((condn_data_imagined1,condn_data_online1,condn_data_batch1),axis=0)    
        Ytotal1 = np.concatenate((Yimagined1,Yonline1,Ybatch1),axis=0)   

        #de-mean                     
        #condn_data_total1 = condn_data_total1 - np.mean(condn_data_total1,axis=0)
        
        #### train the model 
        Ytest = np.zeros((2,2))
        while len(np.unique(np.argmax(Ytest,axis=1)))<num_classes:
            Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_total1,
                                                            Ytotal1,0.8)                                    
        
        if 'model1' in locals():
             del model1 
        model1 = iAutoencoder(input_size,hidden_size,latent_dims,num_classes).to(device)        
        model1,acc = training_loop_iAE(model1,num_epochs,batch_size,learning_rate,batch_val,
                              patience,gradient_clipping,nn_filename,
                              Xtrain,Ytrain,Xtest,Ytest,
                              input_size,hidden_size,latent_dims,num_classes)   
        
        
        # how similar are the layers of the root AE to this query AE?         
        d1 = linear_cka_dist2(condn_data_imagined,condn_data_imagined1,model,model1)
        d2 = linear_cka_dist2(condn_data_imagined,condn_data_imagined1,model,model1)
        
        # target by target         
        d1 = linear_cka_dist(condn_data_online,model,model1)
        d2 = linear_cka_dist(condn_data_online1,model,model1)
        
        # idx = np.argmax(Yimagined,axis=1)
        # idx = np.where(idx==3)[0]
        # A = condn_data_imagined[idx,:]
        # B = Yimagined[idx,:]
        plot_latent(model, condn_data_batch, Ybatch, condn_data_batch.shape[0], 2)
        plot_latent(model1, condn_data_batch, Ybatch, condn_data_batch.shape[0], 2)
        A = torch.from_numpy(condn_data_batch).float().to(device)
        z=model.encoder(A).to('cpu').detach().numpy()
        z1=model1.encoder(A).to('cpu').detach().numpy()
        
        d1 = linear_cka_dist(A,model,model1)
        d2 = linear_cka_dist(A,model,model1)
        
        # in1 = rnd.randn(2000,96)
        # in2 = rnd.randn(2000,96)
        # d1 = linear_cka_dist2(in1,in2,model,model1)
        # d2 = linear_cka_dist2(in2,in2,model,model1)
        
        d = (d1+d2)/2
        print(np.argmax(d1,axis=1))
        plt.imshow(d,cmap='magma')
        
        
        plot_latent(model, condn_data_online1, Yonline1, condn_data_online1.shape[0], 2)
        plot_latent(model1, condn_data_online1, Yonline1, condn_data_online1.shape[0], 2)
        
        
        
        
        
        
from scipy.linalg import orthogonal_procrustes as op
A = np.array([[ 2,  0,  1], [-2,  0,  0]])
R, sca = op(A, np.fliplr(A))
bhat = A@R
