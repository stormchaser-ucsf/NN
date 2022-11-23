# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 22:42:19 2022

@author: nikic

LOOP OVER DAYS, ANALYZE LATENT SPACE FOR IMAGINED AND ONLINE SEPERATELY.
AT SAMPLE LEVEL CROSS VALIDATE AND AVERAGE LATENT ACTIVITY AND INVESTIGATE 
STATISTICS

"""

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
import os
plt.rcParams['figure.dpi'] = 200
from iAE_utils_models import *
import sklearn as skl
from sklearn.metrics import silhouette_score as sil
from sklearn.metrics import silhouette_samples as sil_samples
from tempfile import TemporaryFile
from scipy.ndimage import gaussian_filter1d
import scipy as scipy
import scipy.stats as stats
# setting up GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model params
input_size=96
hidden_size=32
latent_dims=3
num_classes = 7

# training params 
num_epochs=200
batch_size=32
learning_rate = 1e-3
batch_val=512
patience=6
gradient_clipping=10

# file location
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker'
root_imag_filename = '\condn_data_Imagined_Day'
root_online_filename = '\condn_data_Online_Day'
root_batch_filename = '\condn_data_Batch_Day'

# init variables across days 
dist_means_overall_imag = np.empty([10,0])
dist_var_overall_imag = np.empty([10,0])
mahab_dist_overall_imag = np.empty([10,0])
dist_means_overall_online = np.empty([10,0])
dist_var_overall_online = np.empty([10,0])
mahab_dist_overall_online = np.empty([10,0])
dist_means_overall_batch = np.empty([10,0])
dist_var_overall_batch = np.empty([10,0])
mahab_dist_overall_batch = np.empty([10,0])

# num of days
num_days=10

# iterations to bootstrap
iterations = 50

# init overall variables 
mahab_distances_imagined_days = np.zeros([21,iterations,10])
mean_distances_imagined_days = np.zeros([21,iterations,10])
var_imagined_days = np.zeros([7,iterations,10])
silhoutte_imagined_days = np.zeros((iterations,10))
accuracy_imagined_days = np.zeros((iterations,10))
mahab_distances_online_days = np.zeros([21,iterations,10])
mean_distances_online_days = np.zeros([21,iterations,10])
var_online_days = np.zeros([7,iterations,10])
silhoutte_online_days = np.zeros((iterations,10))
accuracy_online_days = np.zeros((iterations,10))
mahab_distances_batch_days = np.zeros([21,iterations,10])
mean_distances_batch_days = np.zeros([21,iterations,10])
var_batch_days = np.zeros([7,iterations,10])
silhoutte_batch_days = np.zeros((iterations,10))
accuracy_batch_days = np.zeros((iterations,10))
var_overall_imagined_days = np.zeros((iterations,10))
var_overall_batch_days = np.zeros((iterations,10))
var_overall_online_days = np.zeros((iterations,10))
# init vars for channel variance stuff
delta_recon_imag_var_days = np.zeros([iterations,224,num_days])
beta_recon_imag_var_days = np.zeros([iterations,224,num_days])
hg_recon_imag_var_days = np.zeros([iterations,224,num_days])
delta_recon_online_var_days = np.zeros([iterations,224,num_days])
beta_recon_online_var_days = np.zeros([iterations,224,num_days])
hg_recon_online_var_days = np.zeros([iterations,224,num_days])
delta_recon_batch_var_days = np.zeros([iterations,224,num_days])
beta_recon_batch_var_days = np.zeros([iterations,224,num_days])
hg_recon_batch_var_days = np.zeros([iterations,224,num_days])


# main loop 
for days in (np.arange(10)+1):
    imagined_file_name = root_path + root_imag_filename +  str(days) + '.mat'
    condn_data_imagined,Yimagined = get_data(imagined_file_name)
    online_file_name = root_path + root_online_filename +  str(days) + '.mat'
    condn_data_online,Yonline = get_data(online_file_name)
    batch_file_name = root_path + root_batch_filename +  str(days) + '.mat'
    condn_data_batch,Ybatch = get_data(batch_file_name)    
    nn_filename = 'iAE_' + str(days) + '.pth'
    
    # init vars
    mahab_distances_imagined_iter = np.empty([21,0])
    mean_distances_imagined_iter = np.empty([21,0])
    var_distances_imagined_iter = np.empty([7,0])
    silhoutte_imagined_iter = np.array([])
    accuracy_imagined_iter = np.array([])
    mahab_distances_online_iter = np.empty([21,0])
    mean_distances_online_iter = np.empty([21,0])
    var_distances_online_iter = np.empty([7,0])    
    silhoutte_online_iter = np.array([])        
    accuracy_online_iter = np.array([])
    mahab_distances_batch_iter = np.empty([21,0])
    mean_distances_batch_iter = np.empty([21,0])
    var_distances_batch_iter = np.empty([7,0])    
    silhoutte_batch_iter = np.array([])    
    accuracy_batch_iter = np.array([])
    var_overall_imagined_iter = np.array([])
    var_overall_online_iter = np.array([])
    var_overall_batch_iter = np.array([])    
    # variances in reconstruction
    delta_recon_imag_var_iter = []
    beta_recon_imag_var_iter = []
    hg_recon_imag_var_iter = []
    delta_recon_online_var_iter = []
    beta_recon_online_var_iter = []
    hg_recon_online_var_iter = []
    delta_recon_batch_var_iter = []
    beta_recon_batch_var_iter = []
    hg_recon_batch_var_iter = []
    
    # inner loop
    for loop in np.arange(iterations):
        
        #### IMAGINED DATA FIRST #######        
        # train the model
        Ytest = np.zeros((2,2))
        while len(np.unique(np.argmax(Ytest,axis=1)))<7:
            Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_imagined,Yimagined,0.8)                        
            
        if 'model' in locals():
            del model    
        model = iAutoencoder(input_size,hidden_size,latent_dims,num_classes).to(device)        
        model,acc = training_loop_iAE(model,num_epochs,batch_size,learning_rate,batch_val,
                              patience,gradient_clipping,nn_filename,
                              Xtrain,Ytrain,Xtest,Ytest,
                              input_size,hidden_size,latent_dims,num_classes)
        # store acc from latent space
        accuracy_imagined_iter = np.append(accuracy_imagined_iter,acc)        
       
        # get reconstructed activity as images in the three bands
        delta_recon_imag,beta_recon_imag,hg_recon_imag = return_recon(model,
                                                    condn_data_imagined,Yimagined)
        # get the variance of each channel and storing 
        delta_imag_variances = get_recon_channel_variances(delta_recon_imag)
        beta_imag_variances = get_recon_channel_variances(beta_recon_imag)
        hg_imag_variances = get_recon_channel_variances(hg_recon_imag)
        delta_recon_imag_var_iter.append(delta_imag_variances)
        beta_recon_imag_var_iter.append(beta_imag_variances)
        hg_recon_imag_var_iter.append(hg_imag_variances)
        
        # get latent activity and plot
        D,z,idx,fig_imagined = plot_latent(model, condn_data_imagined,Yimagined,condn_data_imagined.shape[0],
                              latent_dims)        
        silhoutte_imagined_iter = np.append(silhoutte_imagined_iter,D)
        plt.close()
        # mahab distance
        mahab_distances = get_mahab_distance_latent(z,idx)
        mahab_distances = mahab_distances[np.triu_indices(mahab_distances.shape[0])]
        mahab_distances = mahab_distances[mahab_distances>0]
        mahab_distances_imagined_iter = np.append(mahab_distances_imagined_iter,
                                                  mahab_distances[:,None],axis=1)
        # euclidean distance between means
        dist_means = get_distance_means(z,idx)
        mean_distances_imagined_iter = np.append(mean_distances_imagined_iter,
                                                  dist_means[:,None],axis=1)
        # variance of the data spread in latent space
        dist_var = get_variances(z,idx)
        var_distances_imagined_iter = np.append(var_distances_imagined_iter,
                                                  dist_var[:,None],axis=1)
        
        # variance of overall data spread
        var_overall_imagined = get_variance_overall(z)
        var_overall_imagined_iter = np.append(var_overall_imagined_iter,
                                           var_overall_imagined)
        
        ###### RUN IT THROUGH ONLINE DATA MEXT ######
        del Xtrain,Xtest,Ytrain,Ytest
        Ytest = np.zeros((2,2))
        while len(np.unique(np.argmax(Ytest,axis=1)))<7:                     
            Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_online,Yonline,0.8)             
        model,acc = training_loop_iAE(model,num_epochs,batch_size,learning_rate,batch_val,
                              patience,gradient_clipping,nn_filename,
                              Xtrain,Ytrain,Xtest,Ytest,
                              input_size,hidden_size,latent_dims,num_classes)   
         # store acc from latent space
        accuracy_online_iter = np.append(accuracy_online_iter,acc)      
        
        # get reconstructed activity as images in the three bands
        delta_recon_online,beta_recon_online,hg_recon_online = return_recon(model,
                                                    condn_data_online,Yonline)
        # get the variance of each channel and storing 
        delta_online_variances = get_recon_channel_variances(delta_recon_online)
        beta_online_variances = get_recon_channel_variances(beta_recon_online)
        hg_online_variances = get_recon_channel_variances(hg_recon_online)
        delta_recon_online_var_iter.append(delta_online_variances)
        beta_recon_online_var_iter.append(beta_online_variances)
        hg_recon_online_var_iter.append(hg_online_variances)
       
        # get latent activity and plot
        del D,z,idx
        D,z,idx,fig_online = plot_latent(model, condn_data_online,Yonline,condn_data_online.shape[0],
                              latent_dims)
        plt.close()
        silhoutte_online_iter = np.append(silhoutte_online_iter,D)
        # mahab distance
        mahab_distances = get_mahab_distance_latent(z,idx)
        mahab_distances = mahab_distances[np.triu_indices(mahab_distances.shape[0])]
        mahab_distances = mahab_distances[mahab_distances>0]
        mahab_distances_online_iter = np.append(mahab_distances_online_iter,
                                                  mahab_distances[:,None],axis=1)
        # euclidean distance between means
        dist_means = get_distance_means(z,idx)
        mean_distances_online_iter = np.append(mean_distances_online_iter,
                                                  dist_means[:,None],axis=1)
        # variance of the data spread in latent space
        dist_var = get_variances(z,idx)
        var_distances_online_iter = np.append(var_distances_online_iter,
                                                  dist_var[:,None],axis=1)
        # variance of overall data spread
        var_overall_online = get_variance_overall(z)
        var_overall_online_iter = np.append(var_overall_online_iter,
                                           var_overall_online)
        
        ###### RUN IT THROUGH BATCH DATA MEXT ######
        del Xtrain,Xtest,Ytrain,Ytest
        Ytest = np.zeros((2,2))
        while len(np.unique(np.argmax(Ytest,axis=1)))<7:                     
            Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_batch,Ybatch,0.8)             
        model,acc = training_loop_iAE(model,num_epochs,batch_size,learning_rate,batch_val,
                              patience,gradient_clipping,nn_filename,
                              Xtrain,Ytrain,Xtest,Ytest,
                              input_size,hidden_size,latent_dims,num_classes)   
         # store acc from latent space
        accuracy_batch_iter = np.append(accuracy_batch_iter,acc)    
        
        # get reconstructed activity as images in the three bands
        delta_recon_batch,beta_recon_batch,hg_recon_batch = return_recon(model,
                                                    condn_data_batch,Ybatch)
        # get the variance of each channel and storing 
        delta_batch_variances = get_recon_channel_variances(delta_recon_batch)
        beta_batch_variances = get_recon_channel_variances(beta_recon_batch)
        hg_batch_variances = get_recon_channel_variances(hg_recon_batch)
        delta_recon_batch_var_iter.append(delta_batch_variances)
        beta_recon_batch_var_iter.append(beta_batch_variances)
        hg_recon_batch_var_iter.append(hg_batch_variances)
        
        # get latent activity and plot
        del D,z,idx
        D,z,idx,fig_batch = plot_latent(model, condn_data_batch,Ybatch,condn_data_batch.shape[0],
                              latent_dims)
        plt.close()
        silhoutte_batch_iter = np.append(silhoutte_batch_iter,D)
        # mahab distance
        mahab_distances = get_mahab_distance_latent(z,idx)
        mahab_distances = mahab_distances[np.triu_indices(mahab_distances.shape[0])]
        mahab_distances = mahab_distances[mahab_distances>0]
        mahab_distances_batch_iter = np.append(mahab_distances_batch_iter,
                                                  mahab_distances[:,None],axis=1)
        # euclidean distance between means
        dist_means = get_distance_means(z,idx)
        mean_distances_batch_iter = np.append(mean_distances_batch_iter,
                                                  dist_means[:,None],axis=1)
        # variance of the data spread in latent space
        dist_var = get_variances(z,idx)
        var_distances_batch_iter = np.append(var_distances_batch_iter,
                                                  dist_var[:,None],axis=1)
        # variance of overall data spread
        var_overall_batch = get_variance_overall(z)
        var_overall_batch_iter = np.append(var_overall_batch_iter,
                                           var_overall_batch)
        
    
    # store it all 
    mahab_distances_imagined_days[:,:,days-1] = mahab_distances_imagined_iter
    mean_distances_imagined_days[:,:,days-1] = mean_distances_imagined_iter
    var_imagined_days[:,:,days-1] = var_distances_imagined_iter
    silhoutte_imagined_days[:,days-1] = silhoutte_imagined_iter.T
    accuracy_imagined_days[:,days-1] = accuracy_imagined_iter.T
    mahab_distances_online_days[:,:,days-1] = mahab_distances_online_iter
    mean_distances_online_days[:,:,days-1] = mean_distances_online_iter
    var_online_days[:,:,days-1] = var_distances_online_iter   
    silhoutte_online_days[:,days-1] = silhoutte_online_iter.T    
    accuracy_online_days[:,days-1] = accuracy_online_iter.T
    mahab_distances_batch_days[:,:,days-1] = mahab_distances_batch_iter
    mean_distances_batch_days[:,:,days-1] = mean_distances_batch_iter
    var_batch_days[:,:,days-1] = var_distances_batch_iter   
    silhoutte_batch_days[:,days-1] = silhoutte_batch_iter.T    
    accuracy_batch_days[:,days-1] = accuracy_batch_iter.T
    var_overall_batch_days[:,days-1] = var_overall_batch_iter.T
    var_overall_online_days[:,days-1] = var_overall_online_iter.T
    var_overall_imagined_days[:,days-1] = var_overall_imagined_iter.T
    # store the channel variances from reconstruction
    delta_recon_imag_var_days[:,:,days-1] = np.array(delta_recon_imag_var_iter)
    beta_recon_imag_var_days[:,:,days-1] = np.array(beta_recon_imag_var_iter)
    hg_recon_imag_var_days[:,:,days-1] = np.array(hg_recon_imag_var_iter)
    delta_recon_online_var_days[:,:,days-1] = np.array(delta_recon_online_var_iter)
    beta_recon_online_var_days[:,:,days-1] = np.array(beta_recon_online_var_iter)
    hg_recon_online_var_days[:,:,days-1] = np.array(hg_recon_online_var_iter)
    delta_recon_batch_var_days[:,:,days-1] = np.array(delta_recon_batch_var_iter)
    beta_recon_batch_var_days[:,:,days-1] = np.array(beta_recon_batch_var_iter)
    hg_recon_batch_var_days[:,:,days-1] = np.array(hg_recon_batch_var_iter)
      


# saving it all 
# orig filename: whole_dataSamples_stats_results_withBatch_Main_withVariance
np.savez('whole_dataSamples_stats_results_withBatch_Main_withVariance_AndChVars', 
         silhoutte_imagined_days = silhoutte_imagined_days,
         silhoutte_online_days = silhoutte_online_days,
         silhoutte_batch_days = silhoutte_batch_days,
         var_online_days = var_online_days,
         mean_distances_online_days = mean_distances_online_days,
         mahab_distances_online_days = mahab_distances_online_days,
         var_imagined_days = var_imagined_days,
         mean_distances_imagined_days = mean_distances_imagined_days,
         mahab_distances_imagined_days = mahab_distances_imagined_days,
         var_batch_days = var_batch_days,
         mean_distances_batch_days = mean_distances_batch_days,
         mahab_distances_batch_days = mahab_distances_batch_days,
         accuracy_imagined_days = accuracy_imagined_days,
         accuracy_online_days = accuracy_online_days,
         accuracy_batch_days = accuracy_batch_days,
         var_overall_batch_days = var_overall_batch_days,
         var_overall_online_days = var_overall_online_days,
         var_overall_imagined_days = var_overall_imagined_days,
         delta_recon_imag_var_days=delta_recon_imag_var_days,
         beta_recon_imag_var_days=beta_recon_imag_var_days,
         hg_recon_imag_var_days=hg_recon_imag_var_days,
         delta_recon_online_var_days=delta_recon_online_var_days,
         beta_recon_online_var_days=beta_recon_online_var_days,
         hg_recon_online_var_days=hg_recon_online_var_days,
         delta_recon_batch_var_days=delta_recon_batch_var_days,
         beta_recon_batch_var_days=beta_recon_batch_var_days,
         hg_recon_batch_var_days=hg_recon_batch_var_days)

    








