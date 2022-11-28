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
import sklearn as skl
from sklearn.metrics import silhouette_score as sil
from sklearn.metrics import silhouette_samples as sil_samples
from tempfile import TemporaryFile
from scipy.ndimage import gaussian_filter1d
import scipy as scipy
import scipy.stats as stats
from iAE_utils_models import *
# setting up GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model params
input_size=96
hidden_size=48
latent_dims=2
num_classes = 4

# training params 
num_epochs=125
batch_size=32
learning_rate = 1e-3
batch_val=512
patience=6
gradient_clipping=10

# file location
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B2'
root_imag_filename = '\B2_condn_data_Imagined_Day'
root_online_filename = '\B2_condn_data_Online_Day'
root_batch_filename = '\B2_condn_data_Batch_Day'

# init variables across days 
num_days = 6
dist_means_overall_imag = np.empty([num_days,0])
dist_var_overall_imag = np.empty([num_days,0])
mahab_dist_overall_imag = np.empty([num_days,0])
dist_means_overall_online = np.empty([num_days,0])
dist_var_overall_online = np.empty([num_days,0])
mahab_dist_overall_online = np.empty([num_days,0])
dist_means_overall_batch = np.empty([num_days,0])
dist_var_overall_batch = np.empty([num_days,0])
mahab_dist_overall_batch = np.empty([num_days,0])

# iterations to bootstrap
iterations = 5

# init overall variables 
mahab_distances_imagined_days = np.zeros([6,iterations,num_days])
mean_distances_imagined_days = np.zeros([6,iterations,num_days])
var_imagined_days = np.zeros([4,iterations,num_days])
silhoutte_imagined_days = np.zeros((iterations,num_days))
accuracy_imagined_days = np.zeros((iterations,num_days))
mahab_distances_online_days = np.zeros([6,iterations,num_days])
mean_distances_online_days = np.zeros([6,iterations,num_days])
var_online_days = np.zeros([4,iterations,num_days])
silhoutte_online_days = np.zeros((iterations,num_days))
accuracy_online_days = np.zeros((iterations,num_days))
mahab_distances_batch_days = np.zeros([6,iterations,num_days])
mean_distances_batch_days = np.zeros([6,iterations,num_days])
var_batch_days = np.zeros([4,iterations,num_days])
silhoutte_batch_days = np.zeros((iterations,num_days))
accuracy_batch_days = np.zeros((iterations,num_days))
var_overall_imagined_days = np.zeros((iterations,num_days))
var_overall_batch_days = np.zeros((iterations,num_days))
var_overall_online_days = np.zeros((iterations,num_days))
# init vars for channel variance stuff
delta_recon_imag_var_days = np.zeros([iterations,128,num_days])
beta_recon_imag_var_days = np.zeros([iterations,128,num_days])
hg_recon_imag_var_days = np.zeros([iterations,128,num_days])
delta_recon_online_var_days = np.zeros([iterations,128,num_days])
beta_recon_online_var_days = np.zeros([iterations,128,num_days])
hg_recon_online_var_days = np.zeros([iterations,128,num_days])
delta_recon_batch_var_days = np.zeros([iterations,128,num_days])
beta_recon_batch_var_days = np.zeros([iterations,128,num_days])
hg_recon_batch_var_days = np.zeros([iterations,128,num_days])
# init vars for storing day to day spatial corr coeff
delta_spatial_corr_days = np.zeros([iterations,12,num_days])
beta_spatial_corr_days = np.zeros([iterations,12,num_days])
hg_spatial_corr_days = np.zeros([iterations,12,num_days])

# main loop 
for days in (np.arange(num_days)+1):
    imagined_file_name = root_path + root_imag_filename +  str(days) + '.mat'
    condn_data_imagined,Yimagined = get_data_B2(imagined_file_name)
    online_file_name = root_path + root_online_filename +  str(days) + '.mat'
    condn_data_online,Yonline = get_data_B2(online_file_name)
    batch_file_name = root_path + root_batch_filename +  str(days) + '.mat'
    if os.path.exists(batch_file_name):
        condn_data_batch,Ybatch = get_data_B2(batch_file_name)    
    nn_filename = 'iAE_' + str(days) + '.pth'
    
    # init vars
    mahab_distances_imagined_iter = np.empty([6,0])
    mean_distances_imagined_iter = np.empty([6,0])
    var_distances_imagined_iter = np.empty([4,0])
    silhoutte_imagined_iter = np.array([])
    accuracy_imagined_iter = np.array([])
    mahab_distances_online_iter = np.empty([6,0])
    mean_distances_online_iter = np.empty([6,0])
    var_distances_online_iter = np.empty([4,0])    
    silhoutte_online_iter = np.array([])        
    accuracy_online_iter = np.array([])
    mahab_distances_batch_iter = np.empty([6,0])
    mean_distances_batch_iter = np.empty([6,0])
    var_distances_batch_iter = np.empty([4,0])    
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
    # spatial correlation arrays
    delta_spatial_corr_iter = np.empty([12,0])
    beta_spatial_corr_iter = np.empty([12,0])
    hg_spatial_corr_iter = np.empty([12,0])
    
    #### DATA AUGMENTATION ###
    #condn_data_imagined,Yimagined = data_aug_mlp(condn_data_imagined,Yimagined,3000)
    condn_data_online,Yonline =   data_aug_mlp(condn_data_online,Yonline,condn_data_imagined.shape[0])
    if os.path.exists(batch_file_name):
        condn_data_batch,Ybatch =   data_aug_mlp(condn_data_batch,Ybatch,condn_data_imagined.shape[0])
        
    # plotting option
    plt_close=True
    
    # inner loop
    for loop in np.arange(iterations):
        
        #### DATA SPLIT OF ALL CONDITIONS FOR CROSS-VALIDATION ####
        condn_data_imagined_train,condn_data_imagined_test,Yimagined_train,Yimagined_test=training_test_split(condn_data_imagined, Yimagined, 0.8)
        condn_data_online_train,condn_data_online_test,Yonline_train,Yonline_test=training_test_split(condn_data_online, Yonline, 0.8)            
        if os.path.exists(batch_file_name):
            condn_data_batch_train,condn_data_batch_test,Ybatch_train,Ybatch_test=training_test_split(condn_data_batch, Ybatch, 0.8)
        else:
            condn_data_batch_train=np.empty([0,96])
            condn_data_batch_test=np.empty([0,96])
            Ybatch_train = np.empty([0,4])
            Ybatch_test = np.empty([0,4])
        # condn_data_imagined_train,condn_data_imagined_test = condn_data_imagined,condn_data_imagined
        # Yimagined_train,Yimagined_test = Yimagined,Yimagined
        # condn_data_online_train,condn_data_online_test = condn_data_online,condn_data_online
        # Yonline_train,Yonline_test = Yonline,Yonline
        # condn_data_batch_train,condn_data_batch_test = condn_data_batch,condn_data_batch
        # Ybatch_train,Ybatch_test = Ybatch,Ybatch
        
        
        #### STACK EVERYTHING TOGETHER ###
        condn_data_total = np.concatenate((condn_data_imagined_train,condn_data_online_train,condn_data_batch_train),axis=0)    
        Ytotal = np.concatenate((Yimagined_train,Yonline_train,Ybatch_train),axis=0)            
                
        #### TRAIN THE MODEL
        Ytest = np.zeros((2,2))
        while len(np.unique(np.argmax(Ytest,axis=1)))<4:
            Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_total,Ytotal,0.8)                        
            
        if 'model' in locals():
            del model    
        model = iAutoencoder(input_size,hidden_size,latent_dims,num_classes).to(device)        
        model,acc = training_loop_iAE(model,num_epochs,batch_size,learning_rate,batch_val,
                              patience,gradient_clipping,nn_filename,
                              Xtrain,Ytrain,Xtest,Ytest,
                              input_size,hidden_size,latent_dims,num_classes)
        
        #### GET IMAGINED DATA STATISTICS
        # store acc from latent space
        acc=0
        accuracy_imagined_iter = np.append(accuracy_imagined_iter,acc)      
        
        # get reconstructed activity as images in the three bands
        delta_recon_imag,beta_recon_imag,hg_recon_imag = return_recon_B2(model,
                                                    condn_data_imagined_test,Yimagined_test)
        # get the variance of each channel and storing 
        delta_imag_variances = get_recon_channel_variances(delta_recon_imag)
        beta_imag_variances = get_recon_channel_variances(beta_recon_imag)
        hg_imag_variances = get_recon_channel_variances(hg_recon_imag)
        delta_recon_imag_var_iter.append(delta_imag_variances)
        beta_recon_imag_var_iter.append(beta_imag_variances)
        hg_recon_imag_var_iter.append(hg_imag_variances)
        
        # get latent activity and plot
        D,z,idx,fig_imagined = plot_latent(model,condn_data_imagined_test,Yimagined_test,
                                           condn_data_imagined_test.shape[0],latent_dims)        
        silhoutte_imagined_iter = np.append(silhoutte_imagined_iter,D)
        if plt_close==True:
            plt.close()
        # mahab distance
        mahab_distances = get_mahab_distance_latent_B2(z,idx)
        mahab_distances = mahab_distances[np.triu_indices(mahab_distances.shape[0])]
        mahab_distances = mahab_distances[mahab_distances>0]
        mahab_distances_imagined_iter = np.append(mahab_distances_imagined_iter,
                                                  mahab_distances[:,None],axis=1)
        # euclidean distance between means
        dist_means = get_distance_means_B2(z,idx)
        mean_distances_imagined_iter = np.append(mean_distances_imagined_iter,
                                                  dist_means[:,None],axis=1)
        # variance of the data spread in latent space
        dist_var = get_variances_B2(z,idx)
        var_distances_imagined_iter = np.append(var_distances_imagined_iter,
                                                  dist_var[:,None],axis=1)
        
        # variance of overall data spread
        var_overall_imagined = get_variance_overall(z)
        var_overall_imagined_iter = np.append(var_overall_imagined_iter,
                                           var_overall_imagined)
        
        
        ###### GET ONLINE DATA STATISTICS ###        
         # store acc from latent space
        acc=0
        accuracy_online_iter = np.append(accuracy_online_iter,acc)      
        
        # get reconstructed activity as images in the three bands
        delta_recon_online,beta_recon_online,hg_recon_online = return_recon_B2(model,
                                                    condn_data_online_test,Yonline_test)
        # get the variance of each channel and storing 
        delta_online_variances = get_recon_channel_variances(delta_recon_online)
        beta_online_variances = get_recon_channel_variances(beta_recon_online)
        hg_online_variances = get_recon_channel_variances(hg_recon_online)
        delta_recon_online_var_iter.append(delta_online_variances)
        beta_recon_online_var_iter.append(beta_online_variances)
        hg_recon_online_var_iter.append(hg_online_variances)
       
        # get latent activity and plot
        del D,z,idx
        D,z,idx,fig_online = plot_latent(model, condn_data_online_test,Yonline_test,condn_data_online_test.shape[0],
                              latent_dims)
        if plt_close==True:
            plt.close()
        silhoutte_online_iter = np.append(silhoutte_online_iter,D)
        # mahab distance
        mahab_distances = get_mahab_distance_latent_B2(z,idx)
        mahab_distances = mahab_distances[np.triu_indices(mahab_distances.shape[0])]
        mahab_distances = mahab_distances[mahab_distances>0]
        mahab_distances_online_iter = np.append(mahab_distances_online_iter,
                                                  mahab_distances[:,None],axis=1)
        # euclidean distance between means
        dist_means = get_distance_means_B2(z,idx)
        mean_distances_online_iter = np.append(mean_distances_online_iter,
                                                  dist_means[:,None],axis=1)
        # variance of the data spread in latent space
        dist_var = get_variances_B2(z,idx)
        var_distances_online_iter = np.append(var_distances_online_iter,
                                                  dist_var[:,None],axis=1)
        # variance of overall data spread
        var_overall_online = get_variance_overall(z)
        var_overall_online_iter = np.append(var_overall_online_iter,
                                           var_overall_online)
        
        
        ##### RUN IT THROUGH BATCH DATA MEXT ######
        if os.path.exists(batch_file_name):
            acc=0
            accuracy_batch_iter = np.append(accuracy_batch_iter,acc)    
            
            # get reconstructed activity as images in the three bands
            delta_recon_batch,beta_recon_batch,hg_recon_batch = return_recon_B2(model,
                                                        condn_data_batch_test,Ybatch_test)
            # get the variance of each channel and storing 
            delta_batch_variances = get_recon_channel_variances(delta_recon_batch)
            beta_batch_variances = get_recon_channel_variances(beta_recon_batch)
            hg_batch_variances = get_recon_channel_variances(hg_recon_batch)
            delta_recon_batch_var_iter.append(delta_batch_variances)
            beta_recon_batch_var_iter.append(beta_batch_variances)
            hg_recon_batch_var_iter.append(hg_batch_variances)
            
            # get latent activity and plot
            del D,z,idx
            D,z,idx,fig_batch = plot_latent(model, condn_data_batch_test,Ybatch_test,condn_data_batch_test.shape[0],
                                  latent_dims)
            if plt_close==True:
                plt.close()
            silhoutte_batch_iter = np.append(silhoutte_batch_iter,D)
            # mahab distance
            mahab_distances = get_mahab_distance_latent_B2(z,idx)
            mahab_distances = mahab_distances[np.triu_indices(mahab_distances.shape[0])]
            mahab_distances = mahab_distances[mahab_distances>0]
            mahab_distances_batch_iter = np.append(mahab_distances_batch_iter,
                                                      mahab_distances[:,None],axis=1)
            # euclidean distance between means
            dist_means = get_distance_means_B2(z,idx)
            mean_distances_batch_iter = np.append(mean_distances_batch_iter,
                                                      dist_means[:,None],axis=1)
            # variance of the data spread in latent space
            dist_var = get_variances_B2(z,idx)
            var_distances_batch_iter = np.append(var_distances_batch_iter,
                                                      dist_var[:,None],axis=1)
            # variance of overall data spread
            var_overall_batch = get_variance_overall(z)
            var_overall_batch_iter = np.append(var_overall_batch_iter,
                                                var_overall_batch)
        else:
            acc=np.zeros((1))
            accuracy_batch_iter = np.append(accuracy_batch_iter,acc)    
            D=np.zeros((1))
            silhoutte_batch_iter = np.append(silhoutte_batch_iter,D)
            mahab_distances = np.zeros((6,))
            mahab_distances_batch_iter = np.append(mahab_distances_batch_iter,
                                                      mahab_distances[:,None],axis=1)
            dist_means = np.zeros((6,))
            mean_distances_batch_iter = np.append(mean_distances_batch_iter,
                                                      dist_means[:,None],axis=1)
            dist_var = np.zeros((4,))
            var_distances_batch_iter = np.append(var_distances_batch_iter,
                                                      dist_var[:,None],axis=1)
            
            var_overall_batch = 0
            var_overall_batch_iter = np.append(var_overall_batch_iter,
                                                var_overall_batch)      
            delta_batch_variances = np.zeros(128)
            beta_batch_variances = np.zeros(128)
            hg_batch_variances = np.zeros(128)
            delta_recon_batch_var_iter.append(delta_batch_variances)
            beta_recon_batch_var_iter.append(beta_batch_variances)
            hg_recon_batch_var_iter.append(hg_batch_variances)
        
        #### GET SPATIAL CORRELATIONS BETWEEN IMAGINED, ONLINE AND BATCH 
        if os.path.exists(batch_file_name):
            delta_corr_coef = get_spatial_correlation(delta_recon_imag,delta_recon_online,delta_recon_batch)
            beta_corr_coef = get_spatial_correlation(beta_recon_imag,beta_recon_online,beta_recon_batch)
            hg_corr_coef = get_spatial_correlation(hg_recon_imag,hg_recon_online,hg_recon_batch)            
        else:
            delta_corr_coef = np.zeros(12)
            beta_corr_coef = np.zeros(12)
            hg_corr_coef = np.zeros(12)
        
        delta_spatial_corr_iter = np.append(delta_spatial_corr_iter,delta_corr_coef[:,None],axis=1)
        beta_spatial_corr_iter = np.append(beta_spatial_corr_iter,beta_corr_coef[:,None],axis=1)
        hg_spatial_corr_iter = np.append(hg_spatial_corr_iter,hg_corr_coef[:,None],axis=1)
    
    # store it all 
    var_overall_imagined_days[:,days-1] = var_overall_imagined_iter.T
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
    var_overall_online_days[:,days-1] = var_overall_online_iter.T
    mahab_distances_batch_days[:,:,days-1] = mahab_distances_batch_iter
    mean_distances_batch_days[:,:,days-1] = mean_distances_batch_iter
    var_batch_days[:,:,days-1] = var_distances_batch_iter   
    silhoutte_batch_days[:,days-1] = silhoutte_batch_iter.T    
    accuracy_batch_days[:,days-1] = accuracy_batch_iter.T
    var_overall_batch_days[:,days-1] = var_overall_batch_iter.T
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
    # store spatial correlations (each day)
    delta_spatial_corr_days[:,:,days-1] = delta_spatial_corr_iter.T
    beta_spatial_corr_days[:,:,days-1] = beta_spatial_corr_iter.T
    hg_spatial_corr_days[:,:,days-1] = hg_spatial_corr_iter.T
     
    
# saving it all 
np.savez('NewB2_NoiseDataAugment_0.1_Stats_OnTrainingData_2D_common_Manifold_whole_dataSamples_stats_results_withBatch_Main_withVariance_AndChVars_AndSpatCorr', 
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
         hg_recon_batch_var_days=hg_recon_batch_var_days,
         delta_spatial_corr_days=delta_spatial_corr_days,
         beta_spatial_corr_days=beta_spatial_corr_days,
         hg_spatial_corr_days=hg_spatial_corr_days)

    








