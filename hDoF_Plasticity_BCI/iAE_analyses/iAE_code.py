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


# model params
input_size=96
hidden_size=32
latent_dims=2
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

# init variables across days 
dist_means_overall_imag = np.empty([10,0])
dist_var_overall_imag = np.empty([10,0])
mahab_dist_overall_imag = np.empty([10,0])
dist_means_overall_online = np.empty([10,0])
dist_var_overall_online = np.empty([10,0])
mahab_dist_overall_online = np.empty([10,0])

# iterations to bootstrap
iterations = 20

# init overall variables 
mahab_distances_imagined_days = np.zeros([21,iterations,10])
mean_distances_imagined_days = np.zeros([21,iterations,10])
var_imagined_days = np.zeros([7,iterations,10])
mahab_distances_online_days = np.zeros([21,iterations,10])
mean_distances_online_days = np.zeros([21,iterations,10])
var_online_days = np.zeros([7,iterations,10])
silhoutte_imagined_days = np.zeros((iterations,10))
silhoutte_online_days = np.zeros((iterations,10))
accuracy_imagined_days = np.zeros((iterations,10))
accuracy_online_days = np.zeros((iterations,10))

# main loop 
for days in (np.arange(10)+1):
    imagined_file_name = root_path + root_imag_filename +  str(days) + '.mat'
    condn_data_imagined,Yimagined = get_data(imagined_file_name)
    online_file_name = root_path + root_online_filename +  str(days) + '.mat'
    condn_data_online,Yonline = get_data(online_file_name)
    nn_filename = 'iAE_' + str(days) + '.pth'
    
    # init vars
    mahab_distances_imagined_iter = np.empty([21,0])
    mean_distances_imagined_iter = np.empty([21,0])
    var_distances_imagined_iter = np.empty([7,0])
    mahab_distances_online_iter = np.empty([21,0])
    mean_distances_online_iter = np.empty([21,0])
    var_distances_online_iter = np.empty([7,0])
    silhoutte_imagined_iter = np.array([])
    silhoutte_online_iter = np.array([])
    accuracy_imagined_iter = np.array([])
    accuracy_online_iter = np.array([])
    
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
        # get latent activity and plot
        D,z,idx = plot_latent(model, condn_data_imagined,Yimagined,condn_data_imagined.shape[0],
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
        # get latent activity and plot
        del D,z,idx
        D,z,idx = plot_latent(model, condn_data_online,Yonline,condn_data_online.shape[0],
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
    
    # store it all 
    mahab_distances_imagined_days[:,:,days-1] = mahab_distances_imagined_iter
    mean_distances_imagined_days[:,:,days-1] = mean_distances_imagined_iter
    var_imagined_days[:,:,days-1] = var_distances_imagined_iter
    mahab_distances_online_days[:,:,days-1] = mahab_distances_online_iter
    mean_distances_online_days[:,:,days-1] = mean_distances_online_iter
    var_online_days[:,:,days-1] = var_distances_online_iter
    silhoutte_imagined_days[:,days-1] = silhoutte_imagined_iter.T
    silhoutte_online_days[:,days-1] = silhoutte_online_iter.T
    accuracy_imagined_days[:,days-1] = accuracy_imagined_iter.T
    accuracy_online_days[:,days-1] = accuracy_online_iter.T
      

# saving it all 
np.savez('whole_dataSamples_stats_results', 
         silhoutte_imagined_days = silhoutte_imagined_days,
         silhoutte_online_days = silhoutte_online_days,
         var_online_days = var_online_days,
         mean_distances_online_days = mean_distances_online_days,
         mahab_distances_online_days = mahab_distances_online_days,
         var_imagined_days = var_imagined_days,
         mean_distances_imagined_days = mean_distances_imagined_days,
         mahab_distances_imagined_days = mahab_distances_imagined_days,
         accuracy_imagined_days = accuracy_imagined_days,
         accuracy_online_days = accuracy_online_days)

    
# loading it back
data=np.load('whole_dataSamples_stats_results.npz')
silhoutte_imagined_days = data.get('silhoutte_imagined_days')
silhoutte_online_days = data.get('silhoutte_online_days')
var_online_days = data.get('var_online_days')
mean_distances_online_days = data.get('mean_distances_online_days')
mahab_distances_online_days = data.get('mahab_distances_online_days')
var_imagined_days = data.get('var_imagined_days')
mean_distances_imagined_days = data.get('mean_distances_imagined_days')
mahab_distances_imagined_days = data.get('mahab_distances_imagined_days')
accuracy_imagined_days = data.get('accuracy_imagined_days')
accuracy_online_days = data.get('accuracy_online_days')

plt.figure()
plt.boxplot(accuracy_online_days)
plt.ylim((20,100))
plt.figure()
plt.boxplot(accuracy_imagined_days)
plt.ylim((20,100))
plt.show()

plt.figure()
tmp = gaussian_filter1d(np.mean(accuracy_imagined_days,axis=0),sigma=1)
plt.plot(tmp,color="black")
plt.ylim((20,100))
tmp = gaussian_filter1d(np.mean(accuracy_online_days,axis=0),sigma=1)
plt.plot(tmp,color="blue")
plt.ylim((20,100))
plt.show()

plt.figure()
tmp = gaussian_filter1d(np.mean(silhoutte_imagined_days,axis=0),sigma=1)
plt.plot(tmp,color="black")
tmp = gaussian_filter1d(np.mean(silhoutte_online_days,axis=0),sigma=1)
plt.plot(tmp,color="blue")
plt.show()

plt.figure()
tmp = gaussian_filter1d(np.mean(silhoutte_imagined_days,axis=0),sigma=1)
plt.plot(tmp,color="black")
tmp = gaussian_filter1d(np.mean(silhoutte_online_days,axis=0),sigma=1)
plt.plot(tmp,color="blue")
plt.show()


sigma = 1
plt.figure()
tmp_main = np.squeeze(np.median(mahab_distances_imagined_days,1))
tmp = np.median(tmp_main,axis=0)
tmp1 = scipy.stats.median_abs_deviation(tmp_main,axis=0)/np.sqrt(21)
tmp = gaussian_filter1d(tmp, sigma=sigma)
tmp1 = gaussian_filter1d(tmp1, sigma=sigma)
plt.plot(tmp,color="black",label = 'Imagined')
plt.plot(tmp+tmp1,color="black",linestyle="dotted")
plt.plot(tmp-tmp1,color="black",linestyle="dotted")
tmp_main = np.squeeze(np.median(mahab_distances_online_days,1))
tmp = np.median(tmp_main,axis=0)
tmp1 = scipy.stats.median_abs_deviation(tmp_main,axis=0)/np.sqrt(21)
tmp = gaussian_filter1d(tmp, sigma=sigma)
tmp1 = gaussian_filter1d(tmp1, sigma=sigma)
plt.plot(tmp,color="blue",label = 'Online')
plt.plot(tmp+tmp1,color="blue",linestyle="dotted")
plt.plot(tmp-tmp1,color="blue",linestyle="dotted")
plt.xlabel('Days')
plt.ylabel('Mahalnobis distance')
plt.legend()
plt.show()



    



         

    
    
    