# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 15:00:36 2022

@author: nikic
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 13:38:24 2022

@author: nikic
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

# file locations
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B2'
root_imag_filename = '\B2_condn_data_Imagined_Day'
root_online_filename = '\B2_condn_data_Online_Day'
root_batch_filename = '\B2_condn_data_Batch_Day'
# init vars for channel variance stuff
num_days=4
num_targets=4
num_features=32*num_targets
delta_recon_imag_var_days = np.zeros([num_features,num_days])
beta_recon_imag_var_days = np.zeros([num_features,num_days])
hg_recon_imag_var_days = np.zeros([num_features,num_days])
delta_recon_online_var_days = np.zeros([num_features,num_days])
beta_recon_online_var_days = np.zeros([num_features,num_days])
hg_recon_online_var_days = np.zeros([num_features,num_days])
delta_recon_batch_var_days = np.zeros([num_features,num_days])
beta_recon_batch_var_days = np.zeros([num_features,num_days])
hg_recon_batch_var_days = np.zeros([num_features,num_days])

# main loop to get the data 
for days in np.arange(2,6):
    print(days)
    imagined_file_name = root_path + root_imag_filename +  str(days) + '.mat'
    condn_data_imagined,Yimagined = get_data_B2(imagined_file_name)
    online_file_name = root_path + root_online_filename +  str(days) + '.mat'
    condn_data_online,Yonline = get_data_B2(online_file_name)
    batch_file_name = root_path + root_batch_filename +  str(days) + '.mat'
    condn_data_batch,Ybatch = get_data_B2(batch_file_name)  
    
    # get variances 
    delta_variances_imag,beta_variances_imag,hg_variances_imag = get_raw_channnel_variances(condn_data_imagined,Yimagined)
    delta_variances_online,beta_variances_online,hg_variances_online = get_raw_channnel_variances(condn_data_online,Yonline)
    delta_variances_batch,beta_variances_batch,hg_variances_batch = get_raw_channnel_variances(condn_data_batch,Ybatch)
    
    delta_recon_imag_var_days[:,days-2] = delta_variances_imag
    beta_recon_imag_var_days[:,days-2] = beta_variances_imag
    hg_recon_imag_var_days[:,days-2] = hg_variances_imag
    delta_recon_online_var_days[:,days-2] = delta_variances_online
    beta_recon_online_var_days[:,days-2] = beta_variances_online
    hg_recon_online_var_days[:,days-2] = hg_variances_online
    delta_recon_batch_var_days[:,days-2] = delta_variances_batch
    beta_recon_batch_var_days[:,days-2] = beta_variances_batch
    hg_recon_batch_var_days[:,days-2] = hg_variances_batch

# plot delta histograms
delta_recon_imag_var_days = np.mean(delta_recon_imag_var_days,axis=1)
delta_recon_online_var_days = np.mean(delta_recon_online_var_days,axis=1)
delta_recon_batch_var_days = np.mean(delta_recon_batch_var_days,axis=1)
x= [delta_recon_imag_var_days ,delta_recon_online_var_days ,
    delta_recon_batch_var_days]
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(x,showfliers=False)
plt.xticks(ticks=[1,2,3],labels=('Imagined','Online','Batch'),**hfont)
plt.ylabel('Delta Variance',**hfont)
plt.show()
# plotting variance histograms
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.hist(x[0],fc=(.2 ,.2,.2,.5),label='Open Loop')
plt.hist(x[1],fc=(.2 ,.2,.8,.5),label='Init Seed')
plt.hist(x[2],fc=(.8 ,.2,.2,.5),label='Batch')  
print(stats.ks_2samp(x[0],x[1]))
print(stats.ks_2samp(x[0],x[2]))
print(stats.ks_2samp(x[1],x[2]))

# plot beta histograms
beta_recon_imag_var_days = np.mean(beta_recon_imag_var_days,axis=1)
beta_recon_online_var_days = np.mean(beta_recon_online_var_days,axis=1)
beta_recon_batch_var_days = np.mean(beta_recon_batch_var_days,axis=1)
x= [beta_recon_imag_var_days.flatten() ,beta_recon_online_var_days.flatten() ,
    beta_recon_batch_var_days.flatten()]
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(x,showfliers=False)
plt.xticks(ticks=[1,2,3],labels=('Imagined','Online','Batch'),**hfont)
plt.ylabel('Beta Variance',**hfont)
plt.show()
# plotting variance histograms
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.hist(x[0],fc=(.2 ,.2,.2,.5),label='Open Loop')
plt.hist(x[1],fc=(.2 ,.2,.8,.5),label='Init Seed')
plt.hist(x[2],fc=(.8 ,.2,.2,.5),label='Batch')  
print(stats.ks_2samp(x[0],x[1]))
print(stats.ks_2samp(x[0],x[2]))
print(stats.ks_2samp(x[1],x[2]))

# plot hG histograms
hg_recon_imag_var_days = np.mean(hg_recon_imag_var_days,axis=1)
hg_recon_online_var_days = np.mean(hg_recon_online_var_days,axis=1)
hg_recon_batch_var_days = np.mean(hg_recon_batch_var_days,axis=1)
x= [hg_recon_imag_var_days.flatten() ,hg_recon_online_var_days.flatten() ,
    hg_recon_batch_var_days.flatten()]
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(x,showfliers=False)
plt.xticks(ticks=[1,2,3],labels=('Imagined','Online','Batch'),**hfont)
plt.ylabel('hG Variance',**hfont)
plt.show()
# plotting variance histograms
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.hist(x[0],fc=(.2 ,.2,.2,.5),label='Open Loop')
plt.hist(x[1],fc=(.2 ,.2,.8,.5),label='Init Seed')
plt.hist(x[2],fc=(.8 ,.2,.2,.5),label='Batch')  
print(stats.ks_2samp(x[0],x[1]))
print(stats.ks_2samp(x[0],x[2]))
print(stats.ks_2samp(x[1],x[2]))





    
    
       
    