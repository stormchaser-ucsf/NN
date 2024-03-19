# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 08:58:03 2022

@author: nikic
"""

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
import pandas as pd
from statsmodels.stats.anova import AnovaRM

#%%
# loading it back
#whole_dataSamples_stats_results_withBatch_Main_withVariance_AndChVars
#whole_dataSamples_stats_results_withBatch_Main_withVariance
#B2_whole_dataSamples_stats_results_withBatch_Main_withVariance
#whole_dataSamples_stats_results_withBatch_Main_withVariance_AndChVars

data=np.load('ProcessedData_B3__pt02_AllDays_2D_Main_v6.npz') #for B3
#data=np.load('ProcessedData_B1_9DoF.npz')
# neural variance from this
#data =np.load('MAIN_MAIN2_B1_NoiseDataAugment_CholIndivFeatEqualSize_pt01Noise_2D_common_Manifold_whole_dataSamples_stats_results_withBatch_Main_withVariance_AndChVars_AndSpatCorr.npz')
# mahab dist from this
#data=np.load('MAIN_NewB1_NoiseDataAugment_CholIndivFeatEqualSize_pt01Noise_Stats_HeldOut_2D_common_Manifold_whole_dataSamples_stats_results_withBatch_Main_withVariance_AndChVars_AndSpatCorr.npz')
#data=np.load('MAIN_MAIN_NewB2_NoiseDataAugmentCholEqualFeat_pt02_Stats_OnAllData_2D_common_Manifold_withBatch_Main_withVariance_AndChVars_AndSpatCorr.npz')
#data=np.load('ProcessedData_B1_01142023.npz')
silhoutte_imagined_days = data.get('silhoutte_imagined_days')
silhoutte_online_days = data.get('silhoutte_online_days')
silhoutte_batch_days = data.get('silhoutte_batch_days')
var_online_days = data.get('var_online_days')
mean_distances_online_days = data.get('mean_distances_online_days')
mahab_distances_online_days = data.get('mahab_distances_online_days')
var_imagined_days = data.get('var_imagined_days')
mean_distances_imagined_days = data.get('mean_distances_imagined_days')
mahab_distances_imagined_days = data.get('mahab_distances_imagined_days')
accuracy_imagined_days = data.get('accuracy_imagined_days')
accuracy_online_days = data.get('accuracy_online_days')
var_overall_imagined_days = data.get('var_overall_imagined_days')
var_overall_online_days = data.get('var_overall_online_days')
var_overall_batch_days = data.get('var_overall_batch_days')
var_batch_days = data.get('var_batch_days')
mean_distances_batch_days = data.get('mean_distances_batch_days')
mahab_distances_batch_days = data.get('mahab_distances_batch_days')
accuracy_batch_days = data.get('accuracy_batch_days')
# channel variance stuff
delta_recon_imag_var_days=data.get('delta_recon_imag_var_days')
beta_recon_imag_var_days=data.get('beta_recon_imag_var_days')
hg_recon_imag_var_days=data.get('hg_recon_imag_var_days')
delta_recon_online_var_days=data.get('delta_recon_online_var_days')
beta_recon_online_var_days=data.get('beta_recon_online_var_days')
hg_recon_online_var_days=data.get('hg_recon_online_var_days')
delta_recon_batch_var_days=data.get('delta_recon_batch_var_days')
beta_recon_batch_var_days=data.get('beta_recon_batch_var_days')
hg_recon_batch_var_days=data.get('hg_recon_batch_var_days')
# spatial correlation stuff
delta_spatial_corr_days=data.get('delta_spatial_corr_days')
beta_spatial_corr_days=data.get('beta_spatial_corr_days')
hg_spatial_corr_days=data.get('hg_spatial_corr_days')


#%%
# plotting latent spaces 3D
az=-45
el=33
x1 = np.array(fig_imagined.axes[0].get_xlim())[:,None]
x2 = np.array(fig_online.axes[0].get_xlim())[:,None]
x3 = np.array(fig_batch.axes[0].get_xlim())[:,None]
x = np.concatenate((x1,x2,x3),axis=1)
xmin = x.min()
xmax = x.max()
y1 = np.array(fig_imagined.axes[0].get_ylim())[:,None]
y2 = np.array(fig_online.axes[0].get_ylim())[:,None]
y3 = np.array(fig_batch.axes[0].get_ylim())[:,None]
y = np.concatenate((y1,y2,y3),axis=1)
ymin = y.min()
ymax = y.max()
z1 = np.array(fig_imagined.axes[0].get_zlim())[:,None]
z2 = np.array(fig_online.axes[0].get_zlim())[:,None]
z3 = np.array(fig_batch.axes[0].get_zlim())[:,None]
z = np.concatenate((z1,z2,z3),axis=1)
zmin = z.min()
zmax = z.max()
fig_online.axes[0].set_xlim(xmin,xmax)
fig_batch.axes[0].set_xlim(xmin,xmax)
fig_imagined.axes[0].set_xlim(xmin,xmax)
fig_online.axes[0].set_ylim(ymin,ymax)
fig_batch.axes[0].set_ylim(ymin,ymax)
fig_imagined.axes[0].set_ylim(ymin,ymax)
fig_online.axes[0].set_zlim(zmin,zmax)
fig_batch.axes[0].set_zlim(zmin,zmax)
fig_imagined.axes[0].set_zlim(zmin,zmax)
fig_batch.axes[0].view_init(elev=el, azim=az)
fig_imagined.axes[0].view_init(elev=el, azim=az)
fig_online.axes[0].view_init(elev=el, azim=az)
image_format = 'png' # e.g .png, .svg, etc.
image_name = 'Latent_Day1_OL_B3_ComMan2d.png'
fig_imagined.savefig(image_name, format=image_format, dpi=300)
image_name = 'Latent_Day1_CL1_B3_ComMan2d.png'
fig_online.savefig(image_name, format=image_format, dpi=300)
image_name = 'Latent_Day1_CL2_B3_ComMan2d.png'
fig_batch.savefig(image_name, format=image_format, dpi=300)
fig_imagined.suptitle('OL')


# plotting latent spaces 2D
x1 = np.array(fig_imagined.axes[0].get_xlim())[:,None]
x2 = np.array(fig_online.axes[0].get_xlim())[:,None]
x3 = np.array(fig_batch.axes[0].get_xlim())[:,None]
x = np.concatenate((x1,x2,x3),axis=1)
xmin = x.min()
xmax = x.max()
y1 = np.array(fig_imagined.axes[0].get_ylim())[:,None]
y2 = np.array(fig_online.axes[0].get_ylim())[:,None]
y3 = np.array(fig_batch.axes[0].get_ylim())[:,None]
y = np.concatenate((y1,y2,y3),axis=1)
ymin = y.min()
ymax = y.max()
fig_online.axes[0].set_xlim(xmin,xmax)
fig_batch.axes[0].set_xlim(xmin,xmax)
fig_imagined.axes[0].set_xlim(xmin,xmax)
fig_online.axes[0].set_ylim(ymin,ymax)
fig_batch.axes[0].set_ylim(ymin,ymax)
fig_imagined.axes[0].set_ylim(ymin,ymax)
fig_online.axes[0].set_xticks(ticks=np.arange(-20,31,10))
fig_online.axes[0].set_yticks(ticks=np.arange(-15,21,5))
fig_imagined.axes[0].set_xticks(ticks=np.arange(-20,31,10))
fig_imagined.axes[0].set_yticks(ticks=np.arange(-15,21,5))
fig_batch.axes[0].set_xticks(ticks=np.arange(-20,31,10))
fig_batch.axes[0].set_yticks(ticks=np.arange(-15,21,5))
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'LatentCKD_Day1_OL_B1_ComMan2d.svg'
fig_imagined.savefig(image_name, format=image_format, dpi=300)
image_name = 'LatentCKD_Day1_CL1_B1_ComMan2d.svg'
fig_online.savefig(image_name, format=image_format, dpi=300)
image_name = 'LatentCKD_Day1_CL2_B1_ComMan2d.svg'
fig_batch.savefig(image_name, format=image_format, dpi=300)
fig_imagined.suptitle('OL')


#%% plotting overall variances over days (MAIN MAIN)
N=1
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(11)+1
X=np.arange(11)+1
# imagined 
tmp1 = np.median(var_overall_imagined_days,axis=0)
tmp_boot, tmp_boot_std = median_bootstrap(var_overall_imagined_days,1000)
tmp1b = tmp_boot_std/1#sqrt(tmp_main.shape[0])
tmp1 = np.insert(tmp1,0,tmp1[0],axis=0)
tmp1b = np.insert(tmp1b,0,tmp1b[0],axis=0)
tmp1 = np.convolve(tmp1, np.ones(N)/N, mode='same')
tmp1 = tmp1[1:]
tmp1b = tmp1b[1:]
plt.plot(X,tmp1,color="black",label = 'Imagined')
plt.fill_between(X, tmp1-tmp1b, tmp1+tmp1b,color="black",alpha=0.2)
# online
tmp2 = np.median(var_overall_online_days,axis=0)
tmp_boot, tmp_boot_std = median_bootstrap(var_overall_online_days,1000)
tmp2b = tmp_boot_std/1#sqrt(tmp_main.shape[0])
tmp2 = np.insert(tmp2,0,tmp2[0],axis=0)
tmp2b = np.insert(tmp2b,0,tmp2b[0],axis=0)
tmp2 = np.convolve(tmp2, np.ones(N)/N, mode='same')
tmp2b = np.convolve(tmp2b, np.ones(N)/N, mode='same')
tmp2 = tmp2[1:]
tmp2b = tmp2b[1:]
plt.plot(X,tmp2,color="blue",label = 'Online')
plt.fill_between(X, tmp2-tmp2b, tmp2+tmp2b,color="blue",alpha=0.2)
# batch
tmp3 = np.median(var_overall_batch_days,axis=0)
tmp_boot, tmp_boot_std = median_bootstrap(var_overall_batch_days,1000)
tmp3b = tmp_boot_std/1#sqrt(tmp_main.shape[0])
tmp3 = np.insert(tmp3,0,tmp3[0],axis=0)
tmp3b = np.insert(tmp3b,0,tmp3b[0],axis=0)
tmp3 = np.convolve(tmp3, np.ones(N)/N, mode='same')
tmp3b = np.convolve(tmp3b, np.ones(N)/N, mode='same')
tmp3 = tmp3[1:]
tmp3b = tmp3b[1:]
plt.plot(X,tmp3,color="red",label = 'Batch')
plt.fill_between(X, tmp3-tmp3b, tmp3+tmp3b,color="red",alpha=0.2)
plt.legend()
plt.xlabel('Days',**hfont)
plt.ylabel('Overall Latent Variance',**hfont)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Overall_Variance_Latent_Days.svg'
fig.savefig(image_name, format=image_format, dpi=300)

tmp = np.concatenate((tmp1[:,None],tmp2[:,None],tmp3[:,None]),axis=1)
# tmp = np.concatenate((np.ndarray.flatten(var_overall_imagined_days)[:,None],
#                       np.ndarray.flatten(var_overall_online_days)[:,None],
#                       np.ndarray.flatten(var_overall_batch_days)[:,None]),axis=1)
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(tmp)
plt.xticks(ticks=[1,2,3],labels=('Imagined','Online','Batch'),**hfont)
plt.ylabel('Overall Latent Variance',**hfont)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Overall_Variance_Latent_Boxplot.svg'
fig.savefig(image_name, format=image_format, dpi=300)
print(np.mean(tmp,axis=0))
stats.ttest_rel(tmp[:,1],tmp[:,2])


# getting the average variance per day
tmp1 = np.squeeze(np.mean(mahab_distances_online_days,axis=0))
tmp2 = np.squeeze(np.mean(mahab_distances_imagined_days,axis=0))
tmp3 = np.squeeze(np.mean(mahab_distances_batch_days,axis=0))
a=np.ndarray.flatten(tmp1)[:,None]
b=np.ndarray.flatten(tmp2)[:,None]
c=np.ndarray.flatten(tmp3)[:,None]
d=np.concatenate((b,a,c),axis=1)


# plotting variances as boxplot 
tmp1 = np.squeeze(np.mean(var_online_days,axis=1))
tmp2 = np.squeeze(np.mean(var_imagined_days,axis=1))
tmp3 = np.squeeze(np.mean(var_batch_days,axis=1))
a=np.ndarray.flatten(tmp1)[:,None]
b=np.ndarray.flatten(tmp2)[:,None]
c=np.ndarray.flatten(tmp3)[:,None]
d=np.concatenate((b,a,c),axis=1)
plt.figure()
plt.boxplot(np.log(d))
d=np.concatenate((var_distances_batch_iter,var_distances_online_iter,
                  var_distances_imagined_iter),axis=1)

# plotting variances 
plt.figure()
tmp = np.squeeze(np.mean(var_imagined_days,axis=1))
tmp = np.mean(tmp,axis=0)
tmp = gaussian_filter1d(tmp,sigma=1)
plt.plot(tmp,label='imagined')
tmp1 = np.squeeze(np.mean(var_online_days,axis=1))
tmp1 = np.mean(tmp1,axis=0)
tmp1 = gaussian_filter1d(tmp1,sigma=1)
plt.plot(tmp1,label='online')
tmp2 = np.squeeze(np.mean(var_batch_days,axis=1))
tmp2 = np.mean(tmp2,axis=0)
tmp2 = gaussian_filter1d(tmp2,sigma=1)
plt.plot(tmp2,label='batch')
plt.legend()
plt.show()
a=np.ndarray.flatten(tmp)[:,None]
b=np.ndarray.flatten(tmp1)[:,None]
c=np.ndarray.flatten(tmp2)[:,None]
d=np.concatenate((a,b,c),axis=1)
plt.figure()
plt.boxplot(np.log(d))


# plotting distance between means (no mahab) over days (MAIN) with median
N=2
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(10)+1
X=np.arange(10)+1
# imagined 
tmp_main = np.median(mean_distances_imagined_days,axis=1)
tmp1 = np.median(tmp_main,axis=0)
tmp_boot, tmp_boot_std = median_bootstrap(tmp_main,1000)
tmp1b = tmp_boot_std/1#sqrt(tmp_main.shape[0])
tmp1 = np.insert(tmp1,0,tmp1[0],axis=0)
tmp1b = np.insert(tmp1b,0,tmp1b[0],axis=0)
tmp1 = np.convolve(tmp1, np.ones(N)/N, mode='same')
tmp1 = tmp1[1:]
tmp1b = tmp1b[1:]
plt.plot(X,tmp1,color="black",label = 'Imagined')
plt.fill_between(X, tmp1-tmp1b, tmp1+tmp1b,color="black",alpha=0.2)
# online
tmp_main = np.median(mean_distances_online_days,axis=1)
tmp2 = np.median(tmp_main,axis=0)
tmp_boot, tmp_boot_std = median_bootstrap(tmp_main,1000)
tmp2b = tmp_boot_std/1#sqrt(tmp_main.shape[0])
tmp2 = np.insert(tmp2,0,tmp2[0],axis=0)
tmp2b = np.insert(tmp2b,0,tmp2b[0],axis=0)
tmp2 = np.convolve(tmp2, np.ones(N)/N, mode='same')
tmp2b = np.convolve(tmp2b, np.ones(N)/N, mode='same')
tmp2 = tmp2[1:]
tmp2b = tmp2b[1:]
plt.plot(X,tmp2,color="blue",label = 'Online')
plt.fill_between(X, tmp2-tmp2b, tmp2+tmp2b,color="blue",alpha=0.2)
# batch
tmp_main = np.median(mean_distances_batch_days,axis=1)
tmp3 = np.median(tmp_main,axis=0)
tmp_boot, tmp_boot_std = median_bootstrap(tmp_main,1000)
tmp3b = tmp_boot_std/1#sqrt(tmp_main.shape[0])
tmp3 = np.insert(tmp3,0,tmp3[0],axis=0)
tmp3b = np.insert(tmp3b,0,tmp3b[0],axis=0)
tmp3 = np.convolve(tmp3, np.ones(N)/N, mode='same')
tmp3b = np.convolve(tmp3b, np.ones(N)/N, mode='same')
tmp3 = tmp3[1:]
tmp3b = tmp3b[1:]
plt.plot(X,tmp3,color="red",label = 'Batch')
plt.fill_between(X, tmp3-tmp3b, tmp3+tmp3b,color="red",alpha=0.2)
plt.legend()
plt.xlabel('Days',**hfont)
plt.ylabel('Distance between means',**hfont)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Distance_Between_Means_Days.svg'
fig.savefig(image_name, format=image_format, dpi=300)

tmp = np.concatenate((tmp1[:,None],tmp2[:,None],tmp3[:,None]),axis=1)
# tmp = np.concatenate((np.ndarray.flatten(var_overall_imagined_days)[:,None],
#                       np.ndarray.flatten(var_overall_online_days)[:,None],
#                       np.ndarray.flatten(var_overall_batch_days)[:,None]),axis=1)
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(tmp)
plt.xticks(ticks=[1,2,3],labels=('Imagined','Online','Batch'),**hfont)
plt.ylabel('Distance between means',**hfont)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Distance_Between_Means_Boxplot.svg'
fig.savefig(image_name, format=image_format, dpi=300)
print(np.mean(tmp,axis=0))


# plotting distance between means (no mahab) over days (MAIN) with mean
N=2
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(10)+1
X=np.arange(10)+1
# imagined 
tmp_main = np.mean(mean_distances_imagined_days,axis=0)
tmp1 = np.mean(tmp_main,axis=0)
tmp_boot_std = np.std(tmp_main,axis=0)
tmp1b = tmp_boot_std/sqrt(tmp_main.shape[0])
tmp1 = np.insert(tmp1,0,tmp1[0],axis=0)
tmp1b = np.insert(tmp1b,0,tmp1b[0],axis=0)
tmp1 = np.convolve(tmp1, np.ones(N)/N, mode='same')
tmp1 = tmp1[1:]
tmp1b = tmp1b[1:]
plt.plot(X,tmp1,color="black",label = 'Imagined')
plt.fill_between(X, tmp1-tmp1b, tmp1+tmp1b,color="black",alpha=0.2)
# online
tmp_main = np.mean(mean_distances_online_days,axis=0)
tmp2 = np.mean(tmp_main,axis=0)
tmp_boot_std = np.std(tmp_main,axis=0)
tmp2b = tmp_boot_std/sqrt(tmp_main.shape[0])
tmp2 = np.insert(tmp2,0,tmp2[0],axis=0)
tmp2b = np.insert(tmp2b,0,tmp2b[0],axis=0)
tmp2 = np.convolve(tmp2, np.ones(N)/N, mode='same')
tmp2b = np.convolve(tmp2b, np.ones(N)/N, mode='same')
tmp2 = tmp2[1:]
tmp2b = tmp2b[1:]
plt.plot(X,tmp2,color="blue",label = 'Online')
plt.fill_between(X, tmp2-tmp2b, tmp2+tmp2b,color="blue",alpha=0.2)
# batch
tmp_main = np.mean(mean_distances_batch_days,axis=0)
tmp3 = np.mean(tmp_main,axis=0)
tmp_boot_std = np.std(tmp_main,axis=0)
tmp3b = tmp_boot_std/sqrt(tmp_main.shape[0])
tmp3 = np.insert(tmp3,0,tmp3[0],axis=0)
tmp3b = np.insert(tmp3b,0,tmp3b[0],axis=0)
tmp3 = np.convolve(tmp3, np.ones(N)/N, mode='same')
tmp3b = np.convolve(tmp3b, np.ones(N)/N, mode='same')
tmp3 = tmp3[1:]
tmp3b = tmp3b[1:]
plt.plot(X,tmp3,color="red",label = 'Batch')
plt.fill_between(X, tmp3-tmp3b, tmp3+tmp3b,color="red",alpha=0.2)
plt.legend()
plt.xlabel('Days',**hfont)
plt.ylabel('Distance between means',**hfont)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Distance_Between_Means_Days.svg'
fig.savefig(image_name, format=image_format, dpi=300)

tmp = np.concatenate((tmp1[:,None],tmp2[:,None],tmp3[:,None]),axis=1)
# tmp = np.concatenate((np.ndarray.flatten(var_overall_imagined_days)[:,None],
#                       np.ndarray.flatten(var_overall_online_days)[:,None],
#                       np.ndarray.flatten(var_overall_batch_days)[:,None]),axis=1)
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(tmp)
plt.xticks(ticks=[1,2,3],labels=('Imagined','Online','Batch'),**hfont)
plt.ylabel('Distance between means',**hfont)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Distance_Between_Means_Boxplot.svg'
fig.savefig(image_name, format=image_format, dpi=300)
print(np.mean(tmp,axis=0))


#%% plotting mean centroid variances over days  (MAIN)

N=1
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
num_days=var_imagined_days.shape[2]
X=np.arange(num_days)+1
# X=np.arange(10)+1
# X=np.arange(10)+1
# imagined 
tmp_main = np.squeeze(np.median(var_imagined_days,axis=1))
tmp1 = np.mean(tmp_main,axis=0)
tmp1b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp1 = np.insert(tmp1,0,tmp1[0],axis=0)
tmp1b = np.insert(tmp1b,0,tmp1b[0],axis=0)
tmp1 = np.convolve(tmp1, np.ones(N)/N, mode='same')
tmp1 = tmp1[1:]
tmp1b = tmp1b[1:]
plt.plot(X,tmp1,color="black",label = 'Imagined')
plt.fill_between(X, tmp1-tmp1b, tmp1+tmp1b,color="black",alpha=0.2)
# online
tmp_main = np.squeeze(np.median(var_online_days,axis=1))
tmp2 = np.mean(tmp_main,axis=0)
tmp2b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp2 = np.insert(tmp2,0,tmp2[0],axis=0)
tmp2b = np.insert(tmp2b,0,tmp2b[0],axis=0)
tmp2 = np.convolve(tmp2, np.ones(N)/N, mode='same')
tmp2b = np.convolve(tmp2b, np.ones(N)/N, mode='same')
tmp2 = tmp2[1:]
tmp2b = tmp2b[1:]
plt.plot(X,tmp2,color="blue",label = 'Online')
plt.fill_between(X, tmp2-tmp2b, tmp2+tmp2b,color="blue",alpha=0.2)
# batch
tmp_main = np.squeeze(np.median(var_batch_days,axis=1))
tmp3 = np.mean(tmp_main,axis=0)
tmp3b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp3 = np.insert(tmp3,0,tmp3[0],axis=0)
tmp3b = np.insert(tmp3b,0,tmp3b[0],axis=0)
tmp3 = np.convolve(tmp3, np.ones(N)/N, mode='same')
tmp3b = np.convolve(tmp3b, np.ones(N)/N, mode='same')
tmp3 = tmp3[1:]
tmp3b = tmp3b[1:]
plt.plot(X,tmp3,color="red",label = 'Batch')
plt.fill_between(X, tmp3-tmp3b, tmp3+tmp3b,color="red",alpha=0.2)
plt.legend()
plt.xlabel('Days',**hfont)
plt.ylabel('Overall Latent Variance',**hfont)
plt.show()
# image_format = 'svg' # e.g .png, .svg, etc.
# image_name = 'Overall_Variance_Latent_Days.svg'
# fig.savefig(image_name, format=image_format, dpi=300)


tmp = np.concatenate((tmp1[:,None],tmp2[:,None],tmp3[:,None]),axis=1)
# tmp = np.concatenate((np.ndarray.flatten(var_overall_imagined_days)[:,None],
#                       np.ndarray.flatten(var_overall_online_days)[:,None],
#                       np.ndarray.flatten(var_overall_batch_days)[:,None]),axis=1)
tmp=np.log(tmp)
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot((tmp),whis=2,showfliers=False)
plt.xticks(ticks=[1,2,3],labels=('Imagined','Online','Batch'),**hfont)
plt.ylabel('Latent variance',**hfont)
plt.show()
plt.xticks(ticks=[1,2,3],labels='')
plt.yticks(ticks=[3,4,5,6,7],labels='')
#image_format = 'svg' # e.g .png, .svg, etc.
#image_name = 'Latent Variance_B3.svg'
#fig.savefig(image_name, format=image_format, dpi=300)
print(np.mean(tmp,axis=0))
print(stats.ttest_rel(tmp[:,1],tmp[:,2]))
print(np.median(tmp,axis=0))
print(stats.wilcoxon(tmp[:,2],tmp[:,1]))


# plotting the same but now across days without averaging
tmp_main = np.squeeze(np.mean(var_imagined_days,axis=1))
tmp1 = np.ndarray.flatten(tmp_main)
tmp_main = np.squeeze(np.mean(var_online_days,axis=1))
tmp2 = np.ndarray.flatten(tmp_main)
tmp_main = np.squeeze(np.mean(var_batch_days,axis=1))
tmp3 = np.ndarray.flatten(tmp_main)
tmp = np.concatenate((tmp1[:,None],tmp2[:,None],tmp3[:,None]),axis=1)

tmp=np.log(tmp)
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot((tmp),whis=2,showfliers=False)
plt.xticks(ticks=[1,2,3],labels=('Imagined','Online','Batch'),**hfont)
#plt.ylabel('Latent variance',**hfont)
plt.show()
plt.xticks(ticks=[1,2,3],labels='')
plt.yticks(ticks=[-1,1,3,5],labels='')
plt.show()
#image_format = 'svg' # e.g .png, .svg, etc.
#image_name = 'New_B1_Latent Variance.svg'
#fig.savefig(image_name, format=image_format, dpi=300)

print(np.mean(tmp,axis=0))
print(stats.ttest_rel(tmp[:,1],tmp[:,2]))
print(np.median(tmp,axis=0))
print(stats.wilcoxon(tmp[:,1],tmp[:,2]))

#as scatter plots
tmp1=np.log(tmp1)
tmp2=np.log(tmp2)
tmp3=np.log(tmp3)

idx = np.where(tmp1>-1)
tmp1=tmp1[idx]
idx = np.where(tmp2>-1)
tmp2=tmp2[idx]
idx = np.where(tmp3>-1)
tmp3=tmp3[idx]

fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
x1=np.ones((tmp1.shape[0],1)) + 0.05*rnd.randn(tmp1.shape[0])[:,None]
x2=2*np.ones((tmp2.shape[0],1)) + 0.05*rnd.randn(tmp2.shape[0])[:,None]
x3=3*np.ones((tmp3.shape[0],1)) + 0.05*rnd.randn(tmp3.shape[0])[:,None]
plt.scatter(x1, tmp1,s=5)
plt.scatter(x2, tmp2,s=5)
plt.scatter(x3, tmp3,s=5)
plt.hlines(np.median(tmp1),0.8,1.2,colors='black',linewidth=3)
plt.hlines(np.median(tmp2),1.8,2.2,colors='black',linewidth=3)
plt.hlines(np.median(tmp3),2.8,3.2,colors='black',linewidth=3)
plt.xticks([1,2,3])
plt.ylim((-1,5))
#plt.yticks([0.23,0.25,0.27])
plt.show()



#%% # plotting mean Mahalanobis distance over days  (MAIN MAIN)
# over days there is a learning effect where Mahab distance grows between the 
# actions and Batch is always greater than all others. 
N=2
num_days = mahab_distances_imagined_days.shape[2]
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(num_days)+1
# imagined 
tmp_main = np.squeeze(np.median(mahab_distances_imagined_days,axis=1))
tmp1 = np.mean(tmp_main,axis=0)
tmp1b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp1 = np.insert(tmp1,0,tmp1[0],axis=0)
tmp1b = np.insert(tmp1b,0,tmp1b[0],axis=0)
tmp1 = np.convolve(tmp1, np.ones(N)/N, mode='same')
tmp1 = tmp1[1:]
tmp1b = tmp1b[1:]
plt.plot(X,tmp1,color="black",label = 'Imagined')
plt.fill_between(X, tmp1-tmp1b, tmp1+tmp1b,color="black",alpha=0.2)
# online
tmp_main = np.squeeze(np.median(mahab_distances_online_days,1))
tmp2 = np.mean(tmp_main,axis=0)
tmp2b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp2 = np.insert(tmp2,0,tmp2[0],axis=0)
tmp2b = np.insert(tmp2b,0,tmp2b[0],axis=0)
tmp2 = np.convolve(tmp2, np.ones(N)/N, mode='same')
tmp2b = np.convolve(tmp2b, np.ones(N)/N, mode='same')
tmp2 = tmp2[1:]
tmp2b = tmp2b[1:]
plt.plot(X,tmp2,color="blue",label = 'Online')
plt.fill_between(X, tmp2-tmp2b, tmp2+tmp2b,color="blue",alpha=0.2)
# batch
tmp_main = np.squeeze(np.median(mahab_distances_batch_days,1))
tmp3 = np.mean(tmp_main,axis=0)
tmp3b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp3 = np.insert(tmp3,0,tmp3[0],axis=0)
tmp3b = np.insert(tmp3b,0,tmp3b[0],axis=0)
tmp3 = np.convolve(tmp3, np.ones(N)/N, mode='same')
tmp3b = np.convolve(tmp3b, np.ones(N)/N, mode='same')
tmp3 = tmp3[1:]
tmp3b = tmp3b[1:]
plt.plot(X,tmp3,color="red",label = 'Batch')
plt.fill_between(X, tmp3-tmp3b, tmp3+tmp3b,color="red",alpha=0.2)
plt.legend(loc='upper left')
plt.xlabel('Days',**hfont)
plt.ylabel('Mahalanobis Distances',**hfont)
plt.show()
# image_format = 'svg' # e.g .png, .svg, etc.
# image_name = 'Mean_Mahalanobis_dist_Days_withBatch.svg'
# fig.savefig(image_name, format=image_format, dpi=300)

tmp = np.concatenate((tmp1[:,None],tmp2[:,None],tmp3[:,None]),axis=1)
# tmp = np.concatenate((np.ndarray.flatten(var_overall_imagined_days)[:,None],
#                       np.ndarray.flatten(var_overall_online_days)[:,None],
#                       np.ndarray.flatten(var_overall_batch_days)[:,None]),axis=1)
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(tmp,whis=2,showfliers=False)
plt.xticks(ticks=[1,2,3],labels=('Imagined','Online','Batch'),**hfont)
plt.ylabel('Mahalanobis Distances',**hfont)
plt.show()
# image_format = 'svg' # e.g .png, .svg, etc.
# image_name = 'Mahab_Dist_Boxplot.svg'
# fig.savefig(image_name, format=image_format, dpi=300)
# print(np.mean(tmp,axis=0))

# linear regression stats
x=np.arange(num_days)+1
plt.figure()
for i in np.arange(tmp.shape[1]):
    y = tmp[:,i]
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    print(p_value)    
    plt.plot(x,y,'.')
    yhat = x*slope + intercept
    plt.plot(x,yhat)

plt.show()


#%% plotting mean silhoutte index over days  (MAIN)
N=1
num_days=silhoutte_imagined_days.shape[1]
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(num_days)+1
X=np.arange(num_days)+1
# imagined 
tmp_main = silhoutte_imagined_days
tmp1 = np.mean(tmp_main,axis=0)
tmp1b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp1 = np.insert(tmp1,0,tmp1[0],axis=0)
tmp1b = np.insert(tmp1b,0,tmp1b[0],axis=0)
tmp1 = np.convolve(tmp1, np.ones(N)/N, mode='same')
tmp1 = tmp1[1:]
tmp1b = tmp1b[1:]
plt.plot(X,tmp1,color="black",label = 'Imagined')
plt.fill_between(X, tmp1-tmp1b, tmp1+tmp1b,color="black",alpha=0.2)
# online
tmp_main = silhoutte_online_days
tmp2 = np.mean(tmp_main,axis=0)
tmp2b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp2 = np.insert(tmp2,0,tmp2[0],axis=0)
tmp2b = np.insert(tmp2b,0,tmp2b[0],axis=0)
tmp2 = np.convolve(tmp2, np.ones(N)/N, mode='same')
tmp2b = np.convolve(tmp2b, np.ones(N)/N, mode='same')
tmp2 = tmp2[1:]
tmp2b = tmp2b[1:]
plt.plot(X,tmp2,color="blue",label = 'Online')
plt.fill_between(X, tmp2-tmp2b, tmp2+tmp2b,color="blue",alpha=0.2)
# batch
tmp_main = silhoutte_batch_days
tmp3 = np.mean(tmp_main,axis=0)
tmp3b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp3 = np.insert(tmp3,0,tmp3[0],axis=0)
tmp3b = np.insert(tmp3b,0,tmp3b[0],axis=0)
tmp3 = np.convolve(tmp3, np.ones(N)/N, mode='same')
tmp3b = np.convolve(tmp3b, np.ones(N)/N, mode='same')
tmp3 = tmp3[1:]
tmp3b = tmp3b[1:]
plt.plot(X,tmp3,color="red",label = 'Batch')
plt.fill_between(X, tmp3-tmp3b, tmp3+tmp3b,color="red",alpha=0.2)
plt.legend()
plt.xlabel('Days',**hfont)
plt.ylabel('Silhoutte Index',**hfont)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Overall_Silhoutte_Index_Days.svg'
fig.savefig(image_name, format=image_format, dpi=300)


tmp = np.concatenate((tmp1[:,None],tmp2[:,None],tmp3[:,None]),axis=1)
# tmp = np.concatenate((np.ndarray.flatten(var_overall_imagined_days)[:,None],
#                       np.ndarray.flatten(var_overall_online_days)[:,None],
#                       np.ndarray.flatten(var_overall_batch_days)[:,None]),axis=1)
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(tmp)
plt.xticks(ticks=[1,2,3],labels=('Imagined','Online','Batch'),**hfont)
plt.ylabel('Silhoutte Index',**hfont)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Overall_Silhoutte_Index_boxplot.svg'
fig.savefig(image_name, format=image_format, dpi=300)
print(np.mean(tmp,axis=0))


#### plot  centroid variances th confidence intervals old method
sigma = 0.05
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(10)+1
# imagined days
tmp_main = np.squeeze(np.mean(var_imagined_days,1))
tmp = np.mean(tmp_main,axis=0)
tmp1 = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp = gaussian_filter1d(tmp, sigma=sigma)
tmp1 = gaussian_filter1d(tmp1, sigma=sigma)
plt.plot(X,tmp,color="black",label = 'Imagined')
plt.fill_between(X, tmp-tmp1, tmp+tmp1,color="black",alpha=0.2)
# online days 
tmp_main = np.squeeze(np.mean(var_online_days,1))
tmp = np.mean(tmp_main,axis=0)
tmp1 = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp = gaussian_filter1d(tmp, sigma=sigma)
tmp1 = gaussian_filter1d(tmp1, sigma=sigma)
plt.plot(X,tmp,color="blue",label = 'Online')
plt.fill_between(X, tmp-tmp1, tmp+tmp1,color="blue",alpha=0.2)
# batch update days 
tmp_main = np.squeeze(np.mean(var_batch_days,1))
tmp = np.mean(tmp_main,axis=0)
tmp1 = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp = gaussian_filter1d(tmp, sigma=sigma)
tmp1 = gaussian_filter1d(tmp1, sigma=sigma)
plt.plot(X,tmp,color="red",label = 'Batch')
plt.fill_between(X, tmp-tmp1, tmp+tmp1,color="red",alpha=0.2)
plt.xlabel('Days',**hfont)
plt.ylabel('Variance of latent centroids',**hfont)
plt.legend()
plt.show()
# save
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Mean_DistanceBetweenCentroids_Days_Latent.svg'
fig.savefig(image_name, format=image_format, dpi=300)

#### plotting variances with confidence intervals 
sigma = 0.01
N=1 # N day running average
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(10)+1
# imagined days
tmp_main = np.squeeze(np.mean(var_imagined_days,1))
tmp = np.mean(tmp_main,axis=0)
tmp1 = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
#tmp = gaussian_filter1d(tmp, sigma=sigma)
#tmp1 = gaussian_filter1d(tmp1, sigma=sigma)
tmp = np.convolve(tmp, np.ones(N)/N, mode='same')
tmp1 = np.convolve(tmp1, np.ones(N)/N, mode='same')
plt.plot(X,tmp,color="black",label = 'Imagined')
plt.fill_between(X, tmp-tmp1, tmp+tmp1,color="black",alpha=0.2)
# online days 
tmp_main = np.squeeze(np.mean(var_online_days,1))
tmp = np.mean(tmp_main,axis=0)
tmp1 = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
#tmp = gaussian_filter1d(tmp, sigma=sigma)
#tmp1 = gaussian_filter1d(tmp1, sigma=sigma)
tmp = np.convolve(tmp, np.ones(N)/N, mode='same')
tmp1 = np.convolve(tmp1, np.ones(N)/N, mode='same')
plt.plot(X,tmp,color="blue",label = 'Online')
plt.fill_between(X, tmp-tmp1, tmp+tmp1,color="blue",alpha=0.2)
# batch update days 
tmp_main = np.squeeze(np.median(var_batch_days,1))
tmp = np.mean(tmp_main,axis=0)
tmp1 = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
#tmp = gaussian_filter1d(tmp, sigma=sigma)
#tmp1 = gaussian_filter1d(tmp1, sigma=sigma)
tmp = np.convolve(tmp, np.ones(N)/N, mode='same')
tmp1 = np.convolve(tmp1, np.ones(N)/N, mode='same')
plt.plot(X,tmp,color="red",label = 'Batch')
plt.fill_between(X, tmp-tmp1, tmp+tmp1,color="red",alpha=0.2)
plt.xlabel('Days',**hfont)
plt.ylabel('Variance of Centroids',**hfont)
plt.legend()
plt.show()
# save
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Variances_Days_Latent_Centroids.svg'
fig.savefig(image_name, format=image_format, dpi=300)



plt.figure()
plt.boxplot(accuracy_online_days)
plt.ylim((20,100))
plt.figure()
plt.boxplot(accuracy_imagined_days)
plt.ylim((20,100))
plt.figure()
plt.boxplot(accuracy_batch_days)
plt.ylim((20,100))
plt.show()


plt.figure()
tmp = gaussian_filter1d(np.mean(accuracy_imagined_days,axis=0),sigma=1)
plt.plot(tmp,color="black")
plt.ylim((20,100))
tmp = gaussian_filter1d(np.mean(accuracy_online_days,axis=0),sigma=1)
plt.plot(tmp,color="blue")
tmp = gaussian_filter1d(np.mean(accuracy_batch_days,axis=0),sigma=1)
plt.plot(tmp,color="red")
plt.ylim((20,100))
plt.show()

plt.figure()
tmp = gaussian_filter1d(np.mean(silhoutte_imagined_days,axis=0),sigma=1)
plt.plot(tmp,color="black")
tmp = gaussian_filter1d(np.mean(silhoutte_online_days,axis=0),sigma=1)
plt.plot(tmp,color="blue")
tmp = gaussian_filter1d(np.mean(silhoutte_batch_days,axis=0),sigma=1)
plt.plot(tmp,color="red")
plt.show()

plt.figure()
tmp = gaussian_filter1d(np.mean(silhoutte_imagined_days,axis=0),sigma=1)
plt.plot(tmp,color="black")
tmp = gaussian_filter1d(np.mean(silhoutte_online_days,axis=0),sigma=1)
plt.plot(tmp,color="blue")
plt.show()



#### plot median Mahalanobis with bootstrap confidence intervals  (MAIN)
sigma = 0.75
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(10)+1
# imagined days
tmp_main = np.squeeze(np.mean(mahab_distances_imagined_days,1))
tmp = np.median(tmp_main,axis=0)
tmp_boot, tmp_boot_std = median_bootstrap(tmp_main,1000)
tmp1 = tmp_boot_std/1#sqrt(tmp_main.shape[0])
tmp = gaussian_filter1d(tmp, sigma=sigma)
tmp1 = gaussian_filter1d(tmp1, sigma=sigma)
plt.plot(X,tmp,color="black",label = 'Imagined')
plt.fill_between(X, tmp-tmp1, tmp+tmp1,color="black",alpha=0.2)
#plt.plot(X,tmp+tmp1,color="black",linestyle="dotted")
#plt.plot(X,tmp-tmp1,color="black",linestyle="dotted")
# online days 
tmp_main = np.squeeze(np.mean(mahab_distances_online_days,1))
tmp = np.median(tmp_main,axis=0)
tmp_boot, tmp_boot_std = median_bootstrap(tmp_main,1000)
tmp2 = tmp_boot_std/1#sqrt(tmp_main.shape[0])
tmp = gaussian_filter1d(tmp, sigma=sigma)
tmp2 = gaussian_filter1d(tmp2, sigma=sigma)
plt.plot(X,tmp,color="blue",label = 'Online')
plt.fill_between(X, tmp-tmp2, tmp+tmp2,color="blue",alpha=0.2)
#plt.plot(X,tmp+tmp1,color="blue",linestyle="dotted")
#plt.plot(X,tmp-tmp1,color="blue",linestyle="dotted")
# batch update days 
tmp_main = np.squeeze(np.mean(mahab_distances_batch_days,1))
tmp = np.median(tmp_main,axis=0)
tmp_boot, tmp_boot_std = median_bootstrap(tmp_main,1000)
tmp3 = tmp_boot_std/1#sqrt(tmp_main.shape[0])
tmp = gaussian_filter1d(tmp, sigma=sigma)
tmp3 = gaussian_filter1d(tmp3, sigma=sigma)
plt.plot(X,tmp,color="red",label = 'Batch')
plt.fill_between(X, tmp-tmp3, tmp+tmp3,color="red",alpha=0.2)
#plt.plot(X,tmp+tmp1,color="red",linestyle="dotted")
#plt.plot(X,tmp-tmp1,color="red",linestyle="dotted")
plt.xlabel('Days',**hfont)
plt.ylabel('Mahalanobis distance between latent centroids',**hfont)
plt.legend()
plt.show()
# save
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Mahab_Days_Latent_withBatch.svg'
fig.savefig(image_name, format=image_format, dpi=300)

tmp = np.concatenate((tmp1[:,None],tmp2[:,None],tmp3[:,None]),axis=1)
# tmp = np.concatenate((np.ndarray.flatten(var_overall_imagined_days)[:,None],
#                       np.ndarray.flatten(var_overall_online_days)[:,None],
#                       np.ndarray.flatten(var_overall_batch_days)[:,None]),axis=1)
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(tmp)
plt.xticks(ticks=[1,2,3],labels=('Imagined','Online','Batch'),**hfont)
plt.ylabel('Mahab Dist',**hfont)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Overall_Mahab_Dist_boxplot.svg'
fig.savefig(image_name, format=image_format, dpi=300)
print(np.mean(tmp,axis=0))



# Median Mahalanobis distance -> with bootstrapped standard errors of the median
sigma = 0.01
N=1 # N day running average
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(10)+1
# imagined days
tmp_main = np.squeeze(np.mean(mahab_distances_imagined_days,1))
tmp = np.median(tmp_main,axis=0)
tmp_boot, tmp_boot_std = median_bootstrap(tmp_main,1000)
tmp1 = tmp_boot_std/1#sqrt(tmp_main.shape[0])
tmp = np.convolve(tmp, np.ones(N)/N, mode='same')
tmp1 = np.convolve(tmp1, np.ones(N)/N, mode='same')
plt.plot(X,tmp,color="black",label = 'Imagined')
plt.fill_between(X, tmp-tmp1, tmp+tmp1,color="black",alpha=0.2)
# online days 
tmp_main = np.squeeze(np.mean(mahab_distances_online_days,1))
tmp = np.median(tmp_main,axis=0)
tmp_boot, tmp_boot_std = median_bootstrap(tmp_main,1000)
tmp2 = tmp_boot_std/1#sqrt(tmp_main.shape[0])
tmp = np.convolve(tmp, np.ones(N)/N, mode='same')
tmp2 = np.convolve(tmp2, np.ones(N)/N, mode='same')
plt.plot(X,tmp,color="blue",label = 'Online')
plt.fill_between(X, tmp-tmp2, tmp+tmp2,color="blue",alpha=0.2)
# batch update days 
tmp_main = np.squeeze(np.mean(mahab_distances_batch_days,1))
tmp = np.median(tmp_main,axis=0)
tmp_boot, tmp_boot_std = median_bootstrap(tmp_main,1000)
tmp3 = tmp_boot_std/1#sqrt(tmp_main.shape[0])
tmp = np.convolve(tmp, np.ones(N)/N, mode='same')
tmp3 = np.convolve(tmp3, np.ones(N)/N, mode='same')
plt.plot(X,tmp,color="red",label = 'Batch')
plt.fill_between(X, tmp-tmp3, tmp+tmp3,color="red",alpha=0.2)
plt.xlabel('Days',**hfont)
plt.ylabel('Mahalanobis Distance',**hfont)
plt.legend()
plt.show()
# save
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Variances_Days_Latent.svg'
fig.savefig(image_name, format=image_format, dpi=300)

tmp = np.concatenate((tmp1[:,None],tmp2[:,None],tmp3[:,None]),axis=1)
# tmp = np.concatenate((np.ndarray.flatten(var_overall_imagined_days)[:,None],
#                       np.ndarray.flatten(var_overall_online_days)[:,None],
#                       np.ndarray.flatten(var_overall_batch_days)[:,None]),axis=1)
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(tmp)
plt.xticks(ticks=[1,2,3],labels=('Imagined','Online','Batch'),**hfont)
plt.ylabel('Mahab Dist',**hfont)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Overall_Mahab_Dist_boxplot.svg'
fig.savefig(image_name, format=image_format, dpi=300)
print(np.mean(tmp,axis=0))



#### plot Mahab dist mean with confidence intervals old method
sigma = 1
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(10)+1
# imagined days
tmp_main = np.squeeze(np.median(mean_distances_imagined_days,1))
tmp = np.mean(tmp_main,axis=0)
tmp1 = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp = gaussian_filter1d(tmp, sigma=sigma)
tmp1 = gaussian_filter1d(tmp1, sigma=sigma)
plt.plot(X,tmp,color="black",label = 'Imagined')
plt.fill_between(X, tmp-tmp1, tmp+tmp1,color="black",alpha=0.2)
# online days 
tmp_main = np.squeeze(np.median(mean_distances_online_days,1))
tmp = np.mean(tmp_main,axis=0)
tmp1 = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp = gaussian_filter1d(tmp, sigma=sigma)
tmp1 = gaussian_filter1d(tmp1, sigma=sigma)
plt.plot(X,tmp,color="blue",label = 'Online')
plt.fill_between(X, tmp-tmp1, tmp+tmp1,color="blue",alpha=0.2)
# batch update days 
tmp_main = np.squeeze(np.median(mean_distances_batch_days,1))
tmp = np.mean(tmp_main,axis=0)
tmp1 = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp = gaussian_filter1d(tmp, sigma=sigma)
tmp1 = gaussian_filter1d(tmp1, sigma=sigma)
plt.plot(X,tmp,color="red",label = 'Batch')
plt.fill_between(X, tmp-tmp1, tmp+tmp1,color="red",alpha=0.2)
plt.xlabel('Days',**hfont)
plt.ylabel('Mahalanobis distance',**hfont)
plt.legend()
plt.show()
# save
# image_format = 'svg' # e.g .png, .svg, etc.
# image_name = 'Mahab_Days_Latent.svg'
# fig.savefig(image_name, format=image_format, dpi=300)


#### plot median Mahalanobis with confidence intervals old method
sigma = 1
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(10)+1
# imagined days
tmp_main = np.squeeze(np.mean(mahab_distances_imagined_days,1))
tmp = np.median(tmp_main,axis=0)
tmp1 = scipy.stats.median_abs_deviation(tmp_main,axis=0)
tmp1 = (1.4826*tmp1)/np.sqrt(21)
#tmp1 = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp = gaussian_filter1d(tmp, sigma=sigma)
tmp1 = gaussian_filter1d(tmp1, sigma=sigma)
plt.plot(X,tmp,color="black",label = 'Imagined')
plt.fill_between(X, tmp-tmp1, tmp+tmp1,color="black",alpha=0.2)
#plt.plot(X,tmp+tmp1,color="black",linestyle="dotted")
#plt.plot(X,tmp-tmp1,color="black",linestyle="dotted")
# online days 
tmp_main = np.squeeze(np.mean(mahab_distances_online_days,1))
tmp = np.median(tmp_main,axis=0)
tmp1 = scipy.stats.median_abs_deviation(tmp_main,axis=0)
tmp1 = (1.4826*tmp1)/np.sqrt(21)
#tmp1 = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp = gaussian_filter1d(tmp, sigma=sigma)
tmp1 = gaussian_filter1d(tmp1, sigma=sigma)
plt.plot(X,tmp,color="blue",label = 'Online')
plt.fill_between(X, tmp-tmp1, tmp+tmp1,color="blue",alpha=0.2)
#plt.plot(X,tmp+tmp1,color="blue",linestyle="dotted")
#plt.plot(X,tmp-tmp1,color="blue",linestyle="dotted")
# batch update days 
tmp_main = np.squeeze(np.mean(mahab_distances_batch_days,1))
tmp = np.median(tmp_main,axis=0)
tmp1 = scipy.stats.median_abs_deviation(tmp_main,axis=0)
tmp1 = (1.4826*tmp1)/np.sqrt(21)
#tmp1 = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp = gaussian_filter1d(tmp, sigma=sigma)
tmp1 = gaussian_filter1d(tmp1, sigma=sigma)
plt.plot(X,tmp,color="red",label = 'Batch')
plt.fill_between(X, tmp-tmp1, tmp+tmp1,color="red",alpha=0.2)
#plt.plot(X,tmp+tmp1,color="red",linestyle="dotted")
#plt.plot(X,tmp-tmp1,color="red",linestyle="dotted")
plt.xlabel('Days',**hfont)
plt.ylabel('Mahalanobis distance',**hfont)
plt.legend()
plt.show()
# save
# image_format = 'svg' # e.g .png, .svg, etc.
# image_name = 'Mahab_Days_Latent.svg'
# fig.savefig(image_name, format=image_format, dpi=300)

    
# new method plotting 
sigma = 1
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(10)+1
# imagined days
tmp_main = np.squeeze(np.mean(mahab_distances_imagined_days,1))
tmp = np.median(tmp_main,axis=0)
tmp_boot = stats.bootstrap((tmp_main,), np.median)
tmp1_low = tmp_boot.confidence_interval.low
tmp1_high = tmp_boot.confidence_interval.high
tmp = gaussian_filter1d(tmp, sigma=sigma)
tmp1_low = gaussian_filter1d(tmp1_low, sigma=sigma)
tmp1_high = gaussian_filter1d(tmp1_high, sigma=sigma)
plt.plot(X,tmp,color="black",label = 'Imagined')
plt.fill_between(X, tmp1_low, tmp1_high,color="black",alpha=0.2)
#plt.plot(X,tmp+tmp1,color="black",linestyle="dotted")
#plt.plot(X,tmp-tmp1,color="black",linestyle="dotted")
# online days 
tmp_main = np.squeeze(np.mean(mahab_distances_online_days,1))
tmp = np.median(tmp_main,axis=0)
tmp_boot = stats.bootstrap((tmp_main,), np.median)
tmp1_low = tmp_boot.confidence_interval.low
tmp1_high = tmp_boot.confidence_interval.high
tmp = gaussian_filter1d(tmp, sigma=sigma)
tmp1_low = gaussian_filter1d(tmp1_low, sigma=sigma)
tmp1_high = gaussian_filter1d(tmp1_high, sigma=sigma)
plt.plot(X,tmp,color="blue",label = 'Online')
plt.fill_between(X, tmp1_low, tmp1_high,color="blue",alpha=0.2)
# batch update days 
tmp_main = np.squeeze(np.mean(mahab_distances_batch_days,1))
tmp = np.median(tmp_main,axis=0)
tmp_boot = stats.bootstrap((tmp_main,), np.median)
tmp1_low = tmp_boot.confidence_interval.low
tmp1_high = tmp_boot.confidence_interval.high
tmp = gaussian_filter1d(tmp, sigma=sigma)
tmp1_low = gaussian_filter1d(tmp1_low, sigma=sigma)
tmp1_high = gaussian_filter1d(tmp1_high, sigma=sigma)
plt.plot(X,tmp,color="red",label = 'Batch')
plt.fill_between(X, tmp1_low, tmp1_high,color="red",alpha=0.2)
plt.xlabel('Days',**hfont)
plt.ylabel('Mahalanobis distance',**hfont)
plt.legend()
plt.show()



#### plot  Mean distance separation with bootstrap confidence intervals  (MAIN)
sigma = 1
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(10)+1
# imagined days
tmp_main = np.squeeze(np.mean(mean_distances_imagined_days,1))
tmp = np.median(tmp_main,axis=0)
tmp_boot, tmp_boot_std = median_bootstrap(tmp_main,1000)
tmp1 = tmp_boot_std/1#sqrt(tmp_main.shape[0])
tmp = gaussian_filter1d(tmp, sigma=sigma)
tmp1 = gaussian_filter1d(tmp1, sigma=sigma)
plt.plot(X,tmp,color="black",label = 'Imagined')
plt.fill_between(X, tmp-tmp1, tmp+tmp1,color="black",alpha=0.2)
#plt.plot(X,tmp+tmp1,color="black",linestyle="dotted")
#plt.plot(X,tmp-tmp1,color="black",linestyle="dotted")
# online days 
tmp_main = np.squeeze(np.mean(mean_distances_online_days,1))
tmp = np.median(tmp_main,axis=0)
tmp_boot, tmp_boot_std = median_bootstrap(tmp_main,1000)
tmp1 = tmp_boot_std/1#sqrt(tmp_main.shape[0])
tmp = gaussian_filter1d(tmp, sigma=sigma)
tmp1 = gaussian_filter1d(tmp1, sigma=sigma)
plt.plot(X,tmp,color="blue",label = 'Online')
plt.fill_between(X, tmp-tmp1, tmp+tmp1,color="blue",alpha=0.2)
#plt.plot(X,tmp+tmp1,color="blue",linestyle="dotted")
#plt.plot(X,tmp-tmp1,color="blue",linestyle="dotted")
# batch update days 
tmp_main = np.squeeze(np.mean(mean_distances_batch_days,1))
tmp = np.median(tmp_main,axis=0)
tmp_boot, tmp_boot_std = median_bootstrap(tmp_main,1000)
tmp1 = tmp_boot_std/1#sqrt(tmp_main.shape[0])
tmp = gaussian_filter1d(tmp, sigma=sigma)
tmp1 = gaussian_filter1d(tmp1, sigma=sigma)
plt.plot(X,tmp,color="red",label = 'Batch')
plt.fill_between(X, tmp-tmp1, tmp+tmp1,color="red",alpha=0.2)
#plt.plot(X,tmp+tmp1,color="red",linestyle="dotted")
#plt.plot(X,tmp-tmp1,color="red",linestyle="dotted")
plt.xlabel('Days',**hfont)
plt.ylabel('Mahalanobis distance',**hfont)
plt.legend()
plt.show()
# save
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Mahab_Days_Latent_withBatch.svg'
fig.savefig(image_name, format=image_format, dpi=300)


#### plot  dist means with confidence intervals old method (MAIN)
sigma = 1
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(10)+1
# imagined days
tmp_main = np.squeeze(np.mean(mean_distances_imagined_days,1))
tmp = np.mean(tmp_main,axis=0)
tmp1 = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp = gaussian_filter1d(tmp, sigma=sigma)
tmp1 = gaussian_filter1d(tmp1, sigma=sigma)
plt.plot(X,tmp,color="black",label = 'Imagined')
plt.fill_between(X, tmp-tmp1, tmp+tmp1,color="black",alpha=0.2)
# online days 
tmp_main = np.squeeze(np.mean(mean_distances_online_days,1))
tmp = np.mean(tmp_main,axis=0)
tmp1 = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp = gaussian_filter1d(tmp, sigma=sigma)
tmp1 = gaussian_filter1d(tmp1, sigma=sigma)
plt.plot(X,tmp,color="blue",label = 'Online')
plt.fill_between(X, tmp-tmp1, tmp+tmp1,color="blue",alpha=0.2)
# batch update days 
tmp_main = np.squeeze(np.mean(mean_distances_batch_days,1))
tmp = np.mean(tmp_main,axis=0)
tmp1 = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp = gaussian_filter1d(tmp, sigma=sigma)
tmp1 = gaussian_filter1d(tmp1, sigma=sigma)
plt.plot(X,tmp,color="red",label = 'Batch')
plt.fill_between(X, tmp-tmp1, tmp+tmp1,color="red",alpha=0.2)
plt.xlabel('Days',**hfont)
plt.ylabel('Mean distance between latent centroids',**hfont)
plt.legend()
plt.show()
# save
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Mean_DistanceBetweenCentroids_Days_Latent.svg'
fig.savefig(image_name, format=image_format, dpi=300)

# running it through a common autoencoder space
condn_data_online_tmp = condn_data_online
Yonline_tmp = Yonline
for i in np.arange(1500):  
    idx = rnd.choice(696,5,replace=False)
    tmp = condn_data_online[idx,:]
    tmp = np.mean(tmp,axis=0)
    tmp = tmp[:,None].T
    condn_data_online_tmp = np.append(condn_data_online_tmp,tmp,axis=0)

condn_data_batch_tmp = condn_data_batch
for i in np.arange(1700):
    idx = rnd.choice(343,5,replace=False)
    tmp = condn_data_batch[idx,:]
    tmp = np.mean(tmp,axis=0)
    tmp = tmp[:,None].T
    condn_data_batch_tmp = np.append(condn_data_batch_tmp,tmp,axis=0)
    
        
    

condn_data = np.concatenate((condn_data_imagined,condn_data_online,condn_data_batch))
Y = np.concatenate((Yimagined,Yonline,Ybatch))
if 'model' in locals():
    del model    
     
Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data,Y,0.8)     
model = iAutoencoder(input_size,hidden_size,latent_dims,num_classes).to(device)        
model,acc = training_loop_iAE(model,num_epochs,batch_size,learning_rate,batch_val,
                      patience,gradient_clipping,nn_filename,
                      Xtrain,Ytrain,Xtest,Ytest,
                      input_size,hidden_size,latent_dims,num_classes)


#%% PLOTTING (MAIN) THE HEAT MAPS OF RECONSTRUCTED ACTIVITY PASSING THRU AE
# plotting some example boxplots comparing channels

# chn  18,  ,3 
chn = [3,18]
query=1
for i in np.arange(len(chn)):
        
    tmp1 = (hg_recon_imag[query])[:,chn[i]]
    tmp2 = (hg_recon_online[query])[:,chn[i]]
    tmp3 = (hg_recon_batch[query])[:,chn[i]]
    
    if 'tmp' in locals():
        del tmp
        
    tmp = [tmp1,tmp2,tmp3]
    fig=plt.figure()
    hfont = {'fontname':'Arial'}
    plt.rc('font',family='Arial')
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams.update({'font.size': 6})
    plt.boxplot(tmp,showfliers=False);
    plt.title(chn[i])
    plt.xticks(ticks=[1,2,3],labels=('OL','CL1','CL2'),**hfont)    
    #plt.tick_params(labelleft=False)
    plt.show()


image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'BMF_Ch31_hG_Std.svg'
fig.savefig(image_name, format=image_format, dpi=300)


    
# plot only the desired channel 
tmp = np.zeros((32,))
tmp[[31,18]]=1
tmp = np.reshape(tmp,(4,8))
plt.figure()
fig1=plt.imshow(tmp)
plt.tick_params(labelleft=False,labelbottom=False,bottom=False,left=False)
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Ch18_31.svg'
plt.savefig(image_name, format=image_format, dpi=300)



# as scatter plot
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
x1=np.ones((tmp1.shape[0],1)) + 0.05*rnd.randn(tmp1.shape[0])[:,None]
x2=2*np.ones((tmp2.shape[0],1)) + 0.05*rnd.randn(tmp2.shape[0])[:,None]
x3=3*np.ones((tmp3.shape[0],1)) + 0.05*rnd.randn(tmp3.shape[0])[:,None]
plt.scatter(x1, tmp1,s=5)
plt.scatter(x2, tmp2,s=5)
plt.scatter(x3, tmp3,s=5)
plt.hlines(np.mean(tmp1),0.8,1.2,colors='black',linewidth=3)
plt.hlines(np.mean(tmp2),1.8,2.2,colors='black',linewidth=3)
plt.hlines(np.mean(tmp3),2.8,3.2,colors='black',linewidth=3)
plt.xticks([1,2,3])
#plt.ylim((-1,5))
#plt.yticks([0.23,0.25,0.27])
plt.show()



# chn  18,  ,3 
chn = [29]
query=6
for i in np.arange(len(chn)):
        
    tmp1 = (beta_recon_imag[query])[:,chn[i]]
    tmp2 = (beta_recon_online[query])[:,chn[i]]
    tmp3 = (beta_recon_batch[query])[:,chn[i]]
    
    if 'tmp' in locals():
        del tmp
        
    tmp = [tmp1,tmp2,tmp3]
    plt.figure();
    plt.boxplot(tmp,showfliers=False);
    plt.title(chn[i])

# as scatter plot
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
x1=np.ones((tmp1.shape[0],1)) + 0.05*rnd.randn(tmp1.shape[0])[:,None]
x2=2*np.ones((tmp2.shape[0],1)) + 0.05*rnd.randn(tmp2.shape[0])[:,None]
x3=3*np.ones((tmp3.shape[0],1)) + 0.05*rnd.randn(tmp3.shape[0])[:,None]
plt.scatter(x1, tmp1,s=5)
plt.scatter(x2, tmp2,s=5)
plt.scatter(x3, tmp3,s=5)
plt.hlines(np.mean(tmp1),0.8,1.2,colors='black',linewidth=3)
plt.hlines(np.mean(tmp2),1.8,2.2,colors='black',linewidth=3)
plt.hlines(np.mean(tmp3),2.8,3.2,colors='black',linewidth=3)
plt.xticks([1,2,3])
#plt.ylim((-1,5))
#plt.yticks([0.23,0.25,0.27])
plt.show()



# =============================================================================
# this code change here relates to trying to see spatial reconstruction after 
# passing data thru AE even though they come from the same data class
# It can be changed to just look at the reconstruction from different classes
# =============================================================================

query=5
tmp = np.mean(hg_recon_imag[query][:300,:],axis=0)
tmp = np.reshape(tmp,(4,8))
xmax1,xmin1 = tmp.max(),tmp.min()
plt.figure()
fig1=plt.imshow(tmp)
plt.colorbar()
plt.show


tmp = np.mean(hg_recon_online[query][109:,:],axis=0)
tmp = np.reshape(tmp,(4,8))
xmax2,xmin2 = tmp.max(),tmp.min()
plt.figure()
fig2=plt.imshow(tmp)
plt.colorbar()

tmp = np.mean(hg_recon_batch[query][300:,:],axis=0)
tmp = np.reshape(tmp,(4,8))
xmax3,xmin3 = tmp.max(),tmp.min()
plt.figure()
fig3=plt.imshow(tmp)
plt.colorbar()



xmax = np.array([xmax1,xmax2,xmax3])
xmin = np.array([xmin1,xmin2,xmin3])
fig1.set_clim(xmin.min(),xmax.max())
fig2.set_clim(xmin.min(),xmax.max())
fig3.set_clim(xmin.min(),xmax.max())

### plotting spatial correlations and recon activity map
# high gamma
actions =['Rt Thumb','Left Leg','Lt Thumb','Head','Lips','Tongue','Both Middle Finger']
corr_coef_hg = np.array([])
hand_knob_act_imag=np.array([])
hand_knob_act_online=np.array([])
hand_knob_act_batch=np.array([])
var_imag=np.array([])
var_online=np.array([])
var_batch=np.array([])
hand_channels = np.array([23,31])
for query in np.arange(7):        
    
    # getting hand knob activation
    tmp = hg_recon_imag[query]
    a = np.std(tmp,axis=0)
    var_imag = np.append(var_imag,a)
    tmp = tmp[:,hand_channels] # get the hand knob channels 30,31,22,23,15
    hand_knob_act_imag = tmp    
    # plotting
    fig=plt.figure()
    plt.suptitle(actions[query])
    tmp = np.mean(hg_recon_imag[query],axis=0)
    #tmp = stats.zscore(tmp)
    tmp1 = np.reshape(tmp,(4,8))
    xmax1,xmin1 = tmp.max(),tmp.min()
    plt.subplot(311)
    fig1=plt.imshow(tmp1)
    plt.axis('off')    
    plt.colorbar()
        
    # getting hand knob activation
    tmp = hg_recon_online[query]
    a = np.std(tmp,axis=0)
    var_online = np.append(var_online,a)
    tmp = tmp[:,hand_channels] # get the hand knob channels 30,31,22,23,15
    hand_knob_act_online = tmp        
    # plotting
    tmp = np.mean(hg_recon_online[query],axis=0)
    #tmp = stats.zscore(tmp)
    tmp2 = np.reshape(tmp,(4,8))
    xmax2,xmin2 = tmp.max(),tmp.min()
    plt.subplot(312)    
    fig2=plt.imshow(tmp2)   
    plt.axis('off')      
    plt.colorbar()
        
    # getting hand knob activation
    tmp = hg_recon_batch[query]
    a = np.std(tmp,axis=0)
    var_batch = np.append(var_batch,a)
    tmp = tmp[:,hand_channels] # get the hand knob channels 30,31,22,23,15
    hand_knob_act_batch = tmp        
    # plotting
    tmp = np.mean(hg_recon_batch[query],axis=0) #first 8 nos form first row, etc
    #tmp = stats.zscore(tmp)
    tmp3 = np.reshape(tmp,(4,8)) # first 8 values form the first row of the grid and so on
    xmax3,xmin3 = tmp.max(),tmp.min()
    plt.subplot(313)    
    fig3=plt.imshow(tmp3)
    plt.axis('off')    
    plt.colorbar()
    
    a = np.corrcoef(np.ndarray.flatten(tmp1),np.ndarray.flatten(tmp2))[0,1]
    b = np.corrcoef(np.ndarray.flatten(tmp1),np.ndarray.flatten(tmp3))[0,1]
    c = np.corrcoef(np.ndarray.flatten(tmp2),np.ndarray.flatten(tmp3))[0,1]
    # a = np.dot(tmp1.flatten(),tmp2.flatten())
    # b = np.dot(tmp1.flatten(),tmp3.flatten())
    # c = np.dot(tmp2.flatten(),tmp3.flatten())
    corr_coef_hg = np.append(corr_coef_hg,[a,b,c])
    
    xmax = np.array([xmax1,xmax2,xmax3])
    xmin = np.array([xmin1,xmin2,xmin3])
    fig1.set_clim(xmin.min(),xmax.max())
    fig2.set_clim(xmin.min(),xmax.max())
    fig3.set_clim(xmin.min(),xmax.max())
    
    image_format = 'svg' # e.g .png, .svg, etc.
    image_name = actions[query] + '_hg_Day1_New.svg'
    fig.savefig(image_name, format=image_format, dpi=300)

# plotting hand knob activation
x= [hand_knob_act_imag.flatten()  ,hand_knob_act_online.flatten(), hand_knob_act_batch.flatten() ]
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(x,showfliers=False)
plt.xticks(ticks=[1,2,3],labels=('Imagined','Online','Batch'),**hfont)
plt.ylabel('hG Hand knob activity',**hfont)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Hand_Knob_Activation.svg'
fig.savefig(image_name, format=image_format, dpi=300)

# plotting variance boxplots 
x= [var_imag  ,var_online, var_batch]
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
#plt.xlim((0,0.14))
#plt.xlabel('hG Std. Deviation',**hfont)
#plt.ylabel('Count',**hfont)
plt.tick_params(labelleft=False,labelbottom=False)
plt.legend()
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Hg_Variance.svg'
fig.savefig(image_name, format=image_format, dpi=300)

# delta
actions =['Rt Thumb','Left Leg','Lt Thumb','Head','Lips','Tongue','Both Middle Finger']
corr_coef_delta = np.array([])
hand_knob_act_imag=np.array([])
hand_knob_act_online=np.array([])
hand_knob_act_batch=np.array([])
var_imag=np.array([])
var_online=np.array([])
var_batch=np.array([])
hand_channels = np.array([23,31])
for query in np.arange(7):        
    
    # getting hand knob activation
    tmp = delta_recon_imag[query]
    a = np.std(tmp,axis=0)
    var_imag = np.append(var_imag,a)
    tmp = tmp[:,hand_channels] # get the hand knob channels 30,31,22,23,15
    hand_knob_act_imag = np.append(hand_knob_act_imag,tmp)
    # plotting
    fig=plt.figure()
    plt.suptitle(actions[query])
    tmp = np.mean(delta_recon_imag[query],axis=0)
    #tmp = stats.zscore(tmp)
    tmp1 = np.reshape(tmp,(4,8))
    xmax1,xmin1 = tmp.max(),tmp.min()
    plt.subplot(311)
    fig1=plt.imshow(tmp1)
    plt.axis('off')    
    plt.colorbar()
        
    # getting hand knob activation
    tmp = delta_recon_online[query]
    a = np.std(tmp,axis=0)
    var_online = np.append(var_online,a)
    tmp = tmp[:,hand_channels] # get the hand knob channels 30,31,22,23,15
    hand_knob_act_online = np.append(hand_knob_act_online,tmp)
    # plotting
    tmp = np.mean(delta_recon_online[query],axis=0)
    #tmp = stats.zscore(tmp)
    tmp2 = np.reshape(tmp,(4,8))
    xmax2,xmin2 = tmp.max(),tmp.min()
    plt.subplot(312)    
    fig2=plt.imshow(tmp2)   
    plt.axis('off')      
    plt.colorbar()
        
    # getting hand knob activation
    tmp = delta_recon_batch[query]
    a = np.std(tmp,axis=0)
    var_batch = np.append(var_batch,a)
    tmp = tmp[:,hand_channels] # get the hand knob channels 30,31,22,23,15
    hand_knob_act_batch = np.append(hand_knob_act_batch,tmp)        
    # plotting
    tmp = np.mean(delta_recon_batch[query],axis=0) #first 8 nos form first row, etc
    #tmp = stats.zscore(tmp)
    tmp3 = np.reshape(tmp,(4,8)) # first 8 values form the first row of the grid and so on
    xmax3,xmin3 = tmp.max(),tmp.min()
    plt.subplot(313)    
    fig3=plt.imshow(tmp3)
    plt.axis('off')    
    plt.colorbar()
    
    a = np.corrcoef(np.ndarray.flatten(tmp1),np.ndarray.flatten(tmp2))[0,1]
    b = np.corrcoef(np.ndarray.flatten(tmp1),np.ndarray.flatten(tmp3))[0,1]
    c = np.corrcoef(np.ndarray.flatten(tmp2),np.ndarray.flatten(tmp3))[0,1]
    # a = np.dot(tmp1.flatten(),tmp2.flatten())
    # b = np.dot(tmp1.flatten(),tmp3.flatten())
    # c = np.dot(tmp2.flatten(),tmp3.flatten())
    corr_coef_delta = np.append(corr_coef_delta,[a,b,c])
    
    xmax = np.array([xmax1,xmax2,xmax3])
    xmin = np.array([xmin1,xmin2,xmin3])
    fig1.set_clim(xmin.min(),xmax.max())
    fig2.set_clim(xmin.min(),xmax.max())
    fig3.set_clim(xmin.min(),xmax.max())
    
    image_format = 'svg' # e.g .png, .svg, etc.
    image_name = actions[query] + '_delta_Day1.svg'
    fig.savefig(image_name, format=image_format, dpi=300)

# plotting hand knob activation
x= [hand_knob_act_imag.flatten()  ,hand_knob_act_online.flatten(), hand_knob_act_batch.flatten() ]
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(x,showfliers=False)
plt.xticks(ticks=[1,2,3],labels=('Open Loop','Init. Seed','Batch'),**hfont)
plt.ylabel('delta Hand knob activity',**hfont)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Delta_Hand_Knob_Activation.svg'
fig.savefig(image_name, format=image_format, dpi=300)

# plotting variance boxplots 
x= [var_imag  ,var_online, var_batch]
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
#plt.xlim((0,0.14))
#plt.xlabel('Delta Std. Deviation',**hfont)
#plt.ylabel('Count',**hfont)
plt.tick_params(labelleft=False,labelbottom=False)
#plt.legend()
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Delta_Variance.svg'
fig.savefig(image_name, format=image_format, dpi=300)

# beta
actions =['Rt Thumb','Left Leg','Lt Thumb','Head','Lips','Tongue','Both Middle Finger']
corr_coef_beta = np.array([])
hand_knob_act_imag=np.array([])
hand_knob_act_online=np.array([])
hand_knob_act_batch=np.array([])
var_imag=np.array([])
var_online=np.array([])
var_batch=np.array([])
hand_channels = np.array([23,31])
for query in np.arange(7):        
    
    # getting hand knob activation
    tmp = beta_recon_imag[query]
    a = np.std(tmp,axis=0)
    var_imag = np.append(var_imag,a)
    tmp = tmp[:,hand_channels] # get the hand knob channels 30,31,22,23,15
    hand_knob_act_imag = tmp    
    # plotting
    fig=plt.figure()
    plt.suptitle(actions[query])
    tmp = np.mean(beta_recon_imag[query],axis=0)
    #tmp = stats.zscore(tmp)
    tmp1 = np.reshape(tmp,(4,8))
    xmax1,xmin1 = tmp.max(),tmp.min()
    plt.subplot(311)
    fig1=plt.imshow(tmp1)
    plt.axis('off')    
    plt.colorbar()
        
    # getting hand knob activation
    tmp = beta_recon_online[query]
    a = np.std(tmp,axis=0)
    var_online = np.append(var_online,a)
    tmp = tmp[:,hand_channels] # get the hand knob channels 30,31,22,23,15
    hand_knob_act_online = tmp        
    # plotting
    tmp = np.mean(beta_recon_online[query],axis=0)
    #tmp = stats.zscore(tmp)
    tmp2 = np.reshape(tmp,(4,8))
    xmax2,xmin2 = tmp.max(),tmp.min()
    plt.subplot(312)    
    fig2=plt.imshow(tmp2)   
    plt.axis('off')      
    plt.colorbar()
        
    # getting hand knob activation
    tmp = beta_recon_batch[query]
    a = np.std(tmp,axis=0)
    var_batch = np.append(var_batch,a)
    tmp = tmp[:,hand_channels] # get the hand knob channels 30,31,22,23,15
    hand_knob_act_batch = tmp        
    # plotting
    tmp = np.mean(beta_recon_batch[query],axis=0) #first 8 nos form first row, etc
    #tmp = stats.zscore(tmp)
    tmp3 = np.reshape(tmp,(4,8)) # first 8 values form the first row of the grid and so on
    xmax3,xmin3 = tmp.max(),tmp.min()
    plt.subplot(313)    
    fig3=plt.imshow(tmp3)
    plt.axis('off')    
    plt.colorbar()
    
    a = np.corrcoef(np.ndarray.flatten(tmp1),np.ndarray.flatten(tmp2))[0,1] #normalized dot product
    b = np.corrcoef(np.ndarray.flatten(tmp1),np.ndarray.flatten(tmp3))[0,1]
    c = np.corrcoef(np.ndarray.flatten(tmp2),np.ndarray.flatten(tmp3))[0,1]
    # a = np.dot(tmp1.flatten(),tmp2.flatten()) # just the dot product
    # b = np.dot(tmp1.flatten(),tmp3.flatten())
    # c = np.dot(tmp2.flatten(),tmp3.flatten())
    corr_coef_beta = np.append(corr_coef_beta,[a,b,c])
    
    xmax = np.array([xmax1,xmax2,xmax3])
    xmin = np.array([xmin1,xmin2,xmin3])
    fig1.set_clim(xmin.min(),xmax.max())
    fig2.set_clim(xmin.min(),xmax.max())
    fig3.set_clim(xmin.min(),xmax.max())
    
    image_format = 'svg' # e.g .png, .svg, etc.
    image_name = actions[query] + '_delta_Day1.svg'
    fig.savefig(image_name, format=image_format, dpi=300)

# plotting hand knob activation
x= [hand_knob_act_imag.flatten()  ,hand_knob_act_online.flatten(), hand_knob_act_batch.flatten() ]
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(x,showfliers=False)
plt.xticks(ticks=[1,2,3],labels=('Open Loop','Init. Seed','Batch'),**hfont)
plt.ylabel('beta Hand knob activity',**hfont)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Beta_Hand_Knob_Activation.svg'
fig.savefig(image_name, format=image_format, dpi=300)

# plotting variance boxplots 
x= [var_imag  ,var_online, var_batch]
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(x,showfliers=False)
plt.xticks(ticks=[1,2,3],labels=('Imagined','Online','Batch'),**hfont)
plt.ylabel('beta Variance',**hfont)
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
#plt.xlim((0,0.14))
#plt.xlabel('Beta Std. Deviation',**hfont)
#plt.ylabel('Count',**hfont)
#plt.legend()
plt.tick_params(labelleft=False,labelbottom=False)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Beta_Variance.svg'
fig.savefig(image_name, format=image_format, dpi=300)

# plotting all spatial correlations together as boxplot
x= [corr_coef_delta  ,corr_coef_beta, corr_coef_hg]
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(x,showfliers=False)
plt.xticks(ticks=[1,2,3],labels=('Delta','Beta','hG'),**hfont)
#plt.ylabel('Norm. Spatial Correlation',**hfont)
plt.ylim((0.4,1))
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Norm. Spatial_Correlation.svg'
fig.savefig(image_name, format=image_format, dpi=300)



# plotting all spatial correlations together as boxplot over all days and simulations
corr_coef_delta = np.mean(delta_spatial_corr_days[:,:,:],axis=0).flatten()
corr_coef_beta = np.mean(beta_spatial_corr_days[:,:,:],axis=0).flatten()
corr_coef_hg = np.mean(hg_spatial_corr_days[:,:,:],axis=0).flatten()
x= [np.square(corr_coef_delta)  ,np.square(corr_coef_beta), np.square(corr_coef_hg)]
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(x,showfliers=False)
plt.xticks(ticks=[1,2,3],labels=('Delta','Beta','hG'),**hfont)
plt.ylabel('Norm. Spatial Correlation',**hfont)
plt.ylim((0.8,1.01))
plt.tick_params(labelleft=False,labelbottom=False)
plt.show()
plt.ylabel('')
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Norm. Spatial_Correlation_B1_New.svg'
fig.savefig(image_name, format=image_format, dpi=300)


x1 = np.concatenate((corr_coef_delta[:,None],corr_coef_beta[:,None],corr_coef_hg[:,None]))
x1b = stats.bootstrap([x1,],np.mean)


# labels = np.argmax(Yimagined,axis=1)
# idx = np.where(labels==5)[0]
# tmp = condn_data_imagined[idx,:]
# hgidx = np.arange(2,96,3)
# tmp = tmp[:,hgidx]
# tmp = np.mean(tmp,axis=0)
# #tmp = tmp/lin.norm(tmp)
# tmp = np.reshape(tmp,(4,8))
# xmax1,xmin1 = tmp.max(),tmp.min()
# plt.figure()
# fig1=plt.imshow(tmp)
# plt.colorbar()

# labels = np.argmax(Yonline,axis=1)
# idx = np.where(labels==5)[0]
# tmp = condn_data_online[idx,:]
# hgidx = np.arange(2,96,3)
# tmp = tmp[:,hgidx]
# tmp = np.mean(tmp,axis=0)
# tmp = np.reshape(tmp,(4,8))
# #tmp = tmp/lin.norm(tmp)
# xmax2,xmin2 = tmp.max(),tmp.min()
# plt.figure()
# fig2=plt.imshow(tmp)
# plt.colorbar()
# xmax = np.array([xmax1,xmax2])
# xmin = np.array([xmin1,xmin2])
# fig1.set_clim(xmin.min(),xmax.max())
# fig2.set_clim(xmin.min(),xmax.max())


#%% plotting channel variances after the averaging over iterations and days (MAIN MAIN)
# high gamma
var_imag = np.mean(hg_recon_imag_var_days,axis=0)
var_online = np.mean(hg_recon_online_var_days,axis=0)
var_batch = np.mean(hg_recon_batch_var_days,axis=0)
#x= [var_imag.flatten()  , var_online.flatten(), var_batch.flatten()]
x= [np.mean(var_imag,axis=1),np.mean(var_online,axis=1),np.mean(var_batch,axis=1)]
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
plt.xlim((0,0.055))
#plt.xlabel('Beta Std. Deviation',**hfont)
#plt.ylabel('Count',**hfont)
#plt.legend()
plt.tick_params(labelleft=False,labelbottom=False)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'hG_Variance_Overall_AcrossChannels.svg'
fig.savefig(image_name, format=image_format, dpi=300)
print(stats.ks_2samp(x[0],x[1]))
print(stats.ks_2samp(x[0],x[2]))
print(stats.ks_2samp(x[1],x[2]))

tmp = np.concatenate((x[0][:,None], x[1][:,None], x[2][:,None]),axis=1)

#beta
var_imag = np.mean(beta_recon_imag_var_days,axis=0)
var_online = np.mean(beta_recon_online_var_days,axis=0)
var_batch = np.mean(beta_recon_batch_var_days,axis=0)
#x= [var_imag.flatten()  , var_online.flatten(), var_batch.flatten()]
x= [np.mean(var_imag,axis=1),np.mean(var_online,axis=1),np.mean(var_batch,axis=1)]
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(x,showfliers=False)
plt.xticks(ticks=[1,2,3],labels=('Imagined','Online','Batch'),**hfont)
plt.ylabel('beta Variance',**hfont)
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
plt.xlim((0.00,.016))
#plt.xlabel('Beta Std. Deviation',**hfont)
#plt.ylabel('Count',**hfont)
#plt.legend()
plt.tick_params(labelleft=False,labelbottom=False)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'beta_Variance_Overall_Acrosschannels_New.svg'
fig.savefig(image_name, format=image_format, dpi=300)
print(stats.ks_2samp(x[0],x[1]))
print(stats.ks_2samp(x[0],x[2]))
print(stats.ks_2samp(x[1],x[2]))

#delta
var_imag = np.mean(delta_recon_imag_var_days,axis=0)
var_online = np.mean(delta_recon_online_var_days,axis=0)
var_batch = np.mean(delta_recon_batch_var_days,axis=0)
#x= [var_imag.flatten()  , var_online.flatten(), var_batch.flatten()]
x= [np.mean(var_imag,axis=1),np.mean(var_online,axis=1),np.mean(var_batch,axis=1)]
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(x,showfliers=False)
plt.xticks(ticks=[1,2,3],labels=('Imagined','Online','Batch'),**hfont)
plt.ylabel('delta Variance',**hfont)
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
plt.xlim((0,0.035))
#plt.xlabel('Beta Std. Deviation',**hfont)
#plt.ylabel('Count',**hfont)
#plt.legend()
plt.tick_params(labelleft=False,labelbottom=False)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'delta_Variance_Overall_AcrossChannels_New.svg'
fig.savefig(image_name, format=image_format, dpi=300)
print(stats.ks_2samp(x[0],x[1]))
print(stats.ks_2samp(x[0],x[2]))
print(stats.ks_2samp(x[1],x[2]))

#%%########### PLOTTING FOR B2 #####################
# OVERALL VARIANCE IN LATENT SPACE 
N=2
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(6)+1
X=np.arange(6)+1
# imagined 
tmp1 = np.median(var_overall_imagined_days,axis=0)
tmp_boot, tmp_boot_std = median_bootstrap(var_overall_imagined_days,1000)
tmp1b = tmp_boot_std/1#sqrt(tmp_main.shape[0])
tmp1 = np.insert(tmp1,0,tmp1[0],axis=0)
tmp1b = np.insert(tmp1b,0,tmp1b[0],axis=0)
tmp1 = np.convolve(tmp1, np.ones(N)/N, mode='same')
tmp1 = tmp1[1:]
tmp1b = tmp1b[1:]
plt.plot(X,tmp1,color="black",label = 'Imagined')
plt.fill_between(X, tmp1-tmp1b, tmp1+tmp1b,color="black",alpha=0.2)
# online
tmp2 = np.median(var_overall_online_days,axis=0)
tmp_boot, tmp_boot_std = median_bootstrap(var_overall_online_days,1000)
tmp2b = tmp_boot_std/1#sqrt(tmp_main.shape[0])
tmp2 = np.insert(tmp2,0,tmp2[0],axis=0)
tmp2b = np.insert(tmp2b,0,tmp2b[0],axis=0)
tmp2 = np.convolve(tmp2, np.ones(N)/N, mode='same')
tmp2b = np.convolve(tmp2b, np.ones(N)/N, mode='same')
tmp2 = tmp2[1:]
tmp2b = tmp2b[1:]
plt.plot(X,tmp2,color="blue",label = 'Online')
plt.fill_between(X, tmp2-tmp2b, tmp2+tmp2b,color="blue",alpha=0.2)
# batch
tmp3 = np.median(var_overall_batch_days[:,1:5],axis=0)
tmp_boot, tmp_boot_std = median_bootstrap(var_overall_batch_days[:,1:5],1000)
tmp3b = tmp_boot_std/1#sqrt(tmp_main.shape[0])
tmp3 = np.insert(tmp3,0,tmp3[0],axis=0)
tmp3b = np.insert(tmp3b,0,tmp3b[0],axis=0)
tmp3 = np.convolve(tmp3, np.ones(N)/N, mode='same')
tmp3b = np.convolve(tmp3b, np.ones(N)/N, mode='same')
tmp3 = tmp3[1:]
tmp3b = tmp3b[1:]
plt.plot(X[1:5],tmp3,color="red",label = 'Batch')
plt.fill_between(X[1:5], tmp3-tmp3b, tmp3+tmp3b,color="red",alpha=0.2)
plt.legend()
plt.xlabel('Days',**hfont)
plt.ylabel('Overall Latent Variance',**hfont)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Overall_Variance_Latent_Days.svg'
fig.savefig(image_name, format=image_format, dpi=300)

tmp3=np.append(tmp3,(np.median(tmp3),np.median(tmp3)))
tmp = np.concatenate((tmp1[:,None],tmp2[:,None],tmp3[:,None]),axis=1) # just the mean
tmp = np.concatenate((var_overall_imagined_days.flatten()[:,None],
                   var_overall_online_days.flatten()[:,None],
                   var_overall_batch_days.flatten()[:,None]),axis=1) # all the data

# acrosall boxplots 
# tmp = np.concatenate((np.ndarray.flatten(var_overall_imagined_days)[:,None],
#                       np.ndarray.flatten(var_overall_online_days)[:,None],
#                       np.ndarray.flatten(var_overall_batch_days)[:,None]),axis=1)
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot((tmp),showfliers=True)
#plt.xticks(ticks=[1,2,3],labels=('Imagined','Online','Batch'),**hfont)
#plt.ylabel('Overall Latent Variance',**hfont)
plt.xticks(ticks=[1,2,3],labels='')
plt.yticks(ticks=[0, 0.1 ,0.2],labels='')
# add scatter
x=np.ones((tmp.shape[0],1))+0.1*rnd.randn(tmp.shape[0])[:,None]
plt.scatter(x, np.log(tmp[:,0]))
plt.scatter(x+1, np.log(tmp[:,1]))
plt.scatter(x+2, np.log(tmp[:,2]))
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Overall_Variance_Latent_Boxplot_B2_withData.svg'
fig.savefig(image_name, format=image_format, dpi=300)
print(np.mean(tmp,axis=0))

# doing a repeated measures anova in the latent space
# have to organize the data accordingly
data_rm = np.empty([0,3])
for i in np.arange(tmp.shape[0]):
    a = tmp[i,:]
    subj = np.repeat(i,tmp.shape[1])+1
    condn  = [1,2,3]
    tmp_data = np.empty([3,3])
    tmp_data[:,0] =  subj
    tmp_data[:,1] =  condn
    tmp_data[:,2] =  a
    data_rm = np.concatenate((data_rm,tmp_data),axis=0)
    
df = pd.DataFrame({'subject': data_rm[:,0],
                   'decoder': data_rm[:,1],
                   'variance':data_rm[:,2]})


print(AnovaRM(data=df, depvar='variance', subject='subject', within=['decoder']).fit())


# rm anova to see if there is a difference between latent mahab distances between 
# OL and CL1 and CL2



# MAHAB DISTANCES  (MAIN)
num_days=6
N=2
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(num_days)+1
X=np.arange(num_days)+1
# imagined 
tmp_main = np.squeeze(np.mean(mahab_distances_imagined_days,1))
tmp1 = np.mean(tmp_main,axis=0)
tmp1b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp1 = tmp1[1:5] # only taking days with batch
tmp1b = tmp1b[1:5] # only taking days with batch
tmp1 = np.insert(tmp1,-1,tmp1[-1],axis=0)
tmp1b = np.insert(tmp1b,-1,tmp1b[-1],axis=0)
tmp1 = np.convolve(tmp1, np.ones(N)/N, mode='same')
tmp1b = np.convolve(tmp1b, np.ones(N)/N, mode='same')
tmp1 = tmp1[1:]
tmp1b = tmp1b[1:]
plt.plot(X[1:5],tmp1,color="black",label = 'Imagined')
plt.fill_between(X[1:5], tmp1-tmp1b, tmp1+tmp1b,color="black",alpha=0.2)
# online
tmp_main = np.squeeze(np.mean(mahab_distances_online_days,1))
tmp2 = np.mean(tmp_main,axis=0)
tmp2b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp2 = tmp2[1:5] # only taking days with batch
tmp2b = tmp2b[1:5] # only taking days with batch
tmp2 = np.insert(tmp2,-1,tmp2[-1],axis=0)
tmp2b = np.insert(tmp2b,-1,tmp2b[-1],axis=0)
tmp2 = np.convolve(tmp2, np.ones(N)/N, mode='same')
tmp2b = np.convolve(tmp2b, np.ones(N)/N, mode='same')
tmp2 = tmp2[1:]
tmp2b = tmp2b[1:]
plt.plot(X[1:5],tmp2,color="blue",label = 'Online')
plt.fill_between(X[1:5], tmp2-tmp2b, tmp2+tmp2b,color="blue",alpha=0.2)
# batch
tmp_main = np.squeeze(np.mean(mahab_distances_batch_days,1))
tmp3 = np.mean(tmp_main,axis=0)
tmp3b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp3 = tmp3[1:5] #get only  days when data available
tmp3b = tmp3b[1:5] #get only days when data available
tmp3 = np.insert(tmp3,-1,tmp3[-1],axis=0)
tmp3b = np.insert(tmp3b,-1,tmp3b[-1],axis=0)
tmp3 = np.convolve(tmp3, np.ones(N)/N, mode='same')
tmp3b = np.convolve(tmp3b, np.ones(N)/N, mode='same')
tmp3 = tmp3[1:]
tmp3b = tmp3b[1:]
plt.plot(X[1:5],tmp3,color="red",label = 'Batch')
plt.fill_between(X[1:5], tmp3-tmp3b, tmp3+tmp3b,color="red",alpha=0.2)
plt.legend(loc='upper left')
plt.xlabel('Days',**hfont)
plt.ylabel('Mahalanobis Distances',**hfont)
plt.show()
plt.xlim((2,5))
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Mean_Mahalanobis_dist_Days_withBatch.svg'
fig.savefig(image_name, format=image_format, dpi=300)

#tmp3 = np.append(tmp3,(np.median(tmp3),np.median(tmp3)))
tmp = np.concatenate((tmp1[:,None],tmp2[:,None],tmp3[:,None]),axis=1)
# tmp = np.concatenate((np.ndarray.flatten(var_overall_imagined_days)[:,None],
#                       np.ndarray.flatten(var_overall_online_days)[:,None],
#                       np.ndarray.flatten(var_overall_batch_days)[:,None]),axis=1)
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(tmp,whis=2,showfliers=False)
plt.xticks(ticks=[1,2,3],labels=('Imagined','Online','Batch'),**hfont)
plt.ylabel('Mahalanobis Distances',**hfont)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Mahab_Dist_Boxplot_B2.svg'
fig.savefig(image_name, format=image_format, dpi=300)
print(np.mean(tmp,axis=0))


# plotting distance between means (no mahab) over days (MAIN) with median
N=2
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(num_days)+1
X=np.arange(num_days)+1
# imagined 
tmp_main = np.squeeze(np.mean(mean_distances_imagined_days,1))
tmp1 = np.mean(tmp_main,axis=0)
tmp1b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp1 = np.insert(tmp1,0,tmp1[0],axis=0)
tmp1b = np.insert(tmp1b,0,tmp1b[0],axis=0)
tmp1 = np.convolve(tmp1, np.ones(N)/N, mode='same')
tmp1 = tmp1[1:]
tmp1b = tmp1b[1:]
plt.plot(X,tmp1,color="black",label = 'Imagined')
plt.fill_between(X, tmp1-tmp1b, tmp1+tmp1b,color="black",alpha=0.2)
# online
tmp_main = np.squeeze(np.mean(mean_distances_online_days,1))
tmp2 = np.mean(tmp_main,axis=0)
tmp2b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp2 = np.insert(tmp2,0,tmp2[0],axis=0)
tmp2b = np.insert(tmp2b,0,tmp2b[0],axis=0)
tmp2 = np.convolve(tmp2, np.ones(N)/N, mode='same')
tmp2b = np.convolve(tmp2b, np.ones(N)/N, mode='same')
tmp2 = tmp2[1:]
tmp2b = tmp2b[1:]
plt.plot(X,tmp2,color="blue",label = 'Online')
plt.fill_between(X, tmp2-tmp2b, tmp2+tmp2b,color="blue",alpha=0.2)
# batch
tmp_main = np.squeeze(np.mean(mean_distances_batch_days,1))
tmp3 = np.mean(tmp_main,axis=0)
tmp3b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp3 = tmp3[1:5] #get only  days when data available
tmp3b = tmp3b[1:5] #get only days when data available
tmp3 = np.insert(tmp3,0,tmp3[0],axis=0)
tmp3b = np.insert(tmp3b,0,tmp3b[0],axis=0)
tmp3 = np.convolve(tmp3, np.ones(N)/N, mode='same')
tmp3b = np.convolve(tmp3b, np.ones(N)/N, mode='same')
tmp3 = tmp3[1:]
tmp3b = tmp3b[1:]
plt.plot(X[1:5],tmp3,color="red",label = 'Batch')
plt.fill_between(X[1:5], tmp3-tmp3b, tmp3+tmp3b,color="red",alpha=0.2)
plt.legend(loc='upper left')
plt.xlabel('Days',**hfont)
plt.ylabel('Mean Distances',**hfont)
plt.show()
plt.xlim((2,5))
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Mean_Mahalanobis_dist_Days_withBatch.svg'
fig.savefig(image_name, format=image_format, dpi=300)

tmp3 = np.append(tmp3,(np.median(tmp3),np.median(tmp3)))
tmp = np.concatenate((tmp1[:,None],tmp2[:,None],tmp3[:,None]),axis=1)
# tmp = np.concatenate((np.ndarray.flatten(var_overall_imagined_days)[:,None],
#                       np.ndarray.flatten(var_overall_online_days)[:,None],
#                       np.ndarray.flatten(var_overall_batch_days)[:,None]),axis=1)
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(tmp,whis=2,showfliers=False)
plt.xticks(ticks=[1,2,3],labels=('Imagined','Online','Batch'),**hfont)
plt.ylabel('Mean Distances',**hfont)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Mahab_Dist_Boxplot_B2.svg'
fig.savefig(image_name, format=image_format, dpi=300)
print(np.mean(tmp,axis=0))


# plotting mean centroid variances over days  (MAIN)
N=2
num_days=6
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(num_days)+1
X=np.arange(num_days)+1
# imagined 
tmp_main = np.squeeze(np.median(var_imagined_days,axis=1))
tmp1 = np.median(tmp_main,axis=0)
tmp1b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp1 = np.insert(tmp1,0,tmp1[0],axis=0)
tmp1b = np.insert(tmp1b,0,tmp1b[0],axis=0)
tmp1 = np.convolve(tmp1, np.ones(N)/N, mode='same')
tmp1 = tmp1[1:]
tmp1b = tmp1b[1:]
plt.plot(X,tmp1,color="black",label = 'Imagined')
plt.fill_between(X, tmp1-tmp1b, tmp1+tmp1b,color="black",alpha=0.2)
# online
tmp_main = np.squeeze(np.median(var_online_days,axis=1))
tmp2 = np.median(tmp_main,axis=0)
tmp2b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp2 = np.insert(tmp2,0,tmp2[0],axis=0)
tmp2b = np.insert(tmp2b,0,tmp2b[0],axis=0)
tmp2 = np.convolve(tmp2, np.ones(N)/N, mode='same')
tmp2b = np.convolve(tmp2b, np.ones(N)/N, mode='same')
tmp2 = tmp2[1:]
tmp2b = tmp2b[1:]
plt.plot(X,tmp2,color="blue",label = 'Online')
plt.fill_between(X, tmp2-tmp2b, tmp2+tmp2b,color="blue",alpha=0.2)
# batch
tmp_main = np.squeeze(np.median(var_batch_days,axis=1))
tmp3 = np.median(tmp_main,axis=0)
tmp3b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp3 = tmp3[1:5] #get only  days when data available
tmp3b = tmp3b[1:5] #get only days when data available
tmp3 = np.insert(tmp3,0,tmp3[0],axis=0)
tmp3b = np.insert(tmp3b,0,tmp3b[0],axis=0)
tmp3 = np.convolve(tmp3, np.ones(N)/N, mode='same')
tmp3b = np.convolve(tmp3b, np.ones(N)/N, mode='same')
tmp3 = tmp3[1:]
tmp3b = tmp3b[1:]
plt.plot(X[1:5],tmp3,color="red",label = 'Batch')
plt.fill_between(X[1:5], tmp3-tmp3b, tmp3+tmp3b,color="red",alpha=0.2)
plt.legend()
plt.xlabel('Days',**hfont)
plt.ylabel('Overall Latent Variance',**hfont)
plt.xlim((2,5))
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Overall_Variance_Latent_Days.svg'
fig.savefig(image_name, format=image_format, dpi=300)


tmp = np.concatenate((tmp1[1:5][:,None],tmp2[1:5][:,None],tmp3[:,None]),axis=1)
# tmp = np.concatenate((np.ndarray.flatten(var_overall_imagined_days)[:,None],
#                       np.ndarray.flatten(var_overall_online_days)[:,None],
#                       np.ndarray.flatten(var_overall_batch_days)[:,None]),axis=1)
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot((tmp),whis=2,showfliers=True)
#plt.xticks(ticks=[1,2,3],labels=('Imagined','Online','Batch'),**hfont)
plt.xticks(ticks=[1,2,3],labels='',**hfont)
plt.ylabel('',**hfont)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'B2_Variance_Latent_Centroids.svg'
fig.savefig(image_name, format=image_format, dpi=300)
print(np.mean(tmp,axis=0))


# plotting the same but now across days without averaging
tmp_main = np.squeeze(np.mean(var_imagined_days,axis=1))
tmp_main = tmp_main[:,1:5]
tmp1 = np.ndarray.flatten(tmp_main)
tmp_main = np.squeeze(np.mean(var_online_days,axis=1))
tmp_main = tmp_main[:,1:5]
tmp2 = np.ndarray.flatten(tmp_main)
tmp_main = np.squeeze(np.mean(var_batch_days,axis=1))
tmp_main = tmp_main[:,1:5]
tmp3 = np.ndarray.flatten(tmp_main)
tmp = np.concatenate((tmp1[:,None],tmp2[:,None],tmp3[:,None]),axis=1)


#tmp=np.log(tmp)
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot((tmp),whis=2,showfliers=False)
plt.xticks(ticks=[1,2,3],labels=('Imagined','Online','Batch'),**hfont)
plt.ylabel('Latent variance',**hfont)
plt.show()
plt.xticks(ticks=[1,2,3],labels='',**hfont)
plt.yticks(ticks=[0.005,0.015,0.025],labels='',**hfont)
plt.ylabel('',**hfont)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'New_B2_Variance_Latent_Centroids.svg'
fig.savefig(image_name, format=image_format, dpi=300)

print(np.mean(tmp,axis=0))
print(stats.ttest_rel(tmp[:,0],tmp[:,1]))
print(stats.ttest_rel(tmp[:,0],tmp[:,2]))
print(stats.ttest_rel(tmp[:,1],tmp[:,2]))

#as scatter plots
tmp1=np.log(tmp1)
tmp2=np.log(tmp2)
tmp3=np.log(tmp3)

idx = np.where(tmp1>-1)
tmp1=tmp1[idx]
idx = np.where(tmp2>-1)
tmp2=tmp2[idx]
idx = np.where(tmp3>-1)
tmp3=tmp3[idx]

fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
x1=np.ones((tmp1.shape[0],1)) + 0.05*rnd.randn(tmp1.shape[0])[:,None]
x2=2*np.ones((tmp2.shape[0],1)) + 0.05*rnd.randn(tmp2.shape[0])[:,None]
x3=3*np.ones((tmp3.shape[0],1)) + 0.05*rnd.randn(tmp3.shape[0])[:,None]
plt.scatter(x1, tmp1,s=5,c=[.2 ,.2 ,.6])
plt.scatter(x2, tmp2,s=5,c=[.2 ,.2 ,.6])
plt.scatter(x3, tmp3,s=5,c=[.2 ,.2 ,.6])
plt.hlines(np.mean(tmp1),0.8,1.2,colors='black',linewidth=3)
plt.hlines(np.mean(tmp2),1.8,2.2,colors='black',linewidth=3)
plt.hlines(np.mean(tmp3),2.8,3.2,colors='black',linewidth=3)
plt.xticks([1,2,3])
plt.ylim((-1,5))
#plt.yticks([0.23,0.25,0.27])
plt.show()


#%% LOOKING AT THE CHANNEL VARIANCES AFTER ITERATING OVER DAYS AND NETWORKS
# high gamma
var_imag = np.mean(hg_recon_imag_var_days[:,:,1:5],axis=0)
var_online = np.mean(hg_recon_online_var_days[:,:,1:5],axis=0)
var_batch = np.mean(hg_recon_batch_var_days[:,:,1:5],axis=0)
#x= [var_imag.flatten()  , var_online.flatten(), var_batch.flatten()]
x= [np.mean(var_imag,axis=1),np.mean(var_online,axis=1),np.mean(var_batch,axis=1)]
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
plt.xlim((0.000,0.025))
#plt.xlabel('Beta Std. Deviation',**hfont)
#plt.ylabel('Count',**hfont)
#plt.legend()
plt.tick_params(labelleft=False,labelbottom=False)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'hG_Variance_Overall_AcrossChannels_B2_New.svg'
fig.savefig(image_name, format=image_format, dpi=300)
#ks test
print(stats.ks_2samp(x[0],x[1]))
print(stats.ks_2samp(x[0],x[2]))
print(stats.ks_2samp(x[1],x[2]))

#beta
var_imag = np.mean(beta_recon_imag_var_days[:,:,1:5],axis=0)
var_online = np.mean(beta_recon_online_var_days[:,:,1:5],axis=0)
var_batch = np.mean(beta_recon_batch_var_days[:,:,1:5],axis=0)
#x= [var_imag.flatten()  , var_online.flatten(), var_batch.flatten()]
x= [np.mean(var_imag,axis=1),np.mean(var_online,axis=1),np.mean(var_batch,axis=1)]
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(x,showfliers=False)
plt.xticks(ticks=[1,2,3],labels=('Imagined','Online','Batch'),**hfont)
plt.ylabel('beta Variance',**hfont)
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
plt.xlim((0.00,.075))
#plt.xlabel('Beta Std. Deviation',**hfont)
#plt.ylabel('Count',**hfont)
#plt.legend()
plt.tick_params(labelleft=False,labelbottom=False)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'beta_Variance_Overall_Acrosschannels_B2_New.svg'
fig.savefig(image_name, format=image_format, dpi=300)
#ks test
print(stats.ks_2samp(x[0],x[1]))
print(stats.ks_2samp(x[0],x[2]))
print(stats.ks_2samp(x[1],x[2]))

#delta
var_imag = np.mean(delta_recon_imag_var_days[:,:,1:5],axis=0)
var_online = np.mean(delta_recon_online_var_days[:,:,1:5],axis=0)
var_batch = np.mean(delta_recon_batch_var_days[:,:,1:5],axis=0)
#x= [var_imag.flatten()  , var_online.flatten(), var_batch.flatten()]
x= [np.mean(var_imag,axis=1),np.mean(var_online,axis=1),np.mean(var_batch,axis=1)]
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(x,showfliers=False)
plt.xticks(ticks=[1,2,3],labels=('Imagined','Online','Batch'),**hfont)
plt.ylabel('delta Variance',**hfont)
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
plt.xlim((0,0.12))
#plt.xlabel('Beta Std. Deviation',**hfont)
#plt.ylabel('Count',**hfont)
#plt.legend()
plt.tick_params(labelleft=False,labelbottom=False)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'delta_Variance_Overall_AcrossChannels_B2_New.svg'
fig.savefig(image_name, format=image_format, dpi=300)
#ks test
print(stats.ks_2samp(x[0],x[1]))
print(stats.ks_2samp(x[0],x[2]))
print(stats.ks_2samp(x[1],x[2]))

# plotting spatial maps
# high gamma
actions =['Rt Thumb','Both Feet','Lt Thumb','Head']
corr_coef_hg = np.array([])
hand_knob_act_imag=np.array([])
hand_knob_act_online=np.array([])
hand_knob_act_batch=np.array([])
var_imag=np.array([])
var_online=np.array([])
var_batch=np.array([])
hand_channels = np.array([23,31])
for query in np.arange(4):        
    
    # getting hand knob activation
    tmp = hg_recon_imag[query]
    a = np.std(tmp,axis=0)
    var_imag = np.append(var_imag,a)
    tmp = tmp[:,hand_channels] # get the hand knob channels 30,31,22,23,15
    hand_knob_act_imag = tmp    
    # plotting
    fig=plt.figure()
    plt.suptitle(actions[query])
    tmp = np.mean(hg_recon_imag[query],axis=0)
    #tmp = stats.zscore(tmp)
    tmp1 = np.reshape(tmp,(4,8))
    xmax1,xmin1 = tmp.max(),tmp.min()
    plt.subplot(311)
    fig1=plt.imshow(tmp1)
    plt.axis('off')    
    plt.colorbar()
        
    # getting hand knob activation
    tmp = hg_recon_online[query]
    a = np.std(tmp,axis=0)
    var_online = np.append(var_online,a)
    tmp = tmp[:,hand_channels] # get the hand knob channels 30,31,22,23,15
    hand_knob_act_online = tmp        
    # plotting
    tmp = np.mean(hg_recon_online[query],axis=0)
    #tmp = stats.zscore(tmp)
    tmp2 = np.reshape(tmp,(4,8))
    xmax2,xmin2 = tmp.max(),tmp.min()
    plt.subplot(312)    
    fig2=plt.imshow(tmp2)   
    plt.axis('off')      
    plt.colorbar()
        
    # getting hand knob activation
    tmp = hg_recon_batch[query]
    a = np.std(tmp,axis=0)
    var_batch = np.append(var_batch,a)
    tmp = tmp[:,hand_channels] # get the hand knob channels 30,31,22,23,15
    hand_knob_act_batch = tmp        
    # plotting
    tmp = np.mean(hg_recon_batch[query],axis=0) #first 8 nos form first row, etc
    #tmp = stats.zscore(tmp)
    tmp3 = np.reshape(tmp,(4,8)) # first 8 values form the first row of the grid and so on
    xmax3,xmin3 = tmp.max(),tmp.min()
    plt.subplot(313)    
    fig3=plt.imshow(tmp3)
    plt.axis('off')    
    plt.colorbar()
    
    a = np.corrcoef(np.ndarray.flatten(tmp1),np.ndarray.flatten(tmp2))[0,1]
    b = np.corrcoef(np.ndarray.flatten(tmp1),np.ndarray.flatten(tmp3))[0,1]
    c = np.corrcoef(np.ndarray.flatten(tmp2),np.ndarray.flatten(tmp3))[0,1]
    # a = np.dot(tmp1.flatten(),tmp2.flatten())
    # b = np.dot(tmp1.flatten(),tmp3.flatten())
    # c = np.dot(tmp2.flatten(),tmp3.flatten())
    corr_coef_hg = np.append(corr_coef_hg,[a,b,c])
    
    xmax = np.array([xmax1,xmax2,xmax3])
    xmin = np.array([xmin1,xmin2,xmin3])
    fig1.set_clim(xmin.min(),xmax.max())
    fig2.set_clim(xmin.min(),xmax.max())
    fig3.set_clim(xmin.min(),xmax.max())


# plotting hand knob activation
x= [hand_knob_act_imag.flatten()  ,hand_knob_act_online.flatten(), hand_knob_act_batch.flatten() ]
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(x,showfliers=False)
plt.xticks(ticks=[1,2,3],labels=('Open Loop','Init. Seed','Batch'),**hfont)
plt.ylabel('hg Hand knob activity',**hfont)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'hg_Hand_Knob_Activation_B2.svg'
fig.savefig(image_name, format=image_format, dpi=300)


# delta
actions =['Rt Thumb','Both Feet','Lt Thumb','Head']
corr_coef_delta = np.array([])
hand_knob_act_imag=np.array([])
hand_knob_act_online=np.array([])
hand_knob_act_batch=np.array([])
var_imag=np.array([])
var_online=np.array([])
var_batch=np.array([])
hand_channels = np.array([23,31])
for query in np.arange(4):        
    
    # getting hand knob activation
    tmp = delta_recon_imag[query]
    a = np.std(tmp,axis=0)
    var_imag = np.append(var_imag,a)
    tmp = tmp[:,hand_channels] # get the hand knob channels 30,31,22,23,15
    hand_knob_act_imag = np.append(hand_knob_act_imag,tmp)
    # plotting
    fig=plt.figure()
    plt.suptitle(actions[query])
    tmp = np.mean(delta_recon_imag[query],axis=0)
    #tmp = stats.zscore(tmp)
    tmp1 = np.reshape(tmp,(4,8))
    xmax1,xmin1 = tmp.max(),tmp.min()
    plt.subplot(311)
    fig1=plt.imshow(tmp1)
    plt.axis('off')    
    plt.colorbar()
        
    # getting hand knob activation
    tmp = delta_recon_online[query]
    a = np.std(tmp,axis=0)
    var_online = np.append(var_online,a)
    tmp = tmp[:,hand_channels] # get the hand knob channels 30,31,22,23,15
    hand_knob_act_online = np.append(hand_knob_act_online,tmp)
    # plotting
    tmp = np.mean(delta_recon_online[query],axis=0)
    #tmp = stats.zscore(tmp)
    tmp2 = np.reshape(tmp,(4,8))
    xmax2,xmin2 = tmp.max(),tmp.min()
    plt.subplot(312)    
    fig2=plt.imshow(tmp2)   
    plt.axis('off')      
    plt.colorbar()
        
    # getting hand knob activation
    tmp = delta_recon_batch[query]
    a = np.std(tmp,axis=0)
    var_batch = np.append(var_batch,a)
    tmp = tmp[:,hand_channels] # get the hand knob channels 30,31,22,23,15
    hand_knob_act_batch = np.append(hand_knob_act_batch,tmp)        
    # plotting
    tmp = np.mean(delta_recon_batch[query],axis=0) #first 8 nos form first row, etc
    #tmp = stats.zscore(tmp)
    tmp3 = np.reshape(tmp,(4,8)) # first 8 values form the first row of the grid and so on
    xmax3,xmin3 = tmp.max(),tmp.min()
    plt.subplot(313)    
    fig3=plt.imshow(tmp3)
    plt.axis('off')    
    plt.colorbar()
    
    a = np.corrcoef(np.ndarray.flatten(tmp1),np.ndarray.flatten(tmp2))[0,1]
    b = np.corrcoef(np.ndarray.flatten(tmp1),np.ndarray.flatten(tmp3))[0,1]
    c = np.corrcoef(np.ndarray.flatten(tmp2),np.ndarray.flatten(tmp3))[0,1]
    # a = np.dot(tmp1.flatten(),tmp2.flatten())
    # b = np.dot(tmp1.flatten(),tmp3.flatten())
    # c = np.dot(tmp2.flatten(),tmp3.flatten())
    corr_coef_delta = np.append(corr_coef_delta,[a,b,c])
    
    xmax = np.array([xmax1,xmax2,xmax3])
    xmin = np.array([xmin1,xmin2,xmin3])
    fig1.set_clim(xmin.min(),xmax.max())
    fig2.set_clim(xmin.min(),xmax.max())
    fig3.set_clim(xmin.min(),xmax.max())    


# plotting hand knob activation
x= [hand_knob_act_imag.flatten()  ,hand_knob_act_online.flatten(), hand_knob_act_batch.flatten() ]
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(x,showfliers=False)
plt.xticks(ticks=[1,2,3],labels=('Open Loop','Init. Seed','Batch'),**hfont)
plt.ylabel('delta Hand knob activity',**hfont)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Delta_Hand_Knob_Activation_B2.svg'
fig.savefig(image_name, format=image_format, dpi=300)

# beta
actions =['Rt Thumb','Both Feet','Lt Thumb','Head']
corr_coef_beta = np.array([])
hand_knob_act_imag=np.array([])
hand_knob_act_online=np.array([])
hand_knob_act_batch=np.array([])
var_imag=np.array([])
var_online=np.array([])
var_batch=np.array([])
hand_channels = np.array([23,31])
for query in np.arange(4):        
    
    # getting hand knob activation
    tmp = beta_recon_imag[query]
    a = np.std(tmp,axis=0)
    var_imag = np.append(var_imag,a)
    tmp = tmp[:,hand_channels] # get the hand knob channels 30,31,22,23,15
    hand_knob_act_imag = tmp    
    # plotting
    fig=plt.figure()
    plt.suptitle(actions[query])
    tmp = np.mean(beta_recon_imag[query],axis=0)
    #tmp = stats.zscore(tmp)
    tmp1 = np.reshape(tmp,(4,8))
    xmax1,xmin1 = tmp.max(),tmp.min()
    plt.subplot(311)
    fig1=plt.imshow(tmp1)
    plt.axis('off')    
    plt.colorbar()
        
    # getting hand knob activation
    tmp = beta_recon_online[query]
    a = np.std(tmp,axis=0)
    var_online = np.append(var_online,a)
    tmp = tmp[:,hand_channels] # get the hand knob channels 30,31,22,23,15
    hand_knob_act_online = tmp        
    # plotting
    tmp = np.mean(beta_recon_online[query],axis=0)
    #tmp = stats.zscore(tmp)
    tmp2 = np.reshape(tmp,(4,8))
    xmax2,xmin2 = tmp.max(),tmp.min()
    plt.subplot(312)    
    fig2=plt.imshow(tmp2)   
    plt.axis('off')      
    plt.colorbar()
        
    # getting hand knob activation
    tmp = beta_recon_batch[query]
    a = np.std(tmp,axis=0)
    var_batch = np.append(var_batch,a)
    tmp = tmp[:,hand_channels] # get the hand knob channels 30,31,22,23,15
    hand_knob_act_batch = tmp        
    # plotting
    tmp = np.mean(beta_recon_batch[query],axis=0) #first 8 nos form first row, etc
    #tmp = stats.zscore(tmp)
    tmp3 = np.reshape(tmp,(4,8)) # first 8 values form the first row of the grid and so on
    xmax3,xmin3 = tmp.max(),tmp.min()
    plt.subplot(313)    
    fig3=plt.imshow(tmp3)
    plt.axis('off')    
    plt.colorbar()
    
    a = np.corrcoef(np.ndarray.flatten(tmp1),np.ndarray.flatten(tmp2))[0,1] #normalized dot product
    b = np.corrcoef(np.ndarray.flatten(tmp1),np.ndarray.flatten(tmp3))[0,1]
    c = np.corrcoef(np.ndarray.flatten(tmp2),np.ndarray.flatten(tmp3))[0,1]
    # a = np.dot(tmp1.flatten(),tmp2.flatten()) # just the dot product
    # b = np.dot(tmp1.flatten(),tmp3.flatten())
    # c = np.dot(tmp2.flatten(),tmp3.flatten())
    corr_coef_beta = np.append(corr_coef_beta,[a,b,c])
    
    xmax = np.array([xmax1,xmax2,xmax3])
    xmin = np.array([xmin1,xmin2,xmin3])
    fig1.set_clim(xmin.min(),xmax.max())
    fig2.set_clim(xmin.min(),xmax.max())
    fig3.set_clim(xmin.min(),xmax.max())


# plotting hand knob activation
x= [hand_knob_act_imag.flatten()  ,hand_knob_act_online.flatten(), hand_knob_act_batch.flatten() ]
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(x,showfliers=False)
plt.xticks(ticks=[1,2,3],labels=('Open Loop','Init. Seed','Batch'),**hfont)
plt.ylabel('delta Hand knob activity',**hfont)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Beta_Hand_Knob_Activation_B2.svg'
fig.savefig(image_name, format=image_format, dpi=300)


# plotting all spatial correlations together as boxplot after iterating thru simulations
corr_coef_delta = np.mean(delta_spatial_corr_days[:,:,1:5],axis=0).flatten()
corr_coef_beta = np.mean(beta_spatial_corr_days[:,:,1:5],axis=0).flatten()
corr_coef_hg = np.mean(hg_spatial_corr_days[:,:,1:5],axis=0).flatten()
x= [np.square(corr_coef_delta)  ,np.square(corr_coef_beta), np.square(corr_coef_hg)]
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
plt.boxplot(x,showfliers=False)
plt.xticks(ticks=[1,2,3],labels=('Delta','Beta','hG'),**hfont)
plt.ylabel('Norm. Spatial Correlation',**hfont)
plt.ylim((0.6,1.01))
plt.tick_params(labelleft=False,labelbottom=False)
plt.show()
plt.ylabel('')
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Norm. Spatial_Correlation_B2_New.svg'
fig.savefig(image_name, format=image_format, dpi=300)

x1 = np.concatenate((corr_coef_delta[:,None],corr_coef_beta[:,None],corr_coef_hg[:,None]))
x1b = stats.bootstrap([x1,],np.mean)

#%% PLOTTING MAHALANOBIS DISTANCES BETWEEN CKD AND IBID - B1 

# load data for stereotyped movements 
data =np.load('MAIN_MAIN2_B1_NoiseDataAugment_CholIndivFeatEqualSize_pt01Noise_2D_common_Manifold_whole_dataSamples_stats_results_withBatch_Main_withVariance_AndChVars_AndSpatCorr.npz')
mahab_distances_online_days = data.get('mahab_distances_online_days')
mahab_distances_imagined_days = data.get('mahab_distances_imagined_days')
mahab_distances_batch_days = data.get('mahab_distances_batch_days')

N=2
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(10)+1
X=np.arange(10)+1
# imagined 
tmp_main = np.squeeze(np.mean(mahab_distances_imagined_days,axis=1))
tmp1 = np.mean(tmp_main,axis=0)
tmp1b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp1 = np.insert(tmp1,0,tmp1[0],axis=0)
tmp1b = np.insert(tmp1b,0,tmp1b[0],axis=0)
tmp1 = np.convolve(tmp1, np.ones(N)/N, mode='same')
tmp1 = tmp1[1:]
tmp1b = tmp1b[1:]
plt.plot(X,tmp1,color="black",label = 'Imagined')
plt.fill_between(X, tmp1-tmp1b, tmp1+tmp1b,color="black",alpha=0.2)
# online
tmp_main = np.squeeze(np.mean(mahab_distances_online_days,1))
tmp2 = np.mean(tmp_main,axis=0)
tmp2b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp2 = np.insert(tmp2,0,tmp2[0],axis=0)
tmp2b = np.insert(tmp2b,0,tmp2b[0],axis=0)
tmp2 = np.convolve(tmp2, np.ones(N)/N, mode='same')
tmp2b = np.convolve(tmp2b, np.ones(N)/N, mode='same')
tmp2 = tmp2[1:]
tmp2b = tmp2b[1:]
plt.plot(X,tmp2,color="blue",label = 'Online')
plt.fill_between(X, tmp2-tmp2b, tmp2+tmp2b,color="blue",alpha=0.2)
# batch
tmp_main = np.squeeze(np.mean(mahab_distances_batch_days,1))
tmp3 = np.mean(tmp_main,axis=0)
tmp3b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp3 = np.insert(tmp3,0,tmp3[0],axis=0)
tmp3b = np.insert(tmp3b,0,tmp3b[0],axis=0)
tmp3 = np.convolve(tmp3, np.ones(N)/N, mode='same')
tmp3b = np.convolve(tmp3b, np.ones(N)/N, mode='same')
tmp3 = tmp3[1:]
tmp3b = tmp3b[1:]
plt.plot(X,tmp3,color="red",label = 'Batch')
plt.fill_between(X, tmp3-tmp3b, tmp3+tmp3b,color="red",alpha=0.2)
plt.legend(loc='upper left')
plt.xlabel('Days',**hfont)
plt.ylabel('Mahalanobis Distances',**hfont)
plt.show()
#tmp = np.concatenate((tmp1[:,None],tmp2[:,None],tmp3[:,None]),axis=1)
tmp = np.concatenate((tmp2[:,None],tmp3[:,None]),axis=1)
#tmp =(tmp1[:,None])
mahab_stereo = tmp.flatten()

data =np.load('ProcessedData_B1_CKD_First2pt6s_New_New.npz')
mahab_distances_online_days = data.get('mahab_distances_online_days')
mahab_distances_imagined_days = data.get('mahab_distances_imagined_days')
mahab_distances_batch_days = data.get('mahab_distances_batch_days')
num_days=5
N=2
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(num_days)+1
# imagined 
tmp_main = np.squeeze(np.mean(mahab_distances_imagined_days,axis=1))
tmp1 = np.mean(tmp_main,axis=0)
tmp1b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp1 = np.insert(tmp1,0,tmp1[0],axis=0)
tmp1b = np.insert(tmp1b,0,tmp1b[0],axis=0)
tmp1 = np.convolve(tmp1, np.ones(N)/N, mode='same')
tmp1 = tmp1[1:]
tmp1b = tmp1b[1:]
plt.plot(X,tmp1,color="black",label = 'Imagined')
plt.fill_between(X, tmp1-tmp1b, tmp1+tmp1b,color="black",alpha=0.2)
# online
tmp_main = np.squeeze(np.mean(mahab_distances_online_days,1))
tmp2 = np.mean(tmp_main,axis=0)
tmp2b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp2 = np.insert(tmp2,0,tmp2[0],axis=0)
tmp2b = np.insert(tmp2b,0,tmp2b[0],axis=0)
tmp2 = np.convolve(tmp2, np.ones(N)/N, mode='same')
tmp2b = np.convolve(tmp2b, np.ones(N)/N, mode='same')
tmp2 = tmp2[1:]
tmp2b = tmp2b[1:]
plt.plot(X,tmp2,color="blue",label = 'Online')
plt.fill_between(X, tmp2-tmp2b, tmp2+tmp2b,color="blue",alpha=0.2)
# batch
tmp_main = np.squeeze(np.mean(mahab_distances_batch_days,1))
tmp3 = np.mean(tmp_main,axis=0)
tmp3b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp3 = np.insert(tmp3,0,tmp3[0],axis=0)
tmp3b = np.insert(tmp3b,0,tmp3b[0],axis=0)
tmp3 = np.convolve(tmp3, np.ones(N)/N, mode='same')
tmp3b = np.convolve(tmp3b, np.ones(N)/N, mode='same')
tmp3 = tmp3[1:]
tmp3b = tmp3b[1:]
plt.plot(X,tmp3,color="red",label = 'Batch')
plt.fill_between(X, tmp3-tmp3b, tmp3+tmp3b,color="red",alpha=0.2)
plt.legend(loc='upper left')
plt.xlabel('Days',**hfont)
plt.ylabel('Mahalanobis Distances',**hfont)
plt.show()
#tmp = np.concatenate((tmp1[:,None],tmp2[:,None],tmp3[:,None]),axis=1)
tmp = np.concatenate((tmp2[:,None],tmp3[:,None]),axis=1)
tmp = tmp2[:,None]
#tmp =(tmp1[:,None])

mahab_ckd_3s = tmp.flatten()


data =np.load('ProcessedData_B1_CKD_45Deg_IntVel.npz')
mahab_distances_online_days = data.get('mahab_distances_online_days')
mahab_distances_imagined_days = data.get('mahab_distances_imagined_days')
mahab_distances_batch_days = data.get('mahab_distances_batch_days')
num_days=5
N=2
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(num_days)+1
# imagined 
tmp_main = np.squeeze(np.mean(mahab_distances_imagined_days,axis=1))
tmp1 = np.mean(tmp_main,axis=0)
tmp1b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp1 = np.insert(tmp1,0,tmp1[0],axis=0)
tmp1b = np.insert(tmp1b,0,tmp1b[0],axis=0)
tmp1 = np.convolve(tmp1, np.ones(N)/N, mode='same')
tmp1 = tmp1[1:]
tmp1b = tmp1b[1:]
plt.plot(X,tmp1,color="black",label = 'Imagined')
plt.fill_between(X, tmp1-tmp1b, tmp1+tmp1b,color="black",alpha=0.2)
# online
tmp_main = np.squeeze(np.mean(mahab_distances_online_days,1))
tmp2 = np.mean(tmp_main,axis=0)
tmp2b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp2 = np.insert(tmp2,0,tmp2[0],axis=0)
tmp2b = np.insert(tmp2b,0,tmp2b[0],axis=0)
tmp2 = np.convolve(tmp2, np.ones(N)/N, mode='same')
tmp2b = np.convolve(tmp2b, np.ones(N)/N, mode='same')
tmp2 = tmp2[1:]
tmp2b = tmp2b[1:]
plt.plot(X,tmp2,color="blue",label = 'Online')
plt.fill_between(X, tmp2-tmp2b, tmp2+tmp2b,color="blue",alpha=0.2)
# batch
tmp_main = np.squeeze(np.mean(mahab_distances_batch_days,1))
tmp3 = np.mean(tmp_main,axis=0)
tmp3b = np.std(tmp_main,axis=0)/sqrt(tmp_main.shape[0])
tmp3 = np.insert(tmp3,0,tmp3[0],axis=0)
tmp3b = np.insert(tmp3b,0,tmp3b[0],axis=0)
tmp3 = np.convolve(tmp3, np.ones(N)/N, mode='same')
tmp3b = np.convolve(tmp3b, np.ones(N)/N, mode='same')
tmp3 = tmp3[1:]
tmp3b = tmp3b[1:]
plt.plot(X,tmp3,color="red",label = 'Batch')
plt.fill_between(X, tmp3-tmp3b, tmp3+tmp3b,color="red",alpha=0.2)
plt.legend(loc='upper left')
plt.xlabel('Days',**hfont)
plt.ylabel('Mahalanobis Distances',**hfont)
plt.show()

#tmp = np.concatenate((tmp1[:,None],tmp2[:,None],tmp3[:,None]),axis=1)
tmp = np.concatenate((tmp2[:,None],tmp3[:,None]),axis=1)
tmp = tmp2[:,None]
#tmp =(tmp1[:,None])
mahab_ckd_45deg = tmp.flatten()

mahab_stereo = np.log(mahab_stereo)
mahab_ckd_3s = np.log(mahab_ckd_3s)
mahab_ckd_45deg = np.log(mahab_ckd_45deg)
fig=plt.figure()
plt.boxplot([mahab_stereo,mahab_ckd_3s,mahab_ckd_45deg])
plt.yticks(np.arange(-2,4.1,2))
plt.xticks([1,2,3])
plt.tick_params(labelbottom=False)
plt.tick_params(labelleft=False)
# image_format = 'svg' # e.g .png, .svg, etc.
# image_name = 'CKA_IBID_Mahab_ClosedLoop.svg'
# fig.savefig(image_name, format=image_format, dpi=300)

# t-test between mahab stereo and ckd
print(stats.ttest_ind(mahab_stereo,mahab_ckd_3s))
print(stats.ttest_ind(mahab_stereo,mahab_ckd_45deg))
print(stats.ttest_rel(mahab_ckd_3s,mahab_ckd_45deg))

#%% DRIFT PLOTTING CONTINUED
# plotting the number of sig layers in pairwise comparisons

data =np.load('ManifoldAnalyses_Main_CKD_All_IntDirTowardsTarget_All3Loop_1000boot.npz',allow_pickle=True)
pval_results = data.get('pval_results')
simal_res = data.get('simal_res')
recon_res = data.get('recon_res')

pval=np.array([])
for i in np.arange(pval_results.shape[0]):
    pval = np.append(pval,pval_results[i][1][:-1])

# plot all the pairwise comparisons... histogram of no. of significant CKA
pfdr,pfdr_thresh=fdr_threshold(pval,0.05,'Parametric')
prop_res=  np.zeros((len(pval_results),6))
for i in np.arange(len(pval_results)):
    tmp = pval_results[i][1][:-1]
    tmp = np.sum(tmp<=pfdr)
    prop_res[i,tmp] = 1
a= np.sum(prop_res,axis=0)/prop_res.shape[0]
t = np.arange(6)
plt.figure();
plt.bar(t,a,width=0.8,color=(0.5,0.5,0.5));   
plt.ylim((0,1))
plt.yticks(np.arange(0,1.01,0.2))
plt.show()
print(a[:,None])


data =np.load('ManifoldAnalyses_Main_CKD_All_First2pt6s_New_All3Loop_1000boot.npz',allow_pickle=True)
pval_results = data.get('pval_results')
simal_res = data.get('simal_res')
recon_res = data.get('recon_res')

pval=np.array([])
for i in np.arange(pval_results.shape[0]):
    pval = np.append(pval,pval_results[i][1][:-1])

# plot all the pairwise comparisons... histogram of no. of significant CKA
pfdr,pfdr_thresh=fdr_threshold(pval,0.05,'Parametric')
prop_res=  np.zeros((len(pval_results),6))
for i in np.arange(len(pval_results)):
    tmp = pval_results[i][1][:-1]
    tmp = np.sum(tmp<=pfdr)
    prop_res[i,tmp] = 1
b= np.sum(prop_res,axis=0)/prop_res.shape[0]
t = np.arange(6)
plt.figure();
plt.bar(t,b,width=0.8,color=(0.5,0.5,0.5));   
plt.ylim((0,1))
plt.yticks(np.arange(0,1.01,0.2))
plt.show()
print(b[:,None])

res = np.vstack((a[None,:][0],b[None,:][0]))
m = np.mean(res,axis=0)
t = np.arange(6)
fig=plt.figure();
plt.bar(t,m,width=0.8,color=(0.5,0.5,0.5));   
plt.scatter(t+(rnd.randn(len(t))*0.05), res[0,:],color = (0.2,0.2,0.8))
plt.scatter(t+(rnd.randn(len(t))*0.05), res[1,:],color=(0.2,0.2,0.8))
plt.xticks(t)
plt.yticks(np.arange(0,1.01,0.2))
plt.ylim((0,1))
plt.show()
plt.tick_params(labelleft=False)
plt.tick_params(labelbottom=False)
# image_format = 'svg' # e.g .png, .svg, etc.
# image_name = 'CKA_res_CKD_B1.svg'
# fig.savefig(image_name, format=image_format, dpi=300)



# PLOTTING THE SIMILARITIES LAYER BY LAYER FOR BOTH CKD AND IBID
data =np.load('ManifoldAnalyses_Main_CKD_All_First2pt6s_New_All3Loop_1000boot.npz',allow_pickle=True)
simal_res_ckd1 = data.get('simal_res')
data =np.load('ManifoldAnalyses_Main_CKD_All_IntDirTowardsTarget_All3Loop_1000boot.npz',allow_pickle=True)
simal_res_ckd2 = data.get('simal_res')
data=np.load('ManifoldAnalyses_Main_1000Boot.npz',allow_pickle=True)
simal_res = data.get('simal_res')

simal_CKD1=np.empty([0,5])
for i in np.arange(simal_res_ckd1.shape[0]):
    #simal_IBID = np.append(simal_IBID,simal_res[i][1])
    tmp = simal_res_ckd1[i][1][:-1][None,:]
    simal_CKD1 = np.concatenate((simal_CKD1,tmp),axis=0)

simal_CKD2=np.empty([0,5])
for i in np.arange(simal_res_ckd2.shape[0]):
    #simal_IBID = np.append(simal_IBID,simal_res[i][1])
    tmp = simal_res_ckd2[i][1][:-1][None,:]
    simal_CKD2 = np.concatenate((simal_CKD2,tmp),axis=0)

simal_ibid=np.empty([0,5])
for i in np.arange(simal_res.shape[0]):
    #simal_IBID = np.append(simal_IBID,simal_res[i][1])
    tmp = simal_res[i][1][:-1][None,:]
    simal_ibid = np.concatenate((simal_ibid,tmp),axis=0)

plt.figure();
plt.boxplot([simal_CKD1.flatten(),simal_CKD2.flatten(),simal_ibid.flatten()])


# plot median with error bars 
m1 = np.median(simal_CKD1.flatten())
m1b = stats.bootstrap((simal_CKD1.flatten(),), np.median,method='percentile')
m1ci = m1b.confidence_interval
m2 = np.median(simal_CKD2.flatten())
m2b = stats.bootstrap((simal_CKD2.flatten(),), np.median,method='percentile')
m2ci = m2b.confidence_interval
m3 = np.median(simal_ibid.flatten())
m3b = stats.bootstrap((simal_ibid.flatten(),), np.median,method='percentile')
m3ci = m3b.confidence_interval
yerr=np.empty((2,3))
yerr[0,:] = [m1-m1ci.low,m2-m2ci.low,m3-m3ci.low]
yerr[1,:] = [m1ci[1]-m1,m2ci[1]-m2,m3ci[1]-m3]
yerr= yerr[:,[2,0,1]]
fig=plt.figure();
plt.errorbar([1,2,3],[m3,m1,m2],yerr=yerr,elinewidth=2,capthick=2,capsize=4,color='k',
             ls='none');
plt.xticks(ticks=[1,2,3])
plt.yticks(ticks=np.arange(0,1.01,0.1))
plt.ylim((0.1,0.7))
plt.plot([1,2,3],[m3,m1,m2],'o',markersize=7,markerfacecolor='k',markeredgecolor='k')
plt.tick_params(labelleft=False)
plt.tick_params(labelbottom=False)
plt.show()
# image_format = 'svg' # e.g .png, .svg, etc.
# image_name = 'Median_CKA_res_CKD_IBID.svg'
# fig.savefig(image_name, format=image_format, dpi=300)


# plot boxplots layer by layer
for i in np.arange(simal_CKD1.shape[1]):
    plt.figure();
    plt.boxplot([simal_CKD1[:,i],simal_CKD2[:,i],simal_ibid[:,i]])
    plt.title('Layer ' + str(i+1))



a=simal_CKD1.flatten()
b=simal_CKD2.flatten()
c=simal_ibid.flatten()
pvalue = bootstrap_difference_test(a,b,'median')
pvalue = bootstrap_difference_test(a,c,'median')
pvalue = bootstrap_difference_test(c,b,'median')

#%% PLOTTING THE DECODING ACCURACY OF DISCERNING THE DAY OF RECORDING (B1, B2, B3)

import os
os.chdir('C:/Users/nikic/Documents/GitHub/NN/hDoF_Plasticity_BCI/iAE_analyses')
from iAE_utils_models import *
data=np.load('RepresentationalDrift_Mean_Across_Days_B1.npz',allow_pickle=True)
res_acc_B1 = data.get('res_acc_B1')
res_acc_B1_demean = data.get('res_acc_B1_demean')

# bootstrp the mean
res_mean_boot = mean_bootstrap(res_acc_B1,1000)[0]
res_demean_boot = mean_bootstrap(res_acc_B1_demean,1000)[0]

# using scipy
# tmp_boot = stats.bootstrap((res_acc_B1,), np.mean)
# res_mean_boot = tmp_boot.confidence_interval
# tmp_boot1 = stats.bootstrap((res_acc_B1_demean,), np.mean)
# res_demean_boot = tmp_boot1.confidence_interval

# bootstrap and plot with confidence intervals 
X = np.arange(res_acc_B1.shape[1])+1
fig=plt.figure()
plt.ylim([0,1])
tmp = np.mean(res_acc_B1,axis=0)
tmp_low =  res_mean_boot[9,:]#res_mean_boot.low
tmp_high = res_mean_boot[989,:]#res_mean_boot.high
plt.plot(X,tmp,color="black")
plt.fill_between(X, tmp_low, tmp_high,color="black",alpha=0.2)
tmp = np.mean(res_acc_B1_demean,axis=0)
tmp_low =  res_demean_boot[9,:]#res_demean_boot.low
tmp_high = res_demean_boot[989,:]#res_demean_boot.high
plt.plot(X,tmp,color="blue")
plt.fill_between(X, tmp_low, tmp_high,color="blue",alpha=0.2)
plt.hlines(1/len(X),1,len(X),color='r',linestyles='dotted')
plt.xticks(X)
plt.yticks(np.arange(0,1.10,0.1))
plt.tick_params(labelleft=False,labelbottom=False)
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Day_Decoding_Acc_B1.svg'
fig.savefig(image_name, format=image_format, dpi=300)


# t-test against null distribution
demean = np.mean(res_acc_B1_demean,axis=0)
t=stats.ttest_1samp(demean,0.1)
print(t)

# now do it for B2
data=np.load('RepresentationalDrift_Mean_Across_Days_B2.npz',allow_pickle=True)
res_acc_B2 = data.get('res_acc_B2')
res_acc_B2_demean = data.get('res_acc_demean_B2')

# bootstrp the mean
res_mean_boot = mean_bootstrap(res_acc_B2,1000)[0]
res_demean_boot = mean_bootstrap(res_acc_B2_demean,1000)[0]

# using scipy
# tmp_boot = stats.bootstrap((res_acc_B2,), np.mean)
# res_mean_boot = tmp_boot.confidence_interval
# tmp_boot1 = stats.bootstrap((res_acc_B2_demean,), np.mean)
# res_demean_boot = tmp_boot1.confidence_interval

# bootstrap and plot with confidence intervals 
X = np.arange(res_acc_B2.shape[1])+1
fig=plt.figure()
plt.ylim([0,1])
tmp = np.mean(res_acc_B2,axis=0)
tmp_low =  res_mean_boot[9,:]#res_mean_boot.low
tmp_high = res_mean_boot[989,:]#res_mean_boot.high
plt.plot(X,tmp,color="black")
plt.fill_between(X, tmp_low, tmp_high,color="black",alpha=0.2)
tmp = np.mean(res_acc_B2_demean,axis=0)
tmp_low =  res_demean_boot[9,:]#res_demean_boot.low
tmp_high = res_demean_boot[989,:]#res_demean_boot.high
plt.plot(X,tmp,color="blue")
plt.fill_between(X, tmp_low, tmp_high,color="blue",alpha=0.2)
plt.hlines(1/len(X),1,len(X),color='r',linestyles='dotted')
plt.xticks(X)
plt.yticks(np.arange(0,1.10,0.2))
plt.tick_params(labelleft=False,labelbottom=False)
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Day_Decoding_Acc_B2.svg'
fig.savefig(image_name, format=image_format, dpi=300)

# stats
demean = np.mean(res_acc_B2_demean,axis=0)
t=stats.ttest_1samp(demean,0.2)
print(t)

#### do it for B3

import os
os.chdir('C:/Users/nikic/Documents/GitHub/NN/hDoF_Plasticity_BCI/iAE_analyses')
from iAE_utils_models import *

data=np.load('RepresentationalDrift_Mean_Across_Days_B3.npz',allow_pickle=True)
res_acc_B3 = data.get('res_acc_B3')
res_acc_B3_demean = data.get('res_acc_B3_demean')

# bootstrp the mean
res_mean_boot = mean_bootstrap(res_acc_B3,1000)[0]
res_demean_boot = mean_bootstrap(res_acc_B3_demean,1000)[0]

# using scipy
# tmp_boot = stats.bootstrap((res_acc_B3,), np.mean)
# res_mean_boot = tmp_boot.confidence_interval
# tmp_boot1 = stats.bootstrap((res_acc_B3_demean,), np.mean)
# res_demean_boot = tmp_boot1.confidence_interval

# bootstrap and plot with confidence intervals 
X = np.arange(res_acc_B3.shape[1])+1
fig=plt.figure()
plt.ylim([0,1])
tmp = np.mean(res_acc_B3,axis=0)
tmp_low =  res_mean_boot[9,:]#res_mean_boot.low
tmp_high = res_mean_boot[989,:]#res_mean_boot.high
plt.plot(X,tmp,color="black")
plt.fill_between(X, tmp_low, tmp_high,color="black",alpha=0.2)
tmp = np.mean(res_acc_B3_demean,axis=0)
tmp_low =  res_demean_boot[9,:]#res_demean_boot.low
tmp_high = res_demean_boot[989,:]#res_demean_boot.high
plt.plot(X,tmp,color="blue")
plt.fill_between(X, tmp_low, tmp_high,color="blue",alpha=0.2)
plt.hlines(1/len(X),1,len(X),color='r',linestyles='dotted')
plt.xticks(X)
plt.yticks(np.arange(0,1.10,0.2))
plt.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tick_params(labelleft=False,labelbottom=False)
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Day_Decoding_Acc_B3.svg'
fig.savefig(image_name, format=image_format, dpi=300)

# stats
demean = np.mean(res_acc_B3_demean,axis=0)
t=stats.ttest_1samp(demean,(1/11))
print(t)


