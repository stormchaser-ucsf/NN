# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 08:58:03 2022

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


# loading it back
data=np.load('whole_dataSamples_stats_results_withBatch_Main_withVariance.npz')
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




# plotting latent spaces
az=-54
el=24
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
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Latent_Day10_Imag.svg'
fig_imagined.savefig(image_name, format=image_format, dpi=300)
image_name = 'Latent_Day10_Online.svg'
fig_online.savefig(image_name, format=image_format, dpi=300)
image_name = 'Latent_Day10_Batch.svg'
fig_batch.savefig(image_name, format=image_format, dpi=300)


# plotting overall variances over days (MAIN)
N=2
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(10)+1
X=np.arange(10)+1
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



# plotting variances as boxplot 
tmp1 = np.squeeze(np.mean(var_online_days,axis=1))
tmp2 = np.squeeze(np.mean(var_imagined_days,axis=1))
a=np.ndarray.flatten(tmp1)[:,None]
b=np.ndarray.flatten(tmp2)[:,None]
c=np.concatenate((a,b),axis=1)
plt.figure()
plt.boxplot(c)
c=np.concatenate((var_distances_batch_iter,var_distances_online_iter,
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


# plotting mean centroid variances over days  (MAIN)
N=2
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(10)+1
X=np.arange(10)+1
# imagined 
tmp_main = np.squeeze(np.median(var_imagined_days,1))
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
tmp_main = np.squeeze(np.median(var_online_days,1))
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
tmp_main = np.squeeze(np.median(var_batch_days,1))
tmp3 = np.median(tmp_main,axis=0)
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
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Overall_Variance_Latent_Days.svg'
fig.savefig(image_name, format=image_format, dpi=300)


# plotting mean Mahalanobis distance over days  (MAIN)
N=2
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(10)+1
X=np.arange(10)+1
# imagined 
tmp_main = np.squeeze(np.mean(mahab_distances_imagined_days,1))
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
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Mean_Mahalanobis_dist_Days_withBatch.svg'
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
plt.ylabel('Mahalanobis Distances',**hfont)
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Overall_Variance_Latent_Boxplot.svg'
fig.savefig(image_name, format=image_format, dpi=300)
print(np.mean(tmp,axis=0))


# plotting mean silhoutte index over days  (MAIN)
N=2
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(10)+1
X=np.arange(10)+1
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
sigma = 1
fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
X=np.arange(10)+1
# imagined days
tmp_main = np.squeeze(np.mean(mahab_distances_imagined_days,1))
tmp = np.mean(tmp_main,axis=0)
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
tmp = np.mean(tmp_main,axis=0)
tmp_boot, tmp_boot_std = median_bootstrap(tmp_main,1000)
tmp1 = tmp_boot_std/1#sqrt(tmp_main.shape[0])
tmp = gaussian_filter1d(tmp, sigma=sigma)
tmp1 = gaussian_filter1d(tmp1, sigma=sigma)
plt.plot(X,tmp,color="blue",label = 'Online')
plt.fill_between(X, tmp-tmp1, tmp+tmp1,color="blue",alpha=0.2)
#plt.plot(X,tmp+tmp1,color="blue",linestyle="dotted")
#plt.plot(X,tmp-tmp1,color="blue",linestyle="dotted")
# batch update days 
tmp_main = np.squeeze(np.mean(mahab_distances_batch_days,1))
tmp = np.mean(tmp_main,axis=0)
tmp_boot, tmp_boot_std = median_bootstrap(tmp_main,1000)
tmp1 = tmp_boot_std/1#sqrt(tmp_main.shape[0])
tmp = gaussian_filter1d(tmp, sigma=sigma)
tmp1 = gaussian_filter1d(tmp1, sigma=sigma)
plt.plot(X,tmp,color="red",label = 'Batch')
plt.fill_between(X, tmp-tmp1, tmp+tmp1,color="red",alpha=0.2)
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



# Median Mahalanobis distance -> with bootstrapped standard errors of the median
sigma = 0.01
N=2 # N day running average
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
tmp1 = tmp_boot_std/1#sqrt(tmp_main.shape[0])
tmp = np.convolve(tmp, np.ones(N)/N, mode='same')
tmp1 = np.convolve(tmp1, np.ones(N)/N, mode='same')
plt.plot(X,tmp,color="blue",label = 'Online')
plt.fill_between(X, tmp-tmp1, tmp+tmp1,color="blue",alpha=0.2)
# batch update days 
tmp_main = np.squeeze(np.mean(mahab_distances_batch_days,1))
tmp = np.median(tmp_main,axis=0)
tmp_boot, tmp_boot_std = median_bootstrap(tmp_main,1000)
tmp1 = tmp_boot_std/1#sqrt(tmp_main.shape[0])
tmp = np.convolve(tmp, np.ones(N)/N, mode='same')
tmp1 = np.convolve(tmp1, np.ones(N)/N, mode='same')
plt.plot(X,tmp,color="red",label = 'Batch')
plt.fill_between(X, tmp-tmp1, tmp+tmp1,color="red",alpha=0.2)
plt.xlabel('Days',**hfont)
plt.ylabel('Mahalanobis Distance',**hfont)
plt.legend()
plt.show()
# save
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Variances_Days_Latent.svg'
fig.savefig(image_name, format=image_format, dpi=300)




#### plot Mahab dist mean with confidence intervals old method
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