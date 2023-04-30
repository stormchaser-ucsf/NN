# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 16:58:24 2022

@author: nikic

Goal is to understand representational drift. 
Load two actions across two days into a common manifold and look at how similar
or dissimilar they are. Can do it separately for imagined, online and batch.
Class labels -> can be just for the actions themselves, or for the days or even 
not required. Use Rt thumb, lips and left leg for example
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
import statsmodels.api as sm
# setting up GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model params
input_size=96
hidden_size=48
latent_dims=2
num_classes = 7

# training params 
num_epochs=200
batch_size=32
learning_rate = 1e-3
batch_val=512
patience=5
gradient_clipping=10

# file location
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker'
root_imag_filename = '\condn_data_Imagined_Day'
root_online_filename = '\condn_data_Online_Day'
root_batch_filename = '\condn_data_Batch_Day'

#### PART 1 # get imagined data days 1-2 for lips, rt thumb and left leg
days=1
imagined_file_name = root_path + root_imag_filename +  str(days) + '.mat'
condn_data_imagined,Yimagined = get_data(imagined_file_name)
a = np.argmax(Yimagined,axis=1)
a = np.array([np.where(a==0)[0], np.where(a==1)[0], np.where(a==4)[0]]).flatten()
condn_data_imagined_day1 = condn_data_imagined[a,:]
Yimagined_day1 = Yimagined[a,:]

days = 4
imagined_file_name = root_path + root_imag_filename +  str(days) + '.mat'
condn_data_imagined,Yimagined = get_data(imagined_file_name)
a = np.argmax(Yimagined,axis=1)
a = np.array([np.where(a==0)[0], np.where(a==1)[0], np.where(a==4)[0]]).flatten()
condn_data_imagined_day2 = condn_data_imagined[a,:]
Yimagined_day2 = Yimagined[a,:]


# combining all across both days 
num_classes=3
condn_data_imagined_total =  np.concatenate((condn_data_imagined_day1,
                                            condn_data_imagined_day2),axis=0)
Yimagined_total =  np.concatenate((Yimagined_day1,Yimagined_day2),axis=0)
a = np.argmax(Yimagined_total,axis=1)
idx = np.unique(a)
tmpY = np.zeros((Yimagined_total.shape[0],num_classes))
for i in np.arange(len(idx)):
    x = np.where(a==idx[i])[0]
    tmpY[x,i]=1

plt.figure(); plt.imshow(tmpY,aspect='auto',interpolation='None');

Yimagined_total = tmpY
Yimagined_day1 = Yimagined_total[:Yimagined_day1.shape[0],:]
Yimagined_day2 = Yimagined_total[Yimagined_day1.shape[0]:,:]


# build a model
if 'model' in locals():
    del model   

nn_filename = 'iAE_drift_days12.pth'
model = iAutoencoder(input_size,hidden_size,latent_dims,num_classes).to(device)    
Ytest = np.zeros((2,2))
while len(np.unique(np.argmax(Ytest,axis=1)))<num_classes:
    Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_imagined_total,Yimagined_total,0.8)                            
model,acc = training_loop_iAE(model,num_epochs,batch_size,learning_rate,batch_val,
                      patience,gradient_clipping,nn_filename,
                      Xtrain,Ytrain,Xtest,Ytest,
                      input_size,hidden_size,latent_dims,num_classes)

D,z,idx,fig_imagined = plot_latent(model,Xtest,Ytest,
                                   Xtest.shape[0],latent_dims)        


D,z,idx,fig_imagined = plot_latent(model,condn_data_imagined_day1,Yimagined_day1,
                                   condn_data_imagined_day1.shape[0],latent_dims)        


#%% # PART 2 combining an action from separate days and looking at separability 

days=1
imagined_file_name = root_path + root_imag_filename +  str(days) + '.mat'
condn_data_imagined,Yimagined = get_data(imagined_file_name,7)
a1 = np.argmax(Yimagined,axis=1)
a1 = np.array([np.where(a1==0)[0]]).flatten()
condn_data_imagined_day1 = condn_data_imagined[a1,:]
Yimagined_day1 = Yimagined[a1,:]

days=4
imagined_file_name = root_path + root_imag_filename +  str(days) + '.mat'
condn_data_imagined,Yimagined = get_data(imagined_file_name,7)
a2 = np.argmax(Yimagined,axis=1)
a2 = np.array([np.where(a2==0)[0]]).flatten()
condn_data_imagined_day2 = condn_data_imagined[a2,:]
Yimagined_day2 = Yimagined[a2,:]
Yimagined_day2 = np.roll(Yimagined_day2,1)

days=8
imagined_file_name = root_path + root_imag_filename +  str(days) + '.mat'
condn_data_imagined,Yimagined = get_data(imagined_file_name,7)
a2 = np.argmax(Yimagined,axis=1)
a2 = np.array([np.where(a2==0)[0]]).flatten()
condn_data_imagined_day3 = condn_data_imagined[a2,:]
Yimagined_day3 = Yimagined[a2,:]
Yimagined_day3 = np.roll(Yimagined_day3,2)

# OPIONAL IMPORTANT
# remove the mean to see if there is true drift in somatotopy
condn_data_imagined_day1 = condn_data_imagined_day1 - np.mean(condn_data_imagined_day1,axis=0)
condn_data_imagined_day2 = condn_data_imagined_day2 - np.mean(condn_data_imagined_day2,axis=0)
condn_data_imagined_day3 = condn_data_imagined_day3 - np.mean(condn_data_imagined_day3,axis=0)


num_classes=3
condn_data_imagined_total =  np.concatenate((condn_data_imagined_day1,
                                            condn_data_imagined_day2,
                                            condn_data_imagined_day3),axis=0)
Yimagined_total =  np.concatenate((Yimagined_day1,Yimagined_day2,
                                   Yimagined_day3),axis=0)
a = np.argmax(Yimagined_total,axis=1)
idx = np.unique(a)
tmpY = np.zeros((Yimagined_total.shape[0],num_classes))
for i in np.arange(len(idx)):
    x = np.where(a==idx[i])[0]
    tmpY[x,i]=1

#plt.figure(); plt.imshow(tmpY,aspect='auto',interpolation='None');

Yimagined_total = tmpY
Yimagined_day1 = Yimagined_total[:Yimagined_day1.shape[0],:]
Yimagined_day2 = Yimagined_total[Yimagined_day1.shape[0]:,:]               


# build a model
if 'model' in locals():
    del model   

nn_filename = 'iAE_drift_days12.pth'
model = iAutoencoder(input_size,hidden_size,latent_dims,num_classes).to(device)  
  
Ytest = np.zeros((2,2))
while len(np.unique(np.argmax(Ytest,axis=1)))<num_classes:
    Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_imagined_total,Yimagined_total,0.8)                            
model,acc = training_loop_iAE(model,num_epochs,batch_size,learning_rate,batch_val,
                      patience,gradient_clipping,nn_filename,
                      Xtrain,Ytrain,Xtest,Ytest,
                      input_size,hidden_size,latent_dims,num_classes)

D,z,idx,fig_imagined = plot_latent(model,condn_data_imagined_total,Yimagined_total,
                                   condn_data_imagined_total[0],latent_dims)        

fig_imagined.axes[0].xaxis.set_ticklabels([])
fig_imagined.axes[0].yaxis.set_ticklabels([])
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'LatentRtHand_Days148.svg'
#fig_imagined.savefig(image_name, format=image_format, dpi=300)

# D,z,idx,fig_imagined = plot_latent(model,condn_data_imagined_day1,Yimagined_day1,
#                                    condn_data_imagined_day1.shape[0],latent_dims)     

delta_recon_imag,beta_recon_imag,hg_recon_imag = return_recon(model,
                                              condn_data_imagined_total,Yimagined_total,num_classes)


# getting correlation
corr_coef=[]
tmp1 = np.mean(hg_recon_imag[0],axis=0)
tmp2 = np.mean(hg_recon_imag[1],axis=0)
tmp3 = np.mean(hg_recon_imag[2],axis=0)
a = np.corrcoef(tmp1,tmp2)[0,1]
b = np.corrcoef(tmp1,tmp3)[0,1]
c = np.corrcoef(tmp2,tmp3)[0,1]
corr_coef.append([a,b,c])
print(corr_coef)

corr_coef_flat = [item for sublist in corr_coef for item in sublist]

   
# plotting
hand_channels = np.array([23 ,31]) 

tmp = hg_recon_imag[0]
tmph = tmp[:,hand_channels] # get the hand knob channels 30,31,22,23,15
hand_knob_act1 = np.mean(tmph,axis=1)
tmp = np.mean(hg_recon_imag[0],axis=0)
tmp1 = np.reshape(tmp,(4,8))
xmax1,xmin1 = tmp.max(),tmp.min()
plt.subplot(311)    
fig1=plt.imshow(tmp1)   

tmp = hg_recon_imag[1]
tmph = tmp[:,hand_channels] # get the hand knob channels 30,31,22,23,15
hand_knob_act2 = np.mean(tmph,axis=1)
tmp = np.mean(hg_recon_imag[1],axis=0)
tmp2 = np.reshape(tmp,(4,8))
xmax2,xmin2 = tmp.max(),tmp.min()
plt.subplot(312)    
fig2=plt.imshow(tmp2)   

tmp = hg_recon_imag[2]
tmph = tmp[:,hand_channels] # get the hand knob channels 30,31,22,23,15
hand_knob_act3 = np.mean(tmph,axis=1)
tmp = np.mean(hg_recon_imag[2],axis=0)
tmp3 = np.reshape(tmp,(4,8))
xmax3,xmin3 = tmp.max(),tmp.min()
plt.subplot(313)    
fig3=plt.imshow(tmp3)   

xmax = np.array([xmax1,xmax2,xmax3])
xmin = np.array([xmin1,xmin2,xmin3])
fig1.set_clim(xmin.min(),xmax.max())
fig2.set_clim(xmin.min(),xmax.max())
fig3.set_clim(xmin.min(),xmax.max())

hand_spatial = np.array([tmp1,tmp2,tmp3])
hand_spatial = np.squeeze(np.mean(hand_spatial,axis=0))
fig=plt.figure();
plt.imshow(hand_spatial)
fig.axes[0].xaxis.set_ticklabels([])
fig.axes[0].yaxis.set_ticklabels([])
fig.axes[0].yaxis.set_ticks([])
fig.axes[0].xaxis.set_ticks([])
image_format = 'svg'
image_name = 'AverageRtThumbDays148.svg'
#fig.savefig(image_name, format=image_format, dpi=300)

hand_knob = [hand_knob_act1,hand_knob_act2,hand_knob_act3]
plt.figure()
plt.boxplot(hand_knob)



fig = plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
x1=np.ones((hand_knob_act1.shape[0],1)) + 0.05*rnd.randn(hand_knob_act1.shape[0])[:,None]
x2=2*np.ones((hand_knob_act2.shape[0],1)) + 0.05*rnd.randn(hand_knob_act2.shape[0])[:,None]
x3=3*np.ones((hand_knob_act3.shape[0],1)) + 0.05*rnd.randn(hand_knob_act3.shape[0])[:,None]
plt.scatter(x1, hand_knob_act1)
plt.scatter(x2, hand_knob_act2)
plt.scatter(x3, hand_knob_act3)
plt.hlines(np.mean(hand_knob_act1),0.8,1.2,colors='black',linewidth=3)
plt.hlines(np.mean(hand_knob_act2),1.8,2.2,colors='black',linewidth=3)
plt.hlines(np.mean(hand_knob_act3),2.8,3.2,colors='black',linewidth=3)
plt.xticks([1,2,3])
plt.yticks([0.23,0.25,0.27])
plt.show()
image_format = 'svg'
image_name = 'Drift_Hand_Knob_Days.svg'
#fig.savefig(image_name, format=image_format, dpi=300)

#%% (MAIN) BUILDING A COMMON MANIFOLD ACROSS DAYS AND LOOKING AT SEPARATION IN MEAN
# have to do it condition by condition to evaluate stats 

import os
os.chdir('C:/Users/nikic/Documents/GitHub/NN/hDoF_Plasticity_BCI/iAE_analyses')
from iAE_utils_models import *

# model params
num_days=10
input_size=96
hidden_size=48
latent_dims=3
num_classes = num_days

# training params 
num_epochs=200
batch_size=64
learning_rate = 1e-3
batch_val=512
patience=5
gradient_clipping=10

# file location
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker'
root_imag_filename = '\condn_data_Imagined_Day'
root_online_filename = '\condn_data_Online_Day'
root_batch_filename = '\condn_data_Batch_Day'

#res_acc = np.empty((0,10))
res_acc_demean = np.empty((0,10))
num_iterations = 40

for iterations in np.arange(num_iterations):
    
    condn_data_total=np.empty((0,96))
    Ytotal_tmp = np.empty((0,1))
    for days in np.arange(num_days)+1:
        print('Processing day number ' + str(days))
        imagined_file_name = root_path + root_imag_filename +  str(days) + '.mat'
        condn_data_imagined,Yimagined = get_data(imagined_file_name,7)  
        condn_data_imagined,Yimagined = data_aug_mlp_chol_feature_equalSize(condn_data_imagined,
                                                        Yimagined,3000)
        # # idx = np.argmax(Yimagined,axis=1)
        # idx = np.where(idx==1)[0]
        # condn_data_imagined = condn_data_imagined[idx,:]
        # null - remove the mean
        condn_data_imagined = condn_data_imagined - np.mean(condn_data_imagined,axis=0)
        # Yimagined = Yimagined[idx,:]
        tmp_y = days*np.ones(condn_data_imagined.shape[0])
        Ytotal_tmp = np.append(Ytotal_tmp,tmp_y)
        condn_data_total = np.concatenate((condn_data_total, condn_data_imagined),axis=0)
        
    Ytotal = np.zeros((Ytotal_tmp.shape[0],num_classes))
    for i in range(Ytotal_tmp.shape[0]):
        tmp = round(Ytotal_tmp[i])
        Ytotal[i,tmp-1]=1
    
    plt.figure();
    plt.imshow(Ytotal,aspect='auto',interpolation='none')
    plt.xticks(ticks=np.arange(10))
    plt.close()
    
    # split into training and testing datasets
    Ytest = np.zeros((2,2))
    while len(np.unique(np.argmax(Ytest,axis=1)))<num_classes:
        Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_total,Ytotal,0.8)      
    condn_data_total_train = Xtrain
    Ytotal_train = Ytrain
    
    # build an AE
    if 'model' in locals():
        del model   
    nn_filename = 'iAE_AcrossDays' + str(days) + '.pth'
    Yval = np.zeros((2,2))
    # split into training and validation sets
    while len(np.unique(np.argmax(Yval,axis=1)))<num_classes:
        Xtrain,Xval,Ytrain,Yval = training_test_split(condn_data_total_train,Ytotal_train,0.85)      
    model = iAutoencoder(input_size,hidden_size,latent_dims,num_classes).to(device)                              
    model,acc = training_loop_iAE(model,num_epochs,batch_size,learning_rate,batch_val,
                          patience,gradient_clipping,nn_filename,
                          Xtrain,Ytrain,Xval,Yval,
                          input_size,hidden_size,latent_dims,num_classes)
    # on training data 
    D,z,idx,fig_imagined,acc_train,ypred = plot_latent_acc(model,condn_data_total_train,Ytotal_train,
                                       latent_dims) 
    plt.close()       
    
    # on held out data 
    D,z,idx,fig_imagined,acc_test,ypred = plot_latent_acc(model,Xtest,Ytest,
                                       latent_dims)        
    plt.close()       
    print(acc_test*100)
    
    # accuracy or confusion matrix 
    conf_matrix = np.zeros((len(np.unique(idx)),len(np.unique(idx))))
    for i in np.arange(len(idx)):
        conf_matrix[idx[i],ypred[i]] = conf_matrix[idx[i],ypred[i]]+1
    
    for i in np.arange(conf_matrix.shape[0]):
        conf_matrix[i,:] = conf_matrix[i,:] / np.sum(conf_matrix[i,:])
    
    
    plt.imshow(conf_matrix,aspect='auto',interpolation='none')    
    plt.close()
    
    #res_acc = np.concatenate((res_acc,np.diag(conf_matrix)[None,:]),axis=0)
    res_acc_demean = np.concatenate((res_acc_demean,np.diag(conf_matrix)[None,:]),axis=0)



res_acc_B1_demean = res_acc_demean
res_acc_B1 = res_acc

fig=plt.figure();
plt.plot(np.mean(res_acc_B1_demean,axis=0))
plt.plot(np.mean(res_acc_B1,axis=0))
plt.ylim((0,1))
plt.hlines(1/num_days,0,num_days-1,color='r')

np.savez('RepresentationalDrift_Mean_Across_Days_B1', 
         res_acc_B1_demean = res_acc_B1_demean,
         res_acc_B1=res_acc_B1)



#%% (MAIN) BUILDING A COMMON MANIFOLD (B2) ACROSS DAYS AND LOOKING AT SEPARATION IN MEAN
# have to do it condition by condition to evaluate stats 

import os
os.chdir('C:/Users/nikic/Documents/GitHub/NN/hDoF_Plasticity_BCI/iAE_analyses')
from iAE_utils_models import *

# model params
num_days=5
input_size=96
hidden_size=48
latent_dims=3
num_classes = num_days

# training params 
num_epochs=200
batch_size=32
learning_rate = 1e-3
batch_val=128
patience=5
gradient_clipping=10

# file location
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B2'
root_imag_filename = '\B2_condn_data_Imagined_Day'
root_online_filename = '\B2_condn_data_Online_Day'
root_batch_filename = '\B2_condn_data_Batch_Day'

res_acc = np.empty((0,num_days))
res_acc_demean = np.empty((0,num_days))
num_iterations=50

for iterations in np.arange(num_iterations):
    
    condn_data_total=np.empty((0,96))
    Ytotal_tmp = np.empty((0,1))
    for days in np.arange(num_days)+1:
        print('Processing day number ' + str(days))
        imagined_file_name = root_path + root_imag_filename +  str(days) + '.mat'
        condn_data_imagined,Yimagined = get_data_B2(imagined_file_name)    
        condn_data_imagined,Yimagined = data_aug_mlp_chol_feature_equalSize(condn_data_imagined,
                                                        Yimagined,5500)
        # idx = np.argmax(Yimagined,axis=1)
        # idx = np.where(idx==1)[0]
        # condn_data_imagined = condn_data_imagined[idx,:]
        # null - remove the mean
        condn_data_imagined = condn_data_imagined - np.mean(condn_data_imagined,axis=0)
        # Yimagined = Yimagined[idx,:]
        tmp_y = days*np.ones(condn_data_imagined.shape[0])
        Ytotal_tmp = np.append(Ytotal_tmp,tmp_y)
        condn_data_total = np.concatenate((condn_data_total, condn_data_imagined),axis=0)
        
    Ytotal = np.zeros((Ytotal_tmp.shape[0],num_classes))
    for i in range(Ytotal_tmp.shape[0]):
        tmp = round(Ytotal_tmp[i])
        Ytotal[i,tmp-1]=1
    
    plt.figure();
    plt.imshow(Ytotal,aspect='auto',interpolation='none')
    plt.xticks(ticks=np.arange(num_days))
    plt.close()
    
    # split into training and testing datasets
    Ytest = np.zeros((2,2))
    while len(np.unique(np.argmax(Ytest,axis=1)))<num_classes:
        Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_total,Ytotal,0.8)      
    condn_data_total_train = Xtrain
    Ytotal_train = Ytrain
    
    # build an AE
    if 'model' in locals():
        del model   
    nn_filename = 'iAE_AcrossDays_B2' + str(days) + '.pth'
    Yval = np.zeros((2,2))
    # split into training and validation sets
    while len(np.unique(np.argmax(Yval,axis=1)))<num_classes:
        Xtrain,Xval,Ytrain,Yval = training_test_split(condn_data_total_train,Ytotal_train,0.85)      
    model = iAutoencoder(input_size,hidden_size,latent_dims,num_classes).to(device)                              
    model,acc = training_loop_iAE(model,num_epochs,batch_size,learning_rate,batch_val,
                          patience,gradient_clipping,nn_filename,
                          Xtrain,Ytrain,Xval,Yval,
                          input_size,hidden_size,latent_dims,num_classes)
    # on training data 
    D,z,idx,fig_imagined,acc_train,ypred = plot_latent_acc(model,condn_data_total_train,Ytotal_train,
                                       latent_dims) 
    plt.close()       
    
    # on held out data 
    D,z,idx,fig_imagined,acc_test,ypred = plot_latent_acc(model,Xtest,Ytest,
                                       latent_dims)        
    plt.close()       
    print(acc_test*100)
    
    # accuracy or confusion matrix 
    conf_matrix = np.zeros((len(np.unique(idx)),len(np.unique(idx))))
    for i in np.arange(len(idx)):
        conf_matrix[idx[i],ypred[i]] = conf_matrix[idx[i],ypred[i]]+1
    
    for i in np.arange(conf_matrix.shape[0]):
        conf_matrix[i,:] = conf_matrix[i,:] / np.sum(conf_matrix[i,:])
    
    
    plt.imshow(conf_matrix,aspect='auto',interpolation='none')    
    plt.close()
    
    #res_acc = np.concatenate((res_acc,np.diag(conf_matrix)[None,:]),axis=0)
    res_acc_demean = np.concatenate((res_acc_demean,np.diag(conf_matrix)[None,:]),axis=0)


res_acc_B2 = res_acc
res_acc_demean_B2 = res_acc_demean


fig=plt.figure();
plt.plot(np.mean(res_acc_demean_B2,axis=0))
plt.plot(np.mean(res_acc_B2,axis=0))
plt.ylim((0,1))
plt.hlines(1/num_days,0,num_days-1,color='r')

res_acc_B2 = res_acc

np.savez('RepresentationalDrift_Mean_Across_Days_B2', 
         res_acc_B2 = res_acc_B2,
         res_acc_demean_B2=res_acc_demean_B2)

data=np.load('RepresentationalDrift_Mean_Across_Days_B2.npz',allow_pickle=True)
res_acc_B2 = data.get('res_acc_B2')
res_acc_demean_B2 = data.get('res_acc_demean_B2')


#%% PART C -> TRAIN ON A FEW DAYS AND PROJECT HELD OUT DAYS

# train on first 5 days for example

condn_data_total=np.empty((0,96))
Ytotal = np.empty((0,7))
for days in np.arange(5)+1:    
    
    imagined_file_name = root_path + root_imag_filename +  str(days) + '.mat'
    condn_data_imagined,Yimagined = get_data(imagined_file_name)    
    online_file_name = root_path + root_online_filename +  str(days) + '.mat'
    condn_data_online,Yonline = get_data(online_file_name)
    batch_file_name = root_path + root_batch_filename +  str(days) + '.mat'
    condn_data_batch,Ybatch = get_data(batch_file_name)  
    
    condn_data_total = np.concatenate((condn_data_total,condn_data_imagined,
                                       condn_data_online,condn_data_batch),axis=0)
    Ytotal = np.concatenate((Ytotal,Yimagined,
                                       Yonline,Ybatch),axis=0)

# run it through a model
if 'model' in locals():
    del model   
batch_size=64
patience = 5
latent_dims = 2
nn_filename = 'iAE_AcrossDays' + str(days) + '.pth'
Ytest = np.zeros((2,2))
while len(np.unique(np.argmax(Ytest,axis=1)))<num_classes:
    Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_total,Ytotal,0.8)      
model = iAutoencoder(input_size,hidden_size,latent_dims,num_classes).to(device)                              
model,acc = training_loop_iAE(model,num_epochs,batch_size,learning_rate,batch_val,
                      patience,gradient_clipping,nn_filename,
                      Xtrain,Ytrain,Xtest,Ytest,
                      input_size,hidden_size,latent_dims,num_classes)

D,z,idx,fig_imagined,acc_train = plot_latent_acc(model,condn_data_total,Ytotal,
                                   latent_dims)        

# get the mahab dist when passing a held out day through it 
condn_data_test=np.empty((0,96))
Ytest = np.empty((0,7))
days = 10
online_file_name = root_path + root_online_filename +  str(days) + '.mat'
condn_data_online,Yonline = get_data(online_file_name)
batch_file_name = root_path + root_batch_filename +  str(days) + '.mat'
condn_data_batch,Ybatch = get_data(batch_file_name)  

condn_data_test = np.concatenate((condn_data_test,condn_data_online,condn_data_batch),axis=0)
Ytest = np.concatenate((Ytest,Ytest,Yonline,Ybatch),axis=0)

D,z,idx,fig_test,acc_test = plot_latent_acc(model,condn_data_test,Ytest,
                                   latent_dims)        
print(acc_test)
mahab_distances = get_mahab_distance_latent(z,idx)
mahab_distances = mahab_distances[np.triu_indices(mahab_distances.shape[0])]
mahab_distances = mahab_distances[mahab_distances>0]
print(np.mean(mahab_distances))

#%% PART D -> BUILD A MANIFOLD JUST FOR THE HAND DATA AND EVAL HELD OUT DAY

import os
os.chdir('C:/Users/nikic/Documents/GitHub/NN/hDoF_Plasticity_BCI/iAE_analyses')
from iAE_utils_models import *
# file location
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker'
root_imag_filename = '\condn_data_Imagined_Day'
root_online_filename = '\condn_data_Online_Day'
root_batch_filename = '\condn_data_Batch_Day'

# model params
input_size=96
hidden_size=48
latent_dims=3
num_classes = 3

# training params 
num_epochs=200
batch_size=32
learning_rate = 1e-3
batch_val=512
patience=5
gradient_clipping=10

# build a manifold using only a set number of days
total_days = 10

plt_close=False
for days in np.arange(total_days-1)+1:
    print('Processing days thru ' + str(days))
    
    train_days = np.arange(days)+1
    test_days = np.arange(days+1,total_days+1)
        
    # get training data        
    condn_data_total=np.empty((0,96))
    Ytotal = np.empty((0,7))
    for i in np.arange(len(train_days)):
        imagined_file_name = root_path + root_imag_filename +  str(train_days[i]) + '.mat'
        condn_data_imagined,Yimagined = get_data(imagined_file_name)    
        online_file_name = root_path + root_online_filename +  str(train_days[i]) + '.mat'
        condn_data_online,Yonline = get_data(online_file_name)
        batch_file_name = root_path + root_batch_filename +  str(train_days[i]) + '.mat'
        condn_data_batch,Ybatch = get_data(batch_file_name)  
        
        condn_data_online,Yonline =   data_aug_mlp_chol_feature_equalSize(condn_data_online,Yonline,condn_data_imagined.shape[0])
        condn_data_batch,Ybatch =   data_aug_mlp_chol_feature_equalSize(condn_data_batch,Ybatch,condn_data_imagined.shape[0])
        
        # only on the imagined data 
        condn_data_total = np.concatenate((condn_data_total,condn_data_imagined),axis=0)
        Ytotal = np.concatenate((Ytotal,Yimagined),axis=0)
        
    # get only the hand actions
    idx = np.argmax(Ytotal,axis=1)
    a = np.concatenate((np.where(idx==0)[0],np.where(idx==2)[0],np.where(idx==6)[0]))
    condn_data_total = condn_data_total[a,:]
    Ytotal=Ytotal[a,:]
    idx1 = idx[a]
    idx1[np.where(idx1==2)[0]]=1
    idx1[np.where(idx1==6)[0]]=2
    Ytotal = one_hot_convert(idx1)
    
    if 'model' in locals():
        del model       
    nn_filename = 'iAE_AcrossDays' + str(days) + '.pth'
    Ytest = np.zeros((2,2))
    while len(np.unique(np.argmax(Ytest,axis=1)))<num_classes:
        Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_total,Ytotal,0.8)      
    model = iAutoencoder(input_size,hidden_size,latent_dims,num_classes).to(device)                              
    model,acc = training_loop_iAE(model,num_epochs,batch_size,learning_rate,batch_val,
                          patience,gradient_clipping,nn_filename,
                          Xtrain,Ytrain,Xtest,Ytest,
                          input_size,hidden_size,latent_dims,num_classes)
    
    D,z,idx,fig_imagined,acc_train,dummy = plot_latent_acc(model,condn_data_total,Ytotal,
                                       latent_dims) 
    if plt_close == True:
        plt.close()
        
    # project held out day onto this model
    mahab_tmp=[]
    latent_acc_tmp=[]
    sil_tmp=[]
    for i in np.arange(len(test_days)):
            
        condn_data_heldout=np.empty((0,96))
        Yheldout = np.empty((0,7))        
        imagined_file_name = root_path + root_imag_filename +  str(test_days[i]) + '.mat'
        condn_data_imagined,Yimagined = get_data(imagined_file_name)    
        online_file_name = root_path + root_online_filename +  str(test_days[i]) + '.mat'
        condn_data_online,Yonline = get_data(online_file_name)
        batch_file_name = root_path + root_batch_filename +  str(test_days[i]) + '.mat'
        condn_data_batch,Ybatch = get_data(batch_file_name)  
        
        # project only imagined data 
        condn_data_heldout = np.concatenate((condn_data_heldout,condn_data_imagined),axis=0)
        Yheldout = np.concatenate((Yheldout,Yimagined),axis=0)
        
        # get only the hand actions
        idx = np.argmax(Yheldout,axis=1)
        a = np.concatenate((np.where(idx==0)[0],np.where(idx==2)[0],np.where(idx==6)[0]))
        condn_data_heldout = condn_data_heldout[a,:]
        Yheldout=Yheldout[a,:]
        idx1 = idx[a]
        idx1[np.where(idx1==2)[0]]=1
        idx1[np.where(idx1==6)[0]]=2
        Yheldout = one_hot_convert(idx1)
        
        # project only closed loop data 
        # condn_data_heldout = np.concatenate((condn_data_heldout,condn_data_online,condn_data_batch),axis=0)
        # Yheldout = np.concatenate((Yheldout, Yonline,Ybatch),axis=0)
        
        D,z,idx,fig_test,acc_test,dummy = plot_latent_acc(model,condn_data_heldout,Yheldout,latent_dims) 
        if plt_close == True:
            plt.close()     
        latent_acc_tmp.append(acc_test*100)
        sil_tmp.append(D)
        
        # # for figure plotting
        # ch =[0,2,6]
        # fig_ex = plot_latent_select(model,condn_data_heldout,Yheldout,latent_dims,ch)        
        # fig_ex.axes[0].xaxis.set_ticklabels([])
        # fig_ex.axes[0].yaxis.set_ticklabels([])
        # fig_ex.axes[0].zaxis.set_ticklabels([])
        # fig_ex.axes[0].view_init(elev=24, azim=-130)
        # plt.show()
        # image_format = 'svg' # e.g .png, .svg, etc.
        # image_name = 'Hand_Day10_Days1thru9_AE.svg'
        # fig_ex.savefig(image_name, format=image_format, dpi=300)

        
        mahab_distances = get_mahab_distance_latent(z,idx)
        mahab_distances = mahab_distances[np.triu_indices(mahab_distances.shape[0])]
        mahab_distances = mahab_distances[mahab_distances>0]
        print(np.mean(mahab_distances))

#%% CONTINUATION OF PART C BUT NOW ON ALL THE DATA (MAIN)

import os
os.chdir('C:/Users/nikic/Documents/GitHub/NN/hDoF_Plasticity_BCI/iAE_analyses')
from iAE_utils_models import *
# file location
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker'
root_imag_filename = '\condn_data_Imagined_Day'
root_online_filename = '\condn_data_Online_Day'
root_batch_filename = '\condn_data_Batch_Day'

input_size=96
hidden_size=48
num_classes = 7


mahab_distances_days = []
Sil_days = []
latent_acc_days = []
total_days = 10
batch_size=64
patience = 5
latent_dims = 3
plt_close=False
for days in np.arange(total_days-1)+1:
    print('Processing days thru ' + str(days))
    
    train_days = np.arange(days)+1
    test_days = np.arange(days+1,total_days+1)
        
    # get training data        
    condn_data_total=np.empty((0,96))
    Ytotal = np.empty((0,7))
    for i in np.arange(len(train_days)):
        imagined_file_name = root_path + root_imag_filename +  str(train_days[i]) + '.mat'
        condn_data_imagined,Yimagined = get_data(imagined_file_name)    
        online_file_name = root_path + root_online_filename +  str(train_days[i]) + '.mat'
        condn_data_online,Yonline = get_data(online_file_name)
        batch_file_name = root_path + root_batch_filename +  str(train_days[i]) + '.mat'
        condn_data_batch,Ybatch = get_data(batch_file_name)  
        
        condn_data_online,Yonline =   data_aug_mlp_chol_feature_equalSize(condn_data_online,Yonline,condn_data_imagined.shape[0])
        condn_data_batch,Ybatch =   data_aug_mlp_chol_feature_equalSize(condn_data_batch,Ybatch,condn_data_imagined.shape[0])
        
        # only on the imagined data 
        condn_data_total = np.concatenate((condn_data_total,condn_data_imagined),axis=0)
        Ytotal = np.concatenate((Ytotal,Yimagined),axis=0)
        
        # get only the hand actions
        # idx = np.argmax(Ytotal,axis=1)
        # idx = np.concatenate((np.where(idx==0)[0],np.where(idx==2)[0],np.where(idx==6)[0]))
        # condn_data_total = condn_data_total[idx,:]
        # Ytotal = Ytotal[idx,:]
        # num_classes=3
        
        # on all data 
        # condn_data_total = np.concatenate((condn_data_total,condn_data_imagined,
        #                                     condn_data_online,condn_data_batch),axis=0)
        # Ytotal = np.concatenate((Ytotal,Yimagined,
        #                                     Yonline,Ybatch),axis=0)
    
    # # get testing data 
    # condn_data_heldout=np.empty((0,96))
    # Yheldout = np.empty((0,7))
    # for i in np.arange(len(test_days)):
    #     # imagined_file_name = root_path + root_imag_filename +  str(test_days[i]) + '.mat'
    #     # condn_data_imagined,Yimagined = get_data(imagined_file_name)    
    #     online_file_name = root_path + root_online_filename +  str(test_days[i]) + '.mat'
    #     condn_data_online,Yonline = get_data(online_file_name)
    #     batch_file_name = root_path + root_batch_filename +  str(test_days[i]) + '.mat'
    #     condn_data_batch,Ybatch = get_data(batch_file_name)  
        
    #     condn_data_heldout = np.concatenate((condn_data_heldout,condn_data_online,condn_data_batch),axis=0)
    #     Yheldout = np.concatenate((Yheldout, Yonline,Ybatch),axis=0)
        
    
    # build iAE    
    if 'model' in locals():
        del model       
    nn_filename = 'iAE_AcrossDays' + str(days) + '.pth'
    Ytest = np.zeros((2,2))
    while len(np.unique(np.argmax(Ytest,axis=1)))<num_classes:
        Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_total,Ytotal,0.8)      
    model = iAutoencoder(input_size,hidden_size,latent_dims,num_classes).to(device)                              
    model,acc = training_loop_iAE(model,num_epochs,batch_size,learning_rate,batch_val,
                          patience,gradient_clipping,nn_filename,
                          Xtrain,Ytrain,Xtest,Ytest,
                          input_size,hidden_size,latent_dims,num_classes)

    D,z,idx,fig_imagined,acc_train = plot_latent_acc(model,condn_data_total,Ytotal,
                                       latent_dims) 
    if plt_close == True:
        plt.close()
    
    # # test it on held out data all at once 
    # D,z,idx,fig_test,acc_test = plot_latent_acc(model,condn_data_heldout,Yheldout,
    #                                    latent_dims) 
    # if plt_close == True:
    #     plt.close()     
    # latent_acc_days.append(acc_test*100)
    # Sil_days.append(D)
    
    # mahab_distances = get_mahab_distance_latent(z,idx)
    # mahab_distances = mahab_distances[np.triu_indices(mahab_distances.shape[0])]
    # mahab_distances = mahab_distances[mahab_distances>0]
    # mahab_distances_days.append(mahab_distances)
    
    # test it out on held out data one day at a time 
    mahab_tmp=[]
    latent_acc_tmp=[]
    sil_tmp=[]
    for i in np.arange(len(test_days)):
            
        condn_data_heldout=np.empty((0,96))
        Yheldout = np.empty((0,7))        
        imagined_file_name = root_path + root_imag_filename +  str(test_days[i]) + '.mat'
        condn_data_imagined,Yimagined = get_data(imagined_file_name)    
        online_file_name = root_path + root_online_filename +  str(test_days[i]) + '.mat'
        condn_data_online,Yonline = get_data(online_file_name)
        batch_file_name = root_path + root_batch_filename +  str(test_days[i]) + '.mat'
        condn_data_batch,Ybatch = get_data(batch_file_name)  
        
        # project only imagined data 
        condn_data_heldout = np.concatenate((condn_data_heldout,condn_data_imagined),axis=0)
        Yheldout = np.concatenate((Yheldout,Yimagined),axis=0)
        
        # project only closed loop data 
        # condn_data_heldout = np.concatenate((condn_data_heldout,condn_data_online,condn_data_batch),axis=0)
        # Yheldout = np.concatenate((Yheldout, Yonline,Ybatch),axis=0)
        
        D,z,idx,fig_test,acc_test = plot_latent_acc(model,condn_data_heldout,Yheldout,latent_dims) 
        if plt_close == True:
            plt.close()     
        latent_acc_tmp.append(acc_test*100)
        sil_tmp.append(D)
        
        # # for figure plotting
        # ch =[0,2,6]
        # fig_ex = plot_latent_select(model,condn_data_heldout,Yheldout,latent_dims,ch)        
        # fig_ex.axes[0].xaxis.set_ticklabels([])
        # fig_ex.axes[0].yaxis.set_ticklabels([])
        # fig_ex.axes[0].zaxis.set_ticklabels([])
        # fig_ex.axes[0].view_init(elev=24, azim=-130)
        # plt.show()
        # image_format = 'svg' # e.g .png, .svg, etc.
        # image_name = 'Hand_Day10_Days1thru9_AE.svg'
        # fig_ex.savefig(image_name, format=image_format, dpi=300)

        
        mahab_distances = get_mahab_distance_latent(z,idx)
        mahab_distances = mahab_distances[np.triu_indices(mahab_distances.shape[0])]
        mahab_distances = mahab_distances[mahab_distances>0]
        mahab_tmp.append(np.median(mahab_distances))
    
    #mahab_distances_days.append(np.median(mahab_distances))
    mahab_distances_days.append((mahab_tmp))
    Sil_days.append(np.array(sil_tmp))
    latent_acc_days.append(np.array(latent_acc_tmp))

        

# plt.figure();
# plt.plot(latent_acc_days);
# plt.ylim([35,60])
        
# plt.figure();
# mahab_distances_days1 = np.array(mahab_distances_days)
# a=np.median(mahab_distances_days1,axis=1)
# plt.plot(np.median(mahab_distances_days1,axis=1));
# plt.plot(mahab_distances_days1);


acc_plot=[]
for i in np.arange(len(latent_acc_days)):
    tmp = latent_acc_days[i]
    acc_plot.append(mean(tmp))

plt.plot(acc_plot)   


acc_plot=[]
for i in np.arange(len(latent_acc_days)):
    tmp = latent_acc_days[i]
    acc_plot.append(mean(tmp))

plt.plot(acc_plot)

# mahab plot on all 
mahab_plot=[]
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
days=np.arange(1,10)
for i in np.arange(len(mahab_distances_days)):
    tmp = mahab_distances_days[i]
    mahab_plot.append(median(tmp))
    I = days[i]*np.ones(len(tmp)) + 0.00*rnd.randn(len(tmp))
    plt.scatter(I,tmp,c='k',alpha=0.5,edgecolors='none')
    
# do curve fitting and plot regression line
x=days
y=np.array(mahab_plot)
p = np.polyfit(x,y,1)
#p = [0.3686,6.3109] #from Robust regression below
xx = np.concatenate((np.ones((len(x),1)),x[:,None]),axis=1)
yhat = xx @ np.flip(p)[:,None]
plt.plot(days,yhat,c='k')
plt.xticks(ticks=np.arange(9)+1)
fig.axes[0].xaxis.set_ticklabels([])
fig.axes[0].yaxis.set_ticklabels([])
plt.ylim([0,8])
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'Day1thru10_Days1thru9_AE_New.svg'
fig.savefig(image_name, format=image_format, dpi=300)

# get the pval for the regression
lm = sm.OLS(y,xx).fit()
print(lm.summary())

rlm = sm.RLM(y,xx).fit()
print(rlm.summary())


# mahab plot on day 9 
mahab_plot=[]
fig=plt.figure()
days=np.arange(1,10)
for i in np.arange(len(mahab_distances_days)):
    tmp = mahab_distances_days[i]
    mahab_plot.append(tmp[-1])    

plt.figure()
plt.plot(mahab_plot)   
    
# get the data ready for linear mixed effect model
data = np.empty([0,2])
for i in np.arange(len(mahab_distances_days)):
    tmp = np.array(mahab_distances_days[i])[:,None]
    tmp_days = np.tile(i+1,tmp.shape[0])[:,None]
    a= np.concatenate((tmp,tmp_days),axis=1)
    #a=np.fliplr(np.squeeze(np.array([tmp, tmp_days]).T))
    data = np.append(data,a,axis=0)




#%% CONTINUATION OF PART C BUT NOW ON ALL THE DATA  for B2 (MAIN)


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
import statsmodels.api as sm
# setting up GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model params
input_size=96
hidden_size=48
latent_dims=2
num_classes = 4

# training params 
num_epochs=200
batch_size=32
learning_rate = 1e-3
batch_val=512
patience=5
gradient_clipping=10

# file location
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate B2'
root_imag_filename = '\B2_condn_data_Imagined_Day'
root_online_filename = '\B2_condn_data_Online_Day'
root_batch_filename = '\B2_condn_data_Batch_Day'



from iAE_utils_models import *
total_days = [1,2,3,4,5,6]
mahab_distances_days = []
Sil_days = []
latent_acc_days = []
#total_days = 6
batch_size=64
patience = 5
latent_dims = 3
plt_close=True
#for days in np.arange(total_days-1)+1:
for days in np.arange(len(total_days)-1):
    print('Processing days thru ' + str(days))
    
    # train_days = np.arange(days)+1
    # test_days = np.arange(days+1,total_days+1)
    train_days = total_days[0:days+1]
    test_days = total_days[days+1:len(total_days)]
        
    # get training data        
    condn_data_total=np.empty((0,96))
    Ytotal = np.empty((0,4))
    for i in np.arange(len(train_days)):
        imagined_file_name = root_path + root_imag_filename +  str(train_days[i]) + '.mat'
        condn_data_imagined,Yimagined = get_data_B2(imagined_file_name)
        online_file_name = root_path + root_online_filename +  str(train_days[i]) + '.mat'
        condn_data_online,Yonline = get_data_B2(online_file_name)
        batch_file_name = root_path + root_batch_filename +  str(train_days[i]) + '.mat'
        if os.path.exists(batch_file_name):
            condn_data_batch,Ybatch = get_data_B2(batch_file_name)
        else:
            condn_data_batch,Ybatch = np.empty([0,96]),np.empty([0,4])
            
        # condn_data_total = np.concatenate((condn_data_total,condn_data_imagined,
        #                                    condn_data_online,condn_data_batch),axis=0)
        # Ytotal = np.concatenate((Ytotal,Yimagined,
        #                                    Yonline,Ybatch),axis=0)
        condn_data_total = np.concatenate((condn_data_total,condn_data_imagined)
                                           ,axis=0)
        Ytotal = np.concatenate((Ytotal,Yimagined),axis=0)
    
    # # get testing data 
    # condn_data_heldout=np.empty((0,96))
    # Yheldout = np.empty((0,7))
    # for i in np.arange(len(test_days)):
    #     # imagined_file_name = root_path + root_imag_filename +  str(test_days[i]) + '.mat'
    #     # condn_data_imagined,Yimagined = get_data(imagined_file_name)    
    #     online_file_name = root_path + root_online_filename +  str(test_days[i]) + '.mat'
    #     condn_data_online,Yonline = get_data(online_file_name)
    #     batch_file_name = root_path + root_batch_filename +  str(test_days[i]) + '.mat'
    #     condn_data_batch,Ybatch = get_data(batch_file_name)  
        
    #     condn_data_heldout = np.concatenate((condn_data_heldout,condn_data_online,condn_data_batch),axis=0)
    #     Yheldout = np.concatenate((Yheldout, Yonline,Ybatch),axis=0)
        
    
    # build iAE    
    if 'model' in locals():
        del model       
    nn_filename = 'iAE_AcrossDays' + str(days) + '.pth'
    Ytest = np.zeros((2,2))
    while len(np.unique(np.argmax(Ytest,axis=1)))<num_classes:
        Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_total,Ytotal,0.8)      
    model = iAutoencoder(input_size,hidden_size,latent_dims,num_classes).to(device)                              
    model,acc = training_loop_iAE(model,num_epochs,batch_size,learning_rate,batch_val,
                          patience,gradient_clipping,nn_filename,
                          Xtrain,Ytrain,Xtest,Ytest,
                          input_size,hidden_size,latent_dims,num_classes)

    D,z,idx,fig_imagined,acc_train = plot_latent_acc(model,condn_data_total,Ytotal,
                                       latent_dims) 
    if plt_close == True:
        plt.close()
    
    # # test it on held out data all at once 
    # D,z,idx,fig_test,acc_test = plot_latent_acc(model,condn_data_heldout,Yheldout,
    #                                    latent_dims) 
    # if plt_close == True:
    #     plt.close()     
    # latent_acc_days.append(acc_test*100)
    # Sil_days.append(D)
    
    # mahab_distances = get_mahab_distance_latent(z,idx)
    # mahab_distances = mahab_distances[np.triu_indices(mahab_distances.shape[0])]
    # mahab_distances = mahab_distances[mahab_distances>0]
    # mahab_distances_days.append(mahab_distances)
    
    # test it out on held out data one day at a time 
    mahab_tmp=[]
    latent_acc_tmp=[]
    sil_tmp=[]
    for i in np.arange(len(test_days)):
            
        condn_data_heldout=np.empty((0,96))
        Yheldout = np.empty((0,4))        
        imagined_file_name = root_path + root_imag_filename +  str(test_days[i]) + '.mat'
        condn_data_imagined,Yimagined = get_data_B2(imagined_file_name)    
        online_file_name = root_path + root_online_filename +  str(test_days[i]) + '.mat'
        condn_data_online,Yonline = get_data_B2(online_file_name)
        batch_file_name = root_path + root_batch_filename +  str(test_days[i]) + '.mat'
        if os.path.exists(batch_file_name):
            condn_data_batch,Ybatch = get_data_B2(batch_file_name)
        else:
            condn_data_batch,Ybatch = np.empty([0,96]),np.empty([0,4])
        
        
        # condn_data_heldout = np.concatenate((condn_data_heldout,condn_data_online,condn_data_batch),axis=0)
        # Yheldout = np.concatenate((Yheldout, Yonline,Ybatch),axis=0)
        condn_data_heldout = np.concatenate((condn_data_heldout,condn_data_imagined),axis=0)
        Yheldout = np.concatenate((Yheldout, Yimagined),axis=0)
        
        D,z,idx,fig_test,acc_test = plot_latent_acc(model,condn_data_heldout,Yheldout,latent_dims) 
        
        if plt_close == True:
            plt.close()     
        latent_acc_tmp.append(acc_test*100)
        sil_tmp.append(D)
        
        # # for figure plotting
        # ch =[0,2,6]
        # fig_ex = plot_latent_select(model,condn_data_heldout,Yheldout,latent_dims,ch)        
        # fig_ex.axes[0].xaxis.set_ticklabels([])
        # fig_ex.axes[0].yaxis.set_ticklabels([])
        # fig_ex.axes[0].zaxis.set_ticklabels([])
        # fig_ex.axes[0].view_init(elev=24, azim=-130)
        # plt.show()
        # image_format = 'svg' # e.g .png, .svg, etc.
        # image_name = 'Hand_Day10_Days1thru9_AE.svg'
        # fig_ex.savefig(image_name, format=image_format, dpi=300)

        
        mahab_distances = get_mahab_distance_latent_B2(z,idx)
        mahab_distances = mahab_distances[np.triu_indices(mahab_distances.shape[0])]
        mahab_distances = mahab_distances[mahab_distances>0]
        mahab_tmp.append(np.median(mahab_distances))
    
    #mahab_distances_days.append(np.median(mahab_distances))
    mahab_distances_days.append((mahab_tmp))
    Sil_days.append(np.array(sil_tmp))
    latent_acc_days.append(np.array(latent_acc_tmp))

       

# plt.figure();
# plt.plot(latent_acc_days);
# plt.ylim([35,60])
        
# plt.figure();
# mahab_distances_days1 = np.array(mahab_distances_days)
# a=np.median(mahab_distances_days1,axis=1)
# plt.plot(np.median(mahab_distances_days1,axis=1));
# plt.plot(mahab_distances_days1);


acc_plot=[]
for i in np.arange(len(latent_acc_days)):
    tmp = latent_acc_days[i]
    acc_plot.append(mean(tmp))

plt.plot(acc_plot)   


acc_plot=[]
for i in np.arange(len(latent_acc_days)):
    tmp = latent_acc_days[i]
    acc_plot.append(mean(tmp))

plt.plot(acc_plot)

# mahab plot on all 
mahab_plot=[]
fig=plt.figure()
hfont = {'fontname':'Arial'}
plt.rc('font',family='Arial')
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 6})
days=np.arange(1,len(total_days))
for i in np.arange(len(mahab_distances_days)):
    tmp = mahab_distances_days[i]
    mahab_plot.append(median(tmp))
    I = days[i]*np.ones(len(tmp)) + 0.00*rnd.randn(len(tmp))
    plt.scatter(I,tmp,c='k',alpha=0.5,edgecolors='none')
    
# do curve fitting and plot regression line
x=days
y=np.array(mahab_plot)
p = np.polyfit(x,y,1)
#p = [0.3686,6.3109] #from Robust regression below
xx = np.concatenate((np.ones((len(x),1)),x[:,None]),axis=1)
yhat = xx @ np.flip(p)[:,None]
plt.plot(days,yhat,c='k')
plt.xticks(ticks=np.arange(len(total_days)-1)+1)
fig.axes[0].xaxis.set_ticklabels(['1-1','1-2','1-3','1-4','1-5'])
#fig.axes[0].xaxis.set_ticklabels([])
#fig.axes[0].yaxis.set_ticklabels([])
plt.xlabel('Days used to build latent space')
plt.ylabel('Mahab distance')
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'B2_Drift_AE_Days1thru5.svg'
fig.savefig(image_name, format=image_format, dpi=300)

# get the pval for the regression
lm = sm.OLS(y,xx).fit()
print(lm.summary())

rlm = sm.RLM(y,xx).fit()
print(rlm.summary())


# mahab plot on day 9 
mahab_plot=[]
fig=plt.figure()
days=np.arange(1,10)
for i in np.arange(len(mahab_distances_days)):
    tmp = mahab_distances_days[i]
    mahab_plot.append(tmp[-1])    

plt.figure()
plt.plot(mahab_plot)   
    
#%% LOOKING AT THE STABILITY OF THE MANIFOLD ACROSS DAYS  (TESTING)
# plan here is to build a manifold for each day and then examine the orthgonal 
# procrustus distance between the layers of the manifold 

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
# setting up GPU
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

# get the first day's data and manifold
days=3
imagined_file_name = root_path + root_imag_filename +  str(days) + '.mat'
condn_data_imagined,Yimagined = get_data(imagined_file_name,num_classes)
online_file_name = root_path + root_online_filename +  str(days) + '.mat'
condn_data_online,Yonline = get_data(online_file_name,num_classes)
batch_file_name = root_path + root_batch_filename +  str(days) + '.mat'
condn_data_batch,Ybatch = get_data(batch_file_name,num_classes)    
nn_filename = 'iAE_' + str(days) + '.pth'
# augment
condn_data_online,Yonline =   data_aug_mlp_chol_feature_equalSize(condn_data_online,Yonline,condn_data_imagined.shape[0])
condn_data_batch,Ybatch =   data_aug_mlp_chol_feature_equalSize(condn_data_batch,Ybatch,condn_data_imagined.shape[0])
# assign to train and testing
condn_data_imagined_train,condn_data_imagined_test = condn_data_imagined,condn_data_imagined
Yimagined_train,Yimagined_test = Yimagined,Yimagined
condn_data_online_train,condn_data_online_test = condn_data_online,condn_data_online
Yonline_train,Yonline_test = Yonline,Yonline
condn_data_batch_train,condn_data_batch_test = condn_data_batch,condn_data_batch
Ybatch_train,Ybatch_test = Ybatch,Ybatch


#### STACK EVERYTHING TOGETHER ###
condn_data_total = np.concatenate((condn_data_imagined_train,condn_data_online_train,condn_data_batch_train),axis=0)    
Ytotal = np.concatenate((Yimagined_train,Yonline_train,Ybatch_train),axis=0)            
        
#### TRAIN THE MODEL
Ytest = np.zeros((2,2))
while len(np.unique(np.argmax(Ytest,axis=1)))<num_classes:
    Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_total,Ytotal,0.8)                        
    
if 'model' in locals():
    del model    
model1 = iAutoencoder(input_size,hidden_size,latent_dims,num_classes).to(device)        
model1,acc = training_loop_iAE(model1,num_epochs,batch_size,learning_rate,batch_val,
                      patience,gradient_clipping,nn_filename,
                      Xtrain,Ytrain,Xtest,Ytest,
                      input_size,hidden_size,latent_dims,num_classes)



# get the second day's data and manifold
days=9
imagined_file_name = root_path + root_imag_filename +  str(days) + '.mat'
condn_data_imagined,Yimagined = get_data(imagined_file_name,num_classes)
online_file_name = root_path + root_online_filename +  str(days) + '.mat'
condn_data_online,Yonline = get_data(online_file_name,num_classes)
batch_file_name = root_path + root_batch_filename +  str(days) + '.mat'
condn_data_batch,Ybatch = get_data(batch_file_name,num_classes)    
nn_filename = 'iAE_' + str(days) + '.pth'
# augment
condn_data_online,Yonline =   data_aug_mlp_chol_feature_equalSize(condn_data_online,Yonline,condn_data_imagined.shape[0])
condn_data_batch,Ybatch =   data_aug_mlp_chol_feature_equalSize(condn_data_batch,Ybatch,condn_data_imagined.shape[0])
# assign to train and testing
condn_data_imagined_train,condn_data_imagined_test = condn_data_imagined,condn_data_imagined
Yimagined_train,Yimagined_test = Yimagined,Yimagined
condn_data_online_train,condn_data_online_test = condn_data_online,condn_data_online
Yonline_train,Yonline_test = Yonline,Yonline
condn_data_batch_train,condn_data_batch_test = condn_data_batch,condn_data_batch
Ybatch_train,Ybatch_test = Ybatch,Ybatch


#### STACK EVERYTHING TOGETHER ###
condn_data_total = np.concatenate((condn_data_imagined_train,condn_data_online_train,condn_data_batch_train),axis=0)    
Ytotal = np.concatenate((Yimagined_train,Yonline_train,Ybatch_train),axis=0)            
        
#### TRAIN THE MODEL
Ytest = np.zeros((2,2))
while len(np.unique(np.argmax(Ytest,axis=1)))<num_classes:
    Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_total,Ytotal,0.8)                        
    
if 'model' in locals():
    del model    
model2 = iAutoencoder(input_size,hidden_size,latent_dims,num_classes).to(device)        
model2,acc = training_loop_iAE(model2,num_epochs,batch_size,learning_rate,batch_val,
                      patience,gradient_clipping,nn_filename,
                      Xtrain,Ytrain,Xtest,Ytest,
                      input_size,hidden_size,latent_dims,num_classes)



# examine the procrustus distance between the two 
model1.eval()
model2.eval()
elu = nn.ELU()
input_data = torch.randn(200,96).float().to(device)
#input_data = torch.from_numpy(condn_data_imagined[:200,:]).float().to(device)
X = elu(model1.encoder.linear1(input_data)).to('cpu').detach().numpy()
Y = elu(model2.encoder.linear1(input_data)).to('cpu').detach().numpy()

X = X - np.mean(X,axis=0)
Y = Y - np.mean(X,axis=0)
#plt.plot(np.mean(X,axis=0))
a=lin.norm((X.T@X),'fro')
b=lin.norm((Y.T@Y),'fro')
c=(lin.norm((Y.T @ X),'fro'))**2
d = c/(a*b)
print(d)


# compare to null by shuffling weights and bias
aa = torch.randn(48,96).to(device).float()
bb = torch.randn(48).to(device).float()
# matrix multiplication 
layer1_day1 = (torch.matmul(input_data,torch.transpose(aa,0,1)) 
               + bb).to('cpu').detach().numpy().T
aa = torch.randn(48,96).to(device).float()
bb = torch.randn(48).to(device).float()
layer1_day2 = (torch.matmul(input_data,torch.transpose(aa,0,1)) 
               + bb).to('cpu').detach().numpy().T
#layer1_day2 = model2.encoder.linear1(input_data).to('cpu').detach().numpy().T

a=lin.norm(layer1_day1 @ layer1_day2.T)
b=lin.norm(layer1_day1 @ layer1_day1.T)
c=lin.norm(layer1_day2 @ layer1_day2.T)
d= 1-(a/(b*c))
print(d)


# Linear CKA between two random matrices
random_data=[]
for i in range(1000):        
    X = (torch.randn(200,48)).to('cpu').detach().numpy()
    Y = (torch.randn(200,48)).to('cpu').detach().numpy()
    X = X - np.mean(X,axis=0)
    Y = Y - np.mean(X,axis=0)
    #plt.plot(np.mean(X,axis=0))
    a=lin.norm((X.T@X),'fro')
    b=lin.norm((Y.T@Y),'fro')
    c=(lin.norm((Y.T @ X),'fro'))**2
    d = c/(a*b)
    random_data.append(d)

plt.hist(random_data)


# Linear CKA passing real data through random weights 
input_data = torch.from_numpy(condn_data_imagined).float().to(device)
random_data=[]
for i in range(1000):            
    aa = torch.randn(48,96).to(device).float()
    bb = torch.randn(48).to(device).float()
    X = elu(torch.matmul(input_data,torch.transpose(aa,0,1)) 
                   + bb).to('cpu').detach().numpy()
    aa = torch.randn(48,96).to(device).float()
    bb = torch.randn(48).to(device).float()
    Y = elu(torch.matmul(input_data,torch.transpose(aa,0,1)) 
                   + bb).to('cpu').detach().numpy()    
    
    X = X - np.mean(X,axis=0)
    Y = Y - np.mean(X,axis=0)
    #plt.plot(np.mean(X,axis=0))
    a=lin.norm((X.T@X),'fro')
    b=lin.norm((Y.T@Y),'fro')
    c=(lin.norm((Y.T @ X),'fro'))**2
    d = c/(a*b)
    random_data.append(d)

plt.hist(random_data)

# Linear CKA between activation of two neural networks across two diff. days
real_data=[]
for i in range(1000):
    #input_data = torch.randn(200,96).float().to(device)   
    input_data = torch.from_numpy(condn_data_imagined).float().to(device)
    X = elu(model1.encoder.linear1(input_data)).to('cpu').detach().numpy()
    #Y = elu(model2.encoder.linear1(input_data)).to('cpu').detach().numpy()
    Y = elu(model2.encoder.linear1(input_data))
    Y = elu(model2.encoder.linear2(Y)).to('cpu').detach().numpy()
    X = X - np.mean(X,axis=0)
    Y = Y - np.mean(Y,axis=0)    
    a=lin.norm((X.T@X),'fro')
    b=lin.norm((Y.T@Y),'fro')
    c=(lin.norm((Y.T @ X),'fro'))**2
    d = c/(a*b)
    real_data.append(d)

plt.figure()
plt.hist(real_data)

print(1-(np.sum(np.mean(real_data) > random_data) / len(random_data)))
