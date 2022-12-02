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


#%% # PART 2 combining an action from two separate days and looking at separability 

days=1
imagined_file_name = root_path + root_imag_filename +  str(days) + '.mat'
condn_data_imagined,Yimagined = get_data(imagined_file_name)
a1 = np.argmax(Yimagined,axis=1)
a1 = np.array([np.where(a1==0)[0]]).flatten()
condn_data_imagined_day1 = condn_data_imagined[a1,:]
Yimagined_day1 = Yimagined[a1,:]

days=4
imagined_file_name = root_path + root_imag_filename +  str(days) + '.mat'
condn_data_imagined,Yimagined = get_data(imagined_file_name)
a2 = np.argmax(Yimagined,axis=1)
a2 = np.array([np.where(a2==0)[0]]).flatten()
condn_data_imagined_day2 = condn_data_imagined[a2,:]
Yimagined_day2 = Yimagined[a2,:]
Yimagined_day2 = np.roll(Yimagined_day2,1)

days=8
imagined_file_name = root_path + root_imag_filename +  str(days) + '.mat'
condn_data_imagined,Yimagined = get_data(imagined_file_name)
a2 = np.argmax(Yimagined,axis=1)
a2 = np.array([np.where(a2==0)[0]]).flatten()
condn_data_imagined_day3 = condn_data_imagined[a2,:]
Yimagined_day3 = Yimagined[a2,:]
Yimagined_day3 = np.roll(Yimagined_day3,2)


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

D,z,idx,fig_imagined = plot_latent(model,condn_data_imagined_total,Yimagined_total,
                                   condn_data_imagined_total[0],latent_dims)        

fig_imagined.axes[0].xaxis.set_ticklabels([])
fig_imagined.axes[0].yaxis.set_ticklabels([])
plt.show()
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'LatentRtHand_Days148.svg'
fig_imagined.savefig(image_name, format=image_format, dpi=300)

# D,z,idx,fig_imagined = plot_latent(model,condn_data_imagined_day1,Yimagined_day1,
#                                    condn_data_imagined_day1.shape[0],latent_dims)     

delta_recon_imag,beta_recon_imag,hg_recon_imag = return_recon(model,
                                              condn_data_imagined_total,Yimagined_total)

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
fig.savefig(image_name, format=image_format, dpi=300)

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
fig.savefig(image_name, format=image_format, dpi=300)

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

#%% CONTINUATION OF PART C BUT NOW ON ALL THE DATA 

from iAE_utils_models import *
mahab_distances_days = []
Sil_days = []
latent_acc_days = []
total_days = 10
batch_size=64
patience = 5
latent_dims = 3
plt_close=True
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
        
        condn_data_total = np.concatenate((condn_data_total,condn_data_imagined,
                                           condn_data_online,condn_data_batch),axis=0)
        Ytotal = np.concatenate((Ytotal,Yimagined,
                                           Yonline,Ybatch),axis=0)
    
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
        # imagined_file_name = root_path + root_imag_filename +  str(test_days[i]) + '.mat'
        # condn_data_imagined,Yimagined = get_data(imagined_file_name)    
        online_file_name = root_path + root_online_filename +  str(test_days[i]) + '.mat'
        condn_data_online,Yonline = get_data(online_file_name)
        batch_file_name = root_path + root_batch_filename +  str(test_days[i]) + '.mat'
        condn_data_batch,Ybatch = get_data(batch_file_name)  
        
        condn_data_heldout = np.concatenate((condn_data_heldout,condn_data_online,condn_data_batch),axis=0)
        Yheldout = np.concatenate((Yheldout, Yonline,Ybatch),axis=0)
        
        D,z,idx,fig_test,acc_test = plot_latent_acc(model,condn_data_heldout,Yheldout,latent_dims) 
        if plt_close == True:
            plt.close()     
        latent_acc_tmp.append(acc_test*100)
        sil_tmp.append(D)
        
        # for figure plotting
        ch =[0,2,6]
        fig_ex = plot_latent_select(model,condn_data_heldout,Yheldout,latent_dims,ch)        
        fig_ex.axes[0].xaxis.set_ticklabels([])
        fig_ex.axes[0].yaxis.set_ticklabels([])
        fig_ex.axes[0].zaxis.set_ticklabels([])
        fig_ex.axes[0].view_init(elev=24, azim=-130)
        plt.show()
        image_format = 'svg' # e.g .png, .svg, etc.
        image_name = 'Hand_Day10_Days1thru9_AE.svg'
        fig_ex.savefig(image_name, format=image_format, dpi=300)

        
        mahab_distances = get_mahab_distance_latent(z,idx)
        mahab_distances = mahab_distances[np.triu_indices(mahab_distances.shape[0])]
        mahab_distances = mahab_distances[mahab_distances>0]
        mahab_tmp.append(np.median(mahab_distances))
    
    #mahab_distances_days.append(np.median(mahab_distances))
    mahab_distances_days.append((mahab_tmp))
    Sil_days.append(np.array(sil_tmp))
    latent_acc_days.append(np.array(latent_acc_tmp))

        

plt.figure();
plt.plot(latent_acc_days);
plt.ylim([35,60])
        
plt.figure();
mahab_distances_days1 = np.array(mahab_distances_days)
a=np.median(mahab_distances_days1,axis=1)
plt.plot(np.median(mahab_distances_days1,axis=1));
plt.plot(mahab_distances_days1);


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
days=np.arange(1,10)
for i in np.arange(len(mahab_distances_days)):
    tmp = mahab_distances_days[i]
    mahab_plot.append(median(tmp))
    I = days[i]*np.ones(len(tmp)) + 0.01*rnd.randn(len(tmp))
    plt.scatter(I,tmp,c='k')

plt.figure()
plt.plot(mahab_plot)   
    
# mahab plot on day 9 
mahab_plot=[]
fig=plt.figure()
days=np.arange(1,10)
for i in np.arange(len(mahab_distances_days)):
    tmp = mahab_distances_days[i]
    mahab_plot.append(tmp[-1])    

plt.figure()
plt.plot(mahab_plot)   
    
    
    


    
    






