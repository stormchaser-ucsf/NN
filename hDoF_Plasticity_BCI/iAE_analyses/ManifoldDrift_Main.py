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
import pickle

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
#root_imag_filename = '\Biomimetic_CenterOut_condn_data_Imagined_Day'

#%% MAIN LOOP TO GET THE DATA

pval_results={}
simal_res={}
recon_res={}
num_days=10
import time
t0 = time.time()
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
    # condn_data_total = np.concatenate((condn_data_imagined,condn_data_online,condn_data_batch),axis=0)    
    # Ytotal = np.concatenate((Yimagined,Yonline,Ybatch),axis=0) 
    condn_data_total = np.concatenate((condn_data_online,condn_data_batch),axis=0)    
    Ytotal = np.concatenate((Yonline,Ybatch),axis=0)    

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
    
    # D,z,idx,fig_model = plot_latent(model,condn_data_online,Yonline,
    #                                    condn_data_online.shape[0],latent_dims)        


    for j in np.arange(i+1,11): #COMPARISON TO OTHER DAYS, LAYER BY LAYER
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
        # condn_data_total1 = np.concatenate((condn_data_imagined1,condn_data_online1,condn_data_batch1),axis=0)    
        # Ytotal1 = np.concatenate((Yimagined1,Yonline1,Ybatch1),axis=0) 
        condn_data_total1 = np.concatenate((condn_data_online1,condn_data_batch1),axis=0)    
        Ytotal1 = np.concatenate((Yonline1,Ybatch1),axis=0)   

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
        
        # SIMILARITY BETWEEN THE LAYERS OF THE AUTOENCODERS PAIRWISE
        shuffle_flag=False;shuffle_flag1=False
        d1 = linear_cka_dist(condn_data_total,model,model1,shuffle_flag,shuffle_flag1)
        d2 = linear_cka_dist(condn_data_total1,model,model1,shuffle_flag,shuffle_flag1)
        d = (d1+d2)/2
        #print(np.argmax(d1,axis=1))
        #plt.imshow(d,cmap='magma')
        dmain=np.diag(d)
        print(dmain)
        
        
        
        
        # NEW FOR REVIEWER 4, BOOT STATISTICS AFTER SHUFFLING BETWEEN AE
        boot_val = np.zeros((500,6))
        for boot in np.arange(boot_val.shape[0]):
            print(boot)
            d1 = linear_cka_dist_shuffleLayers_Between_AE(condn_data_total,model,model1)
            d2 = linear_cka_dist_shuffleLayers_Between_AE(condn_data_total1,model,model1)
            d = (d1+d2)/2
            boot_val[boot,:] = d
        
        
        
        # GETTING THE BOOT STATISTICS AFTER SHUFFLING THE WEIGHTS OF THE AE           
        # boot_val = np.zeros((100,6))
        # for boot in np.arange(boot_val.shape[0]):
        #     print(boot)
        #     shuffle_flag=True;shuffle_flag1=True
        #     d1 = linear_cka_dist(condn_data_total,model,model1,shuffle_flag,shuffle_flag1)
        #     shuffle_flag=True;shuffle_flag1=True
        #     d2 = linear_cka_dist(condn_data_total1,model,model1,shuffle_flag,shuffle_flag1)
        #     d = (d1+d2)/2
        #     boot_val[boot,:] = np.diag(d)
        
        # HISTOGRAM
        pval=[]
        for k in np.arange(boot_val.shape[1]):
            fig=plt.figure()
            plt.hist(boot_val[:,k])
            p = 1 - np.sum(dmain[k]>=boot_val[:,k])/boot_val.shape[0]
            plt.axvline(x = dmain[k], color = 'r')
            plt.xlim((0,1))
            plt.xticks(ticks=[0,.2,.4,.6,.8,1])
            #plt.yticks(ticks=np.arange(0,251,50))
            #plt.tick_params(labelbottom=False)
            #plt.tick_params(labelleft=False)
            plt.title(str(p))
            pval.append(p)
            # image_format = 'svg' # e.g .png, .svg, etc.
            # image_name = 'Layer_'  + str(k+1) + '.svg'
            # fig.savefig(image_name, format=image_format, dpi=300)
            plt.close()
       
        # SIMILARITY OF THE RECON TO THE AVERAGE MAP....thru its own or another day AE
        orig, origManifold,swappedManifold = eval_ae_similarity(model,model1,condn_data_total,
                                                                condn_data_total1,Ytotal,Ytotal1)    
        # # SIMILARITY AFTER SHUFFLING AE
        # model_shuffle,model1_shuffle = shuffle_weights(model,model1)
        # orig_s, origManifold_s,swappedManifold_s = eval_ae_similarity(model_shuffle,model1_shuffle,condn_data_total,
        #                                                         condn_data_total1,Ytotal,Ytotal1)    
        
        
        # plt.figure();
        # plt.boxplot([orig, origManifold,swappedManifold])
        
        # STORE RESULTS
        pval_results[i-1,j-1] = pval
        simal_res[i-1,j-1] = dmain
        recon_res[i-1,j-1] = [orig, origManifold,swappedManifold]
        
        # RECONSTRUCTION ERROR FOR A DAY ON ITS OWN MODEL COMPARED TO ANOTHER DAY
        # input_data = torch.from_numpy(condn_data_total).to(device).float()
        # z,y=model(input_data)
        # recon = z.to('cpu').detach().numpy()
        # z1,y1=model1(input_data)
        # recon1 = z1.to('cpu').detach().numpy()        
        # e = recon - recon1
        # a = lin.norm(recon,'fro')
        # b = lin.norm(e,'fro')
        # recon_error = b/a
        # print(recon_error)
        
        
        
            
        
        # how similar are the layers of the root AE to this query AE?         
        # d1 = linear_cka_dist2(condn_data_imagined,condn_data_imagined1,model,model1)
        # d2 = linear_cka_dist2(condn_data_imagined,condn_data_imagined1,model,model1)
        
        
        # idx = np.argmax(Yimagined,axis=1)
        # idx = np.where(idx==3)[0]
        # A = condn_data_imagined[idx,:]
        # B = Yimagined[idx,:]
        # plot_latent(model, condn_data_batch, Ybatch, condn_data_batch.shape[0], 2)
        # plot_latent(model1, condn_data_batch, Ybatch, condn_data_batch.shape[0], 2)
        # A = torch.from_numpy(condn_data_batch).float().to(device)
        # z=model.encoder(A).to('cpu').detach().numpy()
        # z1=model1.encoder(A).to('cpu').detach().numpy()
        
        # d1 = linear_cka_dist(A,model,model1)
        # d2 = linear_cka_dist(A,model,model1)
        
        # # in1 = rnd.randn(2000,96)
        # # in2 = rnd.randn(2000,96)
        # # d1 = linear_cka_dist2(in1,in2,model,model1)
        # # d2 = linear_cka_dist2(in2,in2,model,model1)
        
        # d = (d1+d2)/2
        # print(np.argmax(d1,axis=1))
        # plt.imshow(d,cmap='magma')
        
        
        # plot_latent(model, condn_data_online1, Yonline1, condn_data_online1.shape[0], 2)
        # plot_latent(model1, condn_data_online1, Yonline1, condn_data_online1.shape[0], 2)
        
        
t1 = time.time()
time_taken = t1-t0        
print(str(time_taken) + 's')


# pval_results1 = pval_results.items()
# pval_results1 = list(pval_results1)
# pval_results1 = np.array(pval_results1,dtype=object)

pval_results = np.array(list(pval_results.items()),dtype=object)
simal_res = np.array(list(simal_res.items()),dtype=object)
recon_res = np.array(list(recon_res.items()),dtype=object)
np.savez('ManifoldAnalyses_Main_1000Boot_PaperRevision1', 
         pval_results = pval_results,
         simal_res = simal_res,
         recon_res = recon_res)

#%% PLOTTING THE RESULTS 

from iAE_utils_models import *
data =np.load('ManifoldAnalyses_Main_1000Boot_PaperRevision1.npz',allow_pickle=True)
pval_results = data.get('pval_results')
simal_res = data.get('simal_res')
recon_res = data.get('recon_res')

recon_raw=np.array([])
recon_ae_orig=np.array([])
recon_ae_swap=np.array([])
for i in np.arange(recon_res.shape[0]):
    tmp = recon_res[i][1]
    recon_raw = np.append(recon_raw,tmp[0])
    recon_ae_orig = np.append(recon_ae_orig,tmp[1])
    recon_ae_swap = np.append(recon_ae_swap,tmp[2])

plt.figure()
plt.boxplot([recon_raw,recon_ae_orig,recon_ae_swap])
print(stats.ttest_rel(recon_ae_swap,recon_raw))
print(np.mean([recon_raw,recon_ae_orig,recon_ae_swap],axis=1))


pval=np.array([])
for i in np.arange(pval_results.shape[0]):
    pval = np.append(pval,pval_results[i][1][:-1])

plt.figure()
plt.hist(pval)
pfdr_out,pfdr = fdr(pval,alpha=0.05)
print(pfdr_out)
print(sum(pfdr_out==True)/len(pfdr_out))

simal=np.array([])
for i in np.arange(simal_res.shape[0]):
    simal = np.append(simal,simal_res[i][1][1])
plt.figure()
plt.hist(simal)


pfdr,pfdr_thresh=fdr_threshold(pval,0.05,'Parametric')
prop_res=  np.zeros((len(pval_results),6))
for i in np.arange(len(pval_results)):
    tmp = pval_results[i][1][:-1]
    tmp = np.sum(tmp<=pfdr)
    prop_res[i,tmp] = 1
a= np.sum(prop_res,axis=0)/prop_res.shape[0]
plt.figure();
plt.bar(np.arange(len(a)),a);
plt.ylim((0,1))
plt.xlabel('No. of sig. similar layers')
plt.ylabel('Prop. of across-days pairwise comparisons')
print(a[:,None])


# pval_results = np.array([1,2,3])
# simal_res = np.array([12,3])
# np.savez('ManifoldAnalyses_Main_test', 
#          pval_results = pval_results,
#          simal_res = simal_res)

# data =np.load('ManifoldAnalyses_Main_test.npz')
# pval_results = data.get('pval_results')
# simal_res = data.get('simal_res')

# D = np.zeros((10,10))
# for i in np.arange(10):    
#     for j in np.arange(i+1,10):
#         tmp = simal_res[i,j]
#         D[i,j] = np.mean(tmp)
#         D[j,i] = np.mean(tmp)

# plt.figure()
# plt.imshow(D,aspect='auto',cmap='viridis',vmin=0.5,vmax=0.8)
# plt.colorbar()        

# plt.stem(D[4,:])

#%% PLOTTING COMBINED STATS FOR B1 AND B2 and B3

import os
os.chdir('C:/Users/nikic/Documents/GitHub/NN/hDoF_Plasticity_BCI/iAE_analyses')
from iAE_utils_models import *

#B1
data =np.load('ManifoldAnalyses_Main_1000Boot_PaperRevision1.npz',allow_pickle=True)
pval_results = data.get('pval_results')
simal_res = data.get('simal_res')
recon_res = data.get('recon_res')
p =0.05

pval=np.array([])
for i in np.arange(pval_results.shape[0]):
    pval = np.append(pval,pval_results[i][1])


pfdr,pfdr_thresh=fdr_threshold(pval,p,'Parametric')
prop_res=  np.zeros((len(pval_results),6))
for i in np.arange(len(pval_results)):
    tmp = pval_results[i][1][:-1]
    tmp = np.sum(tmp<=pfdr)
    prop_res[i,tmp] = 1
a= np.sum(prop_res,axis=0)/prop_res.shape[0]
plt.figure();
plt.bar(np.arange(len(a)),a);
print(a[:,None])
b1 = a


#B2
data =np.load('ManifoldAnalyses_Main_B2_Days2to5_1000Boot_Revision1.npz',allow_pickle=True)
pval_results = data.get('pval_results')
simal_res = data.get('simal_res')
recon_res = data.get('recon_res')


pval=np.array([])
for i in np.arange(pval_results.shape[0]):
    pval = np.append(pval,pval_results[i][1][:-1])


pfdr,pfdr_thresh=fdr_threshold(pval,p,'Parametric')
prop_res=  np.zeros((len(pval_results),6))
for i in np.arange(len(pval_results)):
    tmp = pval_results[i][1][:-1]
    tmp = np.sum(tmp<=pfdr)
    prop_res[i,tmp] = 1
a= np.sum(prop_res,axis=0)/prop_res.shape[0]
plt.figure();
plt.bar(np.arange(len(a)),a);
print(a[:,None])
b2 = a

#B3
data = np.load('ManifoldAnalyses_Main_B3.npz',allow_pickle=True)
pval_results = data.get('pval_results')
simal_res = data.get('simal_res')
recon_res = data.get('recon_res')

pval=np.array([])
for i in np.arange(pval_results.shape[0]):
    pval = np.append(pval,pval_results[i][1][:-1])


pfdr,pfdr_thresh=fdr_threshold(pval,p,'Parametric')
prop_res=  np.zeros((len(pval_results),6))
for i in np.arange(len(pval_results)):
    tmp = pval_results[i][1][:-1]
    tmp = np.sum(tmp<=pfdr)
    prop_res[i,tmp] = 1
a= np.sum(prop_res,axis=0)/prop_res.shape[0]
plt.figure();
plt.bar(np.arange(len(a)),a);
print(a[:,None])
b3 = a


# plotting b1, b2, b3
res = np.vstack((b1[None,:][0],b2[None,:][0],b3[None,:][0]))
m = np.mean(res,axis=0)
t = np.arange(6)
fig=plt.figure();
plt.bar(t,m,width=0.8,color=(0.5,0.5,0.5));   
plt.scatter(t+(rnd.randn(len(t))*0.05), res[0,:],color = (0.2,0.2,0.8))
plt.scatter(t+(rnd.randn(len(t))*0.05), res[1,:],color=(0.8,0.2,0.2))
plt.scatter(t+(rnd.randn(len(t))*0.05), res[2,:],color=(0.8,0.2,0.8))
plt.xticks(t)
plt.yticks(np.arange(0,1.2,0.2))
plt.ylim((0,1.05))
plt.show()
plt.tick_params(labelbottom=False)
plt.tick_params(labelleft=False)
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'CKA_res_B1B2B3.svg'
fig.savefig(image_name, format=image_format, dpi=300)

# plotting b1, b3
res = np.vstack((b1[None,:][0],b3[None,:][0]))
m = np.mean(res,axis=0)
t = np.arange(6)
fig=plt.figure();
plt.bar(t,m,width=0.8,color=(0.5,0.5,0.5));   
plt.scatter(t+(rnd.randn(len(t))*0.05), res[0,:],color = (0.8,0.2,0.2))
plt.scatter(t+(rnd.randn(len(t))*0.05), res[1,:],color=(0.2,0.2,0.8))
plt.xticks(t)
plt.yticks(np.arange(0,1.2,0.2))
plt.ylim((0,1.05))
plt.show()
plt.tick_params(labelbottom=False)
plt.tick_params(labelleft=False)
image_format = 'svg' # e.g .png, .svg, etc.
image_name = 'CKA_res_B1B3.svg'
fig.savefig(image_name, format=image_format, dpi=300)


#%% PLOTTING COMPARISONS BETWEEN CKD AND IBID FOR B1 IN CKA SIMAL VALUES


data =np.load('ManifoldAnalyses_Main_1000Boot.npz',allow_pickle=True)
pval_results = data.get('pval_results')
simal_res = data.get('simal_res')
recon_res = data.get('recon_res')

simal_IBID=np.empty([0,6])
for i in np.arange(simal_res.shape[0]):
    #simal_IBID = np.append(simal_IBID,simal_res[i][1])
    tmp = simal_res[i][1][None,:]
    simal_IBID = np.concatenate((simal_IBID,tmp),axis=0)



data =np.load('ManifoldAnalyses_Main_CKD_All_IntDirTowardsTarget45deg_1000boot.npz',allow_pickle=True)
pval_results = data.get('pval_results')
simal_res = data.get('simal_res')
recon_res = data.get('recon_res')

simal_CKD=np.empty([0,6])
for i in np.arange(simal_res.shape[0]):
    #simal_IBID = np.append(simal_IBID,simal_res[i][1])
    tmp = simal_res[i][1][None,:]
    simal_CKD = np.concatenate((simal_CKD,tmp),axis=0)


# plotting final comparisons
for i in np.arange(simal_CKD.shape[1]-1):
    plt.figure()
    plt.title(str(i))
    tmp = [simal_IBID[:,i],simal_CKD[:,i]]
    plt.boxplot(tmp)
    plt.ylim((0,1))
    print([np.mean(tmp[0]),np.mean(tmp[1])])
    
    
tmp = [simal_IBID.flatten(),simal_CKD.flatten()]   
plt.figure()
plt.boxplot(tmp)
plt.ylim((0,1))

# plot with confidence intervals 




    
    
    
    
