# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 21:00:10 2023

@author: nikic
"""

"""
EVALUATING SAMPLE LEVEL CLASSIFICATION ACCURACY WITH AN MLP EITHER WITH OR 
WITHOUT PASSING THE TEST DATA THROUGH AN iAE
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
num_epochs=150
batch_size=32
learning_rate = 1e-3
batch_val=512
patience=5
gradient_clipping=10

# model params
input_size=96
hidden_size=48
latent_dims=6
num_classes = 7

# model params for MLP
num_nodes = 64

# file location
root_path = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker'
root_imag_filename = '\condn_data_Imagined_Day'
root_online_filename = '\condn_data_Online_Day'
root_batch_filename = '\condn_data_Batch_Day'
#root_imag_filename = '\Biomimetic_CenterOut_condn_data_Imagined_Day'


#%% CYCLING THROUGH DAYS AND TESTING OUT THE METHOD 
# test on the batch data test to see how it fares out

res_overall = np.empty([0,4])
import time
for iter in np.arange(1):
    
    num_days=10
    decoding_results = np.zeros((num_days,4))
    for i in np.arange(num_days)+1: #ROOT DAYS
        # load the data
        print('Processing Day ' + str(i) + ' data')
        imagined_file_name = root_path + root_imag_filename +  str(i) + '.mat'
        condn_data_imagined,Yimagined = get_data(imagined_file_name,num_classes)
        online_file_name = root_path + root_online_filename +  str(i) + '.mat'
        condn_data_online,Yonline = get_data(online_file_name,num_classes)
        batch_file_name = root_path + root_batch_filename +  str(i) + '.mat'
        condn_data_batch,Ybatch = get_data(batch_file_name,num_classes)    
        nn_filename = 'mlp_' + str(i) + '.pth'   
        nn_iae_filename = 'iAE_' + str(i) + '.pth'   
        
        # data augment on the online data
        condn_data_online,Yonline =   data_aug_mlp_chol_feature_equalSize(condn_data_online,Yonline,condn_data_imagined.shape[0])
        #condn_data_batch,Ybatch =   data_aug_mlp_chol_feature_equalSize(condn_data_batch,Ybatch,condn_data_imagined.shape[0])
        
        # stack everything together 
        condn_data_total = np.concatenate((condn_data_imagined,condn_data_online),axis=0)    
        Ytotal = np.concatenate((Yimagined,Yonline),axis=0)     
        #condn_data_total = condn_data_imagined
        #Ytotal = Yimagined
    
        # demean
        #condn_data_total = condn_data_total - np.mean(condn_data_total,axis=0)                   
        
        #### train the iAE model         
        Ytest = np.zeros((2,2))
        while len(np.unique(np.argmax(Ytest,axis=1)))<num_classes:
            Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_total,Ytotal,0.8)                        
        
        t0=time.time()
        if 'model' in locals():
            del model 
        
        model = iAutoencoder(input_size,hidden_size,latent_dims,num_classes).to(device)        
        model,acc = training_loop_iAE(model,num_epochs,batch_size,learning_rate,batch_val,
                              patience,gradient_clipping,nn_filename,
                              Xtrain,Ytrain,Xtest,Ytest,
                              input_size,hidden_size,latent_dims,num_classes)
        t1=time.time()
        tt0=t1-t0
        
        ### train the MLP model
        t0=time.time()
        if 'model_mlp' in locals():
            del  model_mlp
        model_mlp = mlp_classifier_1layer(input_size,num_nodes,num_classes).to(device)        
        model_mlp,acc = training_loop_mlp(model_mlp,num_epochs,batch_size,learning_rate,batch_val,
                              patience+1,gradient_clipping,nn_filename,
                              Xtrain,Ytrain,Xtest,Ytest,
                              input_size,num_nodes,num_classes) 
        t1=time.time()
        tt=t1-t0
        
        ### train the MLP model after passing data thru AE    
        model.eval()
        z,idx = model(torch.from_numpy(condn_data_total).to(device).float())
        z = z.to('cpu').detach().numpy()        
        Ytest = np.zeros((2,2))
        while len(np.unique(np.argmax(Ytest,axis=1)))<num_classes:
             Xtrain,Xtest,Ytrain,Ytest = training_test_split(z,Ytotal,0.8)
        t0=time.time()
        if 'model_mlp1' in locals():
            del  model_mlp1
        model_mlp1 = mlp_classifier_1layer(input_size,num_nodes,num_classes).to(device)        
        model_mlp1,acc = training_loop_mlp(model_mlp1,num_epochs,batch_size,learning_rate,batch_val,
                              patience+1,gradient_clipping,nn_filename,
                              Xtrain,Ytrain,Xtest,Ytest,
                              input_size,num_nodes,num_classes) 
        t1=time.time()
        tt2=t1-t0
    
                              
        
        ### TEST THE MLP ON THE BATCH DATA
        model_mlp.eval()
        test_data = torch.from_numpy(condn_data_batch).to(device).float()
        decodes = model_mlp(test_data)
        # compute accuracy
        with torch.no_grad():
            ylabels = convert_to_ClassNumbers(torch.from_numpy(Ybatch).to(device).float())        
            ypred_labels = convert_to_ClassNumbers(decodes)     
            accuracy_noiAE = (torch.sum(ylabels == ypred_labels).item())/ylabels.shape[0]
        
        ### TEST THE MLP ON THE BATCH DATA AFTER IT HAS PASSED THRU IAE
        model.eval()
        model_mlp.eval()    
        test_data = torch.from_numpy(condn_data_batch).to(device).float()
        z,idx=model(test_data)
        decodes = model_mlp(z)
        # compute accuracy
        with torch.no_grad():
            ylabels = convert_to_ClassNumbers(torch.from_numpy(Ybatch).to(device).float())        
            ypred_labels = convert_to_ClassNumbers(decodes)     
            accuracy_iAE = (torch.sum(ylabels == ypred_labels).item())/ylabels.shape[0]
        
        
        ### TEST THE BATCH DATA ON A MLP TRAINED ON DATA PASSED THRU IAE
        model.eval()
        model_mlp1.eval()
        test_data = torch.from_numpy(condn_data_batch).to(device).float()
        z,idx=model(test_data)
        decodes = model_mlp1(z)
        # compute accuracy
        with torch.no_grad():
            ylabels = convert_to_ClassNumbers(torch.from_numpy(Ybatch).to(device).float())        
            ypred_labels = convert_to_ClassNumbers(decodes)     
            accuracy_iAE2 = (torch.sum(ylabels == ypred_labels).item())/ylabels.shape[0]
            
        # print('without AE ' + str(100*accuracy_noiAE))
        # print('with AE ' + str(100*accuracy_iAE))
        # print('with AE + mlp ' + str(100*accuracy_iAE2))
        
        
        ### TEST IT JUST ON THE LATENT SPACE OF THE AE
        model.eval()        
        test_data = torch.from_numpy(condn_data_batch).to(device).float()
        z,idx=model(test_data)
        decodes = idx
        # compute accuracy
        with torch.no_grad():
            ylabels = convert_to_ClassNumbers(torch.from_numpy(Ybatch).to(device).float())        
            ypred_labels = convert_to_ClassNumbers(decodes)     
            accuracy_iAE3 = (torch.sum(ylabels == ypred_labels).item())/ylabels.shape[0]
        
        
        print('without AE ' + str(100*accuracy_noiAE))
        print('with AE ' + str(100*accuracy_iAE))
        print('with AE + mlp ' + str(100*accuracy_iAE2))
        print('with AE latent direct' + str(100*accuracy_iAE3))
            
      
        
        decoding_results[i-1,:] =  [accuracy_noiAE,accuracy_iAE,accuracy_iAE2,accuracy_iAE3]
    
    res_overall = np.concatenate((res_overall,decoding_results),axis=0)

 

plt.figure()
plt.boxplot(decoding_results)
plt.figure()
plt.boxplot(decoding_results[:,[0,2]])
plt.ylabel('Sample decoding accuracy')
plt.title('Offline Analyses')
plt.xticks(ticks=[1,2],labels=('Without AE','With AE'))
print(np.mean(decoding_results,axis=0))
print(stats.ttest_rel(decoding_results[:,0],decoding_results[:,2]))



