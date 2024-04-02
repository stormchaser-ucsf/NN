# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 11:20:32 2024

@author: nikic

Goal is to simulate a) random walk drift, b) systematic drift,
c) drift around a stable attractor state and d) constrained drift within a meta
representational structure. See in which one does using extra days for training 
a classifier help. Start with just three class problem, with known gaussian 
distribution and mean.

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

#%% CONSTRAINED DRIFT

num_samples = 1000
dim = 2
a = rnd.randn(num_samples, dim) + [2,-2]
b = rnd.randn(num_samples, dim) - 2
c = rnd.randn(num_samples, dim) + 0

tmp = rnd.randn(2,2)
a = (a @ tmp)/lin.det(tmp)
# tmp = rnd.randn(2,2)
# b = (b @ tmp)/lin.det(tmp)
# tmp = rnd.randn(2,2)
# c = (c @ tmp)/lin.det(tmp)
i1 = np.zeros([a.shape[0],1])
i2 = np.ones([b.shape[0],1])
i3 = 2* np.ones([c.shape[0],1])
idx = np.concatenate((i1, i2, i3))
plt.figure()
plt.plot(a[:,0],a[:,1],'.')
plt.plot(b[:,0],b[:,1],'.')
plt.plot(c[:,0],c[:,1],'.')
num_classes=3

# get the overall data
condn_data_total = np.concatenate((a, b,c),axis=0)
Ytotal = one_hot_convert(idx)
xlimits = [np.min(condn_data_total[:,0]),np.max(condn_data_total[:,0])]
ylimits = [np.min(condn_data_total[:,1]),np.max(condn_data_total[:,1])]
plt.xlim(xlimits)
plt.ylim(ylimits)
# build a classifier and plot the meta decision boundary 


# get the training data to build the classifier
Ytest = np.zeros((2,2))
while len(np.unique(np.argmax(Ytest,axis=1)))<num_classes:
    Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_total,Ytotal,0.8)                        

# params for training classifier
num_epochs = 150
batch_size = 32
learning_rate = 1e-3
batch_val = 64
patience = 6
gradient_clipping = 10
input_size = dim
num_nodes = 10
num_classes = 3

# build the model
if 'model' in locals():
    del model    
model = mlp_classifier_2Layer(dim,num_nodes,3).to(device)  

filename = 'mlp_drift_sim.pth'
model,acc = training_loop_mlp(model,num_epochs,batch_size,learning_rate,batch_val,
                      patience,gradient_clipping,filename,
                      Xtrain,Ytrain,Xtest,Ytest,
                      input_size,num_nodes,num_classes)
    

# get the classifier contour
x=condn_data_total
h=0.1 # step size in mesh
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


indata = np.array([xx.ravel(),yy.ravel()]).T
indata = torch.from_numpy(indata).to(device).float()

Z = model(indata)
Z = convert_to_ClassNumbers(Z).to('cpu').detach().numpy()
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.cubehelix, alpha=0.5)
plt.show()


## now run k means on a subset of the data and recompute decision boundary 
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5).fit(a)
idx = kmeans.labels_
idx = np.where(idx==1)[0]
a1=a[idx,:]
kmeans = KMeans(n_clusters=5).fit(b)
idx = kmeans.labels_
idx = np.where(idx==2)[0]
b1=b[idx,:]
kmeans = KMeans(n_clusters=5).fit(c)
idx = kmeans.labels_
idx = np.where(idx==3)[0]
c1=c[idx,:]

i1 = np.zeros([a1.shape[0],1])
i2 = np.ones([b1.shape[0],1])
i3 = 2* np.ones([c1.shape[0],1])
idx = np.concatenate((i1, i2, i3))
plt.figure()
plt.plot(a1[:,0],a1[:,1],'.')
plt.plot(b1[:,0],b1[:,1],'.')
plt.plot(c1[:,0],c1[:,1],'.')
num_classes=3

# get the overall data
condn_data_total = np.concatenate((a1, b1,c1),axis=0)
Ytotal = one_hot_convert(idx)

Ytest = np.zeros((2,2))
while len(np.unique(np.argmax(Ytest,axis=1)))<num_classes:
    Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_total,Ytotal,0.8)       

# build the model
if 'model' in locals():
    del model    
model = mlp_classifier_2Layer(dim,num_nodes,3).to(device)  

filename = 'mlp_drift_sim.pth'
model,acc = training_loop_mlp(model,num_epochs,batch_size,learning_rate,batch_val,
                      patience,gradient_clipping,filename,
                      Xtrain,Ytrain,Xtest,Ytest,
                      input_size,num_nodes,num_classes)
    

# get the classifier contour
x=condn_data_total
h=0.1 # step size in mesh
x_min, x_max = xlimits[0] - 1, xlimits[1] + 1
y_min, y_max = ylimits[0] - 1, ylimits[1] + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


indata = np.array([xx.ravel(),yy.ravel()]).T
indata = torch.from_numpy(indata).to(device).float()

Z = model(indata)
Z = convert_to_ClassNumbers(Z).to('cpu').detach().numpy()
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.cubehelix, alpha=0.5)
plt.xlim(xlimits)
plt.ylim(ylimits)
plt.show()
