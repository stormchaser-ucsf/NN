# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 06:22:00 2024

@author: nikic
"""

#%% PRELIMS
import os
os.chdir('C:/Users/nikic/Documents/GitHub/NN/bci_lstm/')
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
from BCI_utils_models import *
import sklearn as skl
from sklearn.metrics import silhouette_score as sil
from sklearn.metrics import silhouette_samples as sil_samples
from tempfile import TemporaryFile
from scipy.ndimage import gaussian_filter1d
import scipy as scipy
import scipy.stats as stats
# setting up GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


#%% GET THE TRAINING DATA


training_data_path = 'F:/DATA/ecog data/ECoG BCI/GangulyServer/Multistate clicker/New_Grid_LSTM_Training_Robot/Training_Validation_Data'
testing_data_path = 'F:/DATA/ecog data/ECoG BCI/GangulyServer/Multistate clicker/New_Grid_LSTM_Training_Robot/Testing_Data'


train_data_filename = training_data_path + '/Data_LSTM_253Grid_Arrow_MoreNoiseDataAug_0920HeldOut.mat'

XTrain,YTrain,XTest,YTest = get_data_lstm(train_data_filename,7)

train_data_filename1 = training_data_path + '/Data_LSTM_253Grid_RobotBatch_MoreNoiseDataAug_0920HeldOut.mat'

XTrain1,YTrain1,XTest1,YTest1 = get_data_lstm(train_data_filename1,7)

XTrain = np.concatenate((XTrain, XTrain1),axis=2)
XTest = np.concatenate((XTest, XTest1),axis=2)
YTrain = np.concatenate((YTrain, YTrain1),axis=0)
YTest = np.concatenate((YTest, YTest1),axis=0)

class_mem = np.empty([0])
for i in np.arange(7):
    class_mem = np.append(class_mem, np.sum(YTrain==(i)))

plt.figure()
plt.bar(np.arange(7),class_mem)


class_mem = np.empty([0])
for i in np.arange(7):
    class_mem = np.append(class_mem, np.sum(YTest==(i)))

plt.figure()
plt.bar(np.arange(7),class_mem)

#%% CONVERT INTEGER LABELS INTO ONE HOT VECTORS

Ytrain_mult = np.zeros((YTrain.shape[0],7))
for i in range(YTrain.shape[0]):
    tmp = round(YTrain[i])-1
    Ytrain_mult[i,tmp]=1

Ytrain = Ytrain_mult

Ytest_mult = np.zeros((YTest.shape[0],7))
for i in range(YTest.shape[0]):
    tmp = round(YTest[i])-1
    Ytest_mult[i,tmp]=1

Ytest = Ytest_mult


#%% BUILD THE LSTM MODEL 


# lstm parameters
input_size=506 
hidden_size=120
num_layers=1
num_classes=7
dropout_val=0.3
fc_nodes=25

#init model
if 'model' in locals():
    del model   
model = rnn_gru(num_classes,input_size,hidden_size,num_layers,
                dropout_val,fc_nodes)
model = model.to(device) #push to GPU


#%% TRAINING THE MODEL

# training params 
num_epochs=150
batch_size=32
learning_rate = 1e-3
batch_val=512
patience=5
gradient_clipping=10

lstm_filename = 'LSTM.pth'

model,acc = training_loop_LSTM(model,num_epochs,batch_size,learning_rate,batch_val,
                      patience,gradient_clipping,lstm_filename,
                      XTrain,Ytrain,XTest,Ytest,
                      input_size,hidden_size,num_classes,
                      num_layers,dropout_val,fc_nodes)

#%% TESTING ON HELD OUT DATA 


test_data_filename = testing_data_path + '/Data_LSTM_253Grid_RobotBatch_MoreNoiseDataAug_0920HeldOut_TestingDataSamples.mat'

X_heldout,Y_heldout = get_data_lstm_heldout(test_data_filename)
Y_heldout = np.round(Y_heldout)

model.eval()
x = torch.from_numpy(X_heldout).to(device).float()
x=torch.permute(x,(2,0,1))
ypred = model(x)                 
ypred_labels = convert_to_ClassNumbers(ypred) + 1     
ypred_labels = ypred_labels.to('cpu').detach().numpy()

heldout_acc = np.sum(Y_heldout == ypred_labels) / len(Y_heldout)
print(heldout_acc)


#%% TESTING STUFF

b=1;
c=2;
t=15;
x=torch.randn(b,c,t)
model = nn.Conv1d(c, 1, 4) #in, out, kernel size
out=model(x)







