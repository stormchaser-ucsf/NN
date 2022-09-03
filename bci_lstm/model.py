# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 19:22:32 2022

@author: nikic
"""


import numpy as np
import numpy.random as rnd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import math
from scipy import io as sio
import mat73
import matplotlib.pyplot as plt

# load the data from matlab
file_name = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\decimated_lstm_data_below25Hz.mat'
data_dict = mat73.loadmat(file_name)
condn_data_new = data_dict.get('condn_data_new')
Y = data_dict.get('Y')


# artifact correction
for i in np.arange(condn_data_new.shape[2]):
    
 
    
    # first 128 features
    xx = condn_data_new[:,0:128,i]
    I = np.abs(xx)>15
    I = np.sum(I,0)
    aa = list(np.where(I>0))
    tmp= np.squeeze(xx[:,aa])
    shape_tmp = list(tmp.shape)
    
    if np.size(shape_tmp)==1:
        shape_tmp.append(1)    
        
    tmp_rand = rnd.randn(shape_tmp[0],shape_tmp[1])    
    xx[:,aa[0]] = 1e-5*tmp_rand    
    condn_data_new[:,0:128,i] = xx    
    
    # features 129 to 256
    xx = condn_data_new[:,128:,i]
    I = np.abs(xx)>15
    I = np.sum(I,0)
    aa = list(np.where(I>0))
    tmp= np.squeeze(xx[:,aa])
    shape_tmp = list(tmp.shape)
    
    if np.size(shape_tmp)==1:
        shape_tmp.append(1)    
        
    tmp_rand = rnd.randn(shape_tmp[0],shape_tmp[1])    
    xx[:,aa[0]] = 1e-5*tmp_rand    
    condn_data_new[:,128:,i] = xx    

    
# normalize , min max scaling 
for i in np.arange(condn_data_new.shape[2]):
    tmp = np.squeeze(condn_data_new[:,:,i])
    tmp1 = tmp[:,:128]
    tmp1  = (tmp1-tmp1.min())/(tmp1.max()-tmp1.min())
    
    tmp2 = tmp[:,128:]
    tmp2  = (tmp2-tmp2.min())/(tmp2.max()-tmp2.min())
    
    tmp = np.concatenate((tmp1,tmp2),axis=1)
    condn_data_new[:,:,i] = tmp


# split into testing and training samples
len = range(condn_data_new.shape[2])
len_cutoff = round(0.85*len[-1])
idx = np.random.choice(len,len_cutoff)
I = np.zeros((condn_data_new.shape[2],1))
I[idx]=1


# data augmentation on the training samples 




# define a bidirection lstm model for classifier 
class LSTM1(nn.Module):
    def __init__(self,num_classes,input_size,hidden_size,num_layers,projection_size,fc_nodes,bidirectional_flag,
                 dropout_val,dim_red):
        super(LSTM1,self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state        
        self.projection_size = projection_size #projection length
        self.fc_nodes = fc_nodes #number of nodes in the forward MLP
        self.bidirectional_flag=bidirectional_flag # flag if the LSTM is bidirectional 
        if self.bidirectional :
            self.layer_ratio = 2         
        else:
            self.layer_ratio = 1
        if self.projection_size>0:
            self.hidden_size_output = projection_size
        else:
            self.hidden_size_output = hidden_size        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, proj_size= projection_size,bidirectional = bidirectional_flag,
                          batch_first=True,dropout=dropout_val) #lstm
        self.linear1 =  nn.Linear(self.hidden_size_output, fc_nodes) #fully connected 1
        self.linear2 = nn.Linear(fc_nodes, num_classes) #fully connected last layer
        self.gelu = nn.GELU()
        self.dim_red1 = nn.Linear(128,dim_red)

    def forward(self,x):              
        
        # dimensionality reduction on hg and LMp separately if needed and concatenate 
        # data comes in as batch, sequence length, channels 
        
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x) 
        # extract last hidden state and reshape
        hn=hn[layer_ratio:,:,:]
        hn=hn.view(-1,hn.shape[1])
        hn=torch.t(hn)
        # pass it thru mlp layers        
        out = self.gelu(hn)
        out = self.linear1(out) #first Dense
        out = self.gelu(out) #relu
        out = self.linear2(out) #Final Output
        return out



# the key variables of interest for the model
num_classes=7
input_size = 256
hidden_size=150
num_layers=2
projection_size=75
fc_nodes=25
bidirectional_flag=True
dropout_val=0.3
dim_red= False

# init the model
model = LSTM1(num_classes, input_size, hidden_size, 
              num_layers, projection_size, fc_nodes, bidirectional_flag,
              dropout_val, dim_red)


# training parameters
loss_function = nn.CrossEntropyLoss()
learning_rate = 1e-4
batch_size = 128
gradient_clipping = 10
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate)


# training loop
num_epochs=100
batch_size=128
num_batches = math.ceil(XTrain.shape[0]/batch_size)

for epoch in range(num_epochs):
    # split the data into batches
    idx = np.random.permutation(data.shape[0])
    data1 = data[idx,:]
    
    for i in range(num_batches):
        k = i*batch_size
        k1 = k+batch_size
        
        if k1>data1.shape[0]:
            k1=data1.shape[0]
            
        x = data1[k:k1,:]
        x=torch.from_numpy(x)
        x=x.to(device) # push it to GPU
        x=x.to(torch.float32) # convert to single
        opt.zero_grad() # flush gradients
        xhat = vae(x)
        loss_kl = vae.encoder.kl()
        #print(loss_kl)
        #loss = ((x - xhat)**2).sum() + loss_kl
        loss=criterion(x,xhat) + beta_params*vae.encoder.kl()                  
        loss.backward()
        opt.step()
    print(epoch+1)
    



# load the data from matlab




# training loop 
num_epochs=100
batch_size=64
num_batches = math.ceil(data.shape[0]/batch_size)


# testing it out with some sample data 
num_classes =3
num_layers=2
input_size=4
hidden_size=10
projection_size=5
fc_nodes = 6
bidir= True

if bidir==True:
    layer_ratio=2
else:
    layer_ratio=1
    
if projection_size>0:
    hidden_output=projection_size
else:
    hidden_output=hidden_size
    
rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                  num_layers=num_layers, proj_size= projection_size,bidirectional = bidir,
                  batch_first=True) #lstm

input=torch.randn(64,23,input_size) # batch_size, seq length, 

out,(hn,cn) = rnn(input)
hn=hn[layer_ratio:,:,:]
hn=hn.view(-1,hn.shape[1])
hn=torch.t(hn)









