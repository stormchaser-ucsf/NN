# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 09:50:18 2022

@author: nikic
"""

from random import triangular
import torch
import torch.nn as nn
import torch.nn.functional as F
#from modeling_torch.data_augmentation import *
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import copy
import math
import numpy as np
import numpy.random as rnd
import mat73
import matplotlib.pyplot as plt
import torch.optim as optim


# setting up GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# load the data from matlab
file_name = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\decimated_lstm_data_below25Hz.mat'
data_dict = mat73.loadmat(file_name)
condn_data_new = data_dict.get('condn_data_new')
Y = data_dict.get('Y')


# artifact correction, geting rid of large artifacts
for i in np.arange(condn_data_new.shape[2]):
    
    print(i)
    # first 128 features -> these are high gamma
    xx = condn_data_new[:,0:128,i]
    I = np.abs(xx)>15
    I = np.sum(I,0)
    aa = (np.where(I>0))    
    tmp= np.squeeze(xx[:,aa])
    shape_tmp = list(tmp.shape)
    
    
    if np.size(shape_tmp)==1:
         shape_tmp.append(1)    
        
    tmp_rand = rnd.randn(shape_tmp[0],shape_tmp[1])    
    xx[:,aa[0]] = 1e-5*tmp_rand    
    condn_data_new[:,0:128,i] = xx    
    
    # second 128 features -> these are DC-25Hz 
    xx = condn_data_new[:,128:,i]
    I = np.abs(xx)>15
    I = np.sum(I,0)
    aa = (np.where(I>0))
    tmp= np.squeeze(xx[:,aa])
    shape_tmp = list(tmp.shape)
    
    if np.size(shape_tmp)==1:
        shape_tmp.append(1)    
        
    tmp_rand = rnd.randn(shape_tmp[0],shape_tmp[1])    
    xx[:,aa[0]] = 1e-5*tmp_rand    
    condn_data_new[:,128:,i] = xx    
    

# look at it 
tmp = np.squeeze(condn_data_new[:,66,1])
plt.plot(tmp)
    
# normalize , min max scaling 
for i in np.arange(condn_data_new.shape[2]):
    print(i)
    
    tmp = np.squeeze(condn_data_new[:,:,i])
    tmp1 = tmp[:,:128]
    tmp1  = (tmp1-tmp1.min())/(tmp1.max()-tmp1.min())
    
    tmp2 = tmp[:,128:]
    tmp2  = (tmp2-tmp2.min())/(tmp2.max()-tmp2.min())
    
    tmp = np.concatenate((tmp1,tmp2),axis=1)
    condn_data_new[:,:,i] = tmp
    
# look at it 
tmp = np.squeeze(condn_data_new[:,66,144])
plt.figure()
plt.plot(tmp)

# split into testing and training samples randomly
len = np.arange(condn_data_new.shape[2])
len_cutoff = round(0.85*len[-1])
idx = np.random.permutation(condn_data_new.shape[2])
train_idx, test_idx = idx[:len_cutoff] , idx[len_cutoff:]
Xtrain, Xtest = condn_data_new[:,:,train_idx] , condn_data_new[:,:,test_idx] 
Ytrain, Ytest = Y[train_idx] , Y[test_idx]

# looking at class membership
class_mem = np.empty([0])
for i in np.arange(7):
    class_mem = np.append(class_mem, np.sum(Ytrain==(i+1)))

plt.figure()
plt.bar(np.arange(7)+1,class_mem)


# # split into testing and training samples with balanced sets using dict
# #get the indices for each of the classs first
# class_idx = {'Class':[], 'indices':[], 'size':[]}
# for i in np.arange(7):
#     idx = list(np.where(Y==(i+1)))
#     class_idx['Class'].append(i+1)
#     class_idx['indices'].append(idx)
#     class_idx['size'].append(np.size(idx))

# #shuffle training samples with smallest size for balanced distribution
# min_size = round(5.2e3)#np.min(class_idx['size'])                    
# train_idx = np.empty([0],dtype=int)
# test_idx = np.empty([0],dtype=int)
# for i in np.arange(7):
#     idx = class_idx['indices'][i][0]    
#     I = np.random.permutation(idx.shape[0])
#     idx= idx[I]
#     idx1 = idx[:min_size]
#     idx2 = idx[min_size:]
#     train_idx = np.append(train_idx, idx1)
#     test_idx = np.append(test_idx, idx2)

# Xtrain, Xtest = condn_data_new[:,:,train_idx] , condn_data_new[:,:,test_idx] 
# Ytrain, Ytest = Y[train_idx] , Y[test_idx]

# idx = rnd.permutation(Xtrain.shape[2])
# Xtrain = Xtrain[:,:,idx]
# Ytrain = Ytrain[idx]

class_mem = np.empty([0])
for i in np.arange(7):
    class_mem = np.append(class_mem, np.sum(Ytrain==(i+1)))

plt.figure()
plt.bar(np.arange(7)+1,class_mem)


# # data augmentation on the training samples  -> introduce random noise plus random shift to each  training sample
# Xtrain_aug = np.zeros(Xtrain.shape)
# len = Xtrain.shape[2]
# for i in np.arange(len):
    
#     print(i)
    
#     tmp = np.squeeze(Xtrain[:,:,i])
#     tid = Ytrain[i]
    
#     # first 128, high gamma
#     tmp1 = tmp[:,:128]
#     # add noise
#     var_noise = 0.3
#     std_dev = np.std(np.concatenate(tmp1))
#     add_noise = rnd.randn(tmp1.shape[0],tmp1.shape[1]) * std_dev * var_noise
#     tmp1n = tmp1 + add_noise
#     #plt.figure();plt.plot(tmp1[:,13]);plt.plot(tmp1n[:,13]);plt.show()    
#     # add variable mean offset
#     m=np.mean(tmp1,0)
#     add_mean = m*0.1
#     flip_sign = rnd.rand(add_mean.shape[0])
#     flip_sign[flip_sign>0.5]=1
#     flip_sign[flip_sign<=0.5]=-1
#     add_mean = np.multiply(flip_sign,add_mean) + m
#     tmp1m = tmp1n + add_mean
#     #tmp1m  = (tmp1m-tmp1m.min())/(tmp1m.max()-tmp1m.min())    
#     #plt.figure();plt.plot(tmp1[:,13]);plt.plot(tmp1m[:,13]);plt.show()
    
#     # next 128, LFOs
#     tmp2 = tmp[:,128:]
#     # add noise
#     var_noise = 0.3
#     std_dev = np.std(np.concatenate(tmp2))
#     add_noise = rnd.randn(tmp2.shape[0],tmp2.shape[1]) * std_dev * var_noise
#     tmp2n = tmp2 + add_noise
#     #plt.figure();plt.plot(tmp2[:,13]);plt.plot(tmp2n[:,13]);plt.show()    
#     # add variable mean offset
#     m=np.mean(tmp2,0)
#     add_mean = m*0.15
#     flip_sign = rnd.rand(add_mean.shape[0])
#     flip_sign[flip_sign>0.5]=1
#     flip_sign[flip_sign<=0.5]=-1
#     add_mean = np.multiply(flip_sign,add_mean) + m
#     tmp2m = tmp2n + add_mean
#     #tmp2m  = (tmp2m-tmp2m.min())/(tmp2m.max()-tmp2m.min())    
#     #plt.figure();plt.plot(tmp2[:,13]);plt.plot(tmp2m[:,13]);plt.show()
    
#     tmp=np.concatenate((tmp1m,tmp2m),axis=1)
    
#     Ytrain = np.append(Ytrain,tid) 
#     Xtrain_aug[:,:,i] =tmp
#     #Xtrain[:,:,Xtrain.shape[2]+1] = tmp
#     #Xtrain=np.dstack((Xtrain,tmp))
#     #Xtrain = np.append(Xtrain,np.atleast_3d(tmp),axis=2)
#     #Xtrain = np.concatenate((Xtrain,np.atleast_3d(tmp)),axis=2)


# Xtrain = np.concatenate((Xtrain,Xtrain_aug),axis=2)
# del Xtrain_aug
# del condn_data_new
print('Data augmentation also done')

class_mem = np.empty([0])
for i in np.arange(7):
    class_mem = np.append(class_mem, np.sum(Ytest==(i+1)))

plt.figure()
plt.bar(np.arange(7),class_mem)


# now convert integer to multinomial rerepsentation for classfication
Ytrain_mult = np.zeros((Ytrain.shape[0],7))
for i in range(Ytrain.shape[0]):
    tmp = round(Ytrain[i])-1
    Ytrain_mult[i,tmp]=1

Ytrain = Ytrain_mult
del Ytrain_mult

Ytest_mult = np.zeros((Ytest.shape[0],7))
for i in range(Ytest.shape[0]):
    tmp = round(Ytest[i])-1
    Ytest_mult[i,tmp]=1

Ytest = Ytest_mult
del Ytest_mult

######## TRANSFORMER MODULE ######
class SinusoidPositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return self.pe[:x.size(0)]


class GatedTransformerClassifier(torch.nn.Module):
    def __init__(self, input_sample, input_dim, hidden_dim, hidden_unit_dim, hidden_unit_dim_time,
                 num_heads_time, num_heads, kernel_size, stride, num_layers, num_layers_time,
                 dropout, conv_dropout, output_dropout, n_target):
        super().__init__()
        self.input_dim = input_dim
        self.input_sample = input_sample
        self.hidden_dim = hidden_dim
        self.hidden_sample = (self.input_sample - kernel_size) // stride + 1
        self.idx_sample = torch.arange(0, input_sample, 1, dtype=torch.int).cuda()

        # positional encoding for the time encoder
        # self.pos_encoder = PositionalEncoding(input_sample, output_dropout)
        self.pos_encoder = nn.Embedding(input_sample, input_dim)
        self.dropout_input = nn.Dropout(output_dropout)

        # define the time encoding layers
        self.time_encoder_layer = nn.TransformerEncoderLayer(d_model=input_sample, nhead=num_heads_time,
                                                             dim_feedforward=hidden_unit_dim_time,
                                                             dropout=dropout, activation='gelu')
        self.time_encoder = nn.TransformerEncoder(self.time_encoder_layer, num_layers=num_layers_time)

        # define the channel encoding layers
        self.channel_encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads,
                                                                dim_feedforward=hidden_unit_dim,
                                                                dropout=dropout, activation='gelu')
        self.channel_encoder = nn.TransformerEncoder(self.channel_encoder_layer, num_layers=num_layers)

        # define conv and linear layers for downsampling the time attention output
        self.conv_time = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
                                   kernel_size=kernel_size, stride=stride)
        self.dropout_time0 = nn.Dropout(conv_dropout)
        self.linear_time = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_time1 = nn.Dropout(conv_dropout)
        self.linear_time1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_time2 = nn.Dropout(conv_dropout)

        # define conv and linear layers for downsampling the time attention output
        self.conv_space = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim,
                                    kernel_size=kernel_size, stride=stride)
        self.dropout_space0 = nn.Dropout(conv_dropout)
        self.linear_space = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_space1 = nn.Dropout(conv_dropout)
        self.linear_space1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout_space2 = nn.Dropout(conv_dropout)

        # define linear function for gating
        self.linear_gate_time = nn.Linear(hidden_dim * self.hidden_sample, 1)
        self.linear_gate_channel = nn.Linear(hidden_dim * self.hidden_sample, 1)

        # define the output_layers
        self.dropout_output0 = nn.Dropout(output_dropout)
        self.linear_out = nn.Linear(hidden_dim * self.hidden_sample * 2, hidden_dim * self.hidden_sample)
        self.dropout_output1 = nn.Dropout(output_dropout)
        self.linear_out1 = nn.Linear(hidden_dim * self.hidden_sample, hidden_dim * self.hidden_sample)
        self.dropout_output2 = nn.Dropout(output_dropout)
        self.output_layer = nn.Linear(hidden_dim * self.hidden_sample, n_target)

    def forward(self, input):
        # input is in shape (sample, time, channel)
        # the time encoder is attending along channel axis
        # first transpose data to shape (channel, sample, time)
        x_time = input.contiguous().permute(2, 0, 1)
        x_time = self.time_encoder(x_time)

        # also transpose data to shape (time, sample, channel)
        # the channel encoder is attending along time axis
        x_channel = input.contiguous().permute(1, 0, 2)
        x_pos = torch.unsqueeze(self.pos_encoder(self.idx_sample), dim=1)
        x_channel = self.dropout_input(x_channel + x_pos)
        x_channel = self.channel_encoder(x_channel)

        # now transpose everything back to (sample, channel, time)
        x_time = x_time.permute(1, 0, 2)
        x_channel = x_channel.permute(1, 2, 0)

        # downsample and down project in time
        x_time = self.conv_time(x_time)
        x_time = self.dropout_time0(x_time)
        x_time = x_time.permute(0, 2, 1)
        x_time = F.gelu(self.linear_time(x_time))
        x_time = self.dropout_time1(x_time)

        # skip connection and second layer
        x_time_lin = F.gelu(self.linear_time1(x_time))
        x_time_lin = self.dropout_time2(x_time_lin)
        x_time = x_time + x_time_lin

        # downsample and down project in space
        x_channel = self.conv_space(x_channel)
        x_channel = self.dropout_space0(x_channel)
        x_channel = x_channel.permute(0, 2, 1)
        x_channel = F.gelu(self.linear_space(x_channel))
        x_channel = self.dropout_space1(x_channel)

        # skip connection and second layer
        x_channel_lin = F.gelu(self.linear_space1(x_channel))
        x_channel_lin = self.dropout_space2(x_channel_lin)
        x_channel = x_channel + x_channel_lin

        # now both vectors go from (sample, hidden_channel, hidden_sample) to (sample, -1)
        x_time = torch.flatten(x_time, start_dim=1)
        x_channel = torch.flatten(x_channel, start_dim=1)

        # compute gate
        x_gate_time = torch.sigmoid(torch.squeeze(self.linear_gate_time(x_time)))
        x_gate_channel = torch.sigmoid(torch.squeeze(self.linear_gate_channel(x_channel)))

        # project through gate
        x_time_gated = torch.einsum('ni, n->ni', x_time, x_gate_time)
        x_channel_gated = torch.einsum('ni, n->ni', x_channel, x_gate_channel)
        x_gated = torch.cat([x_time_gated, x_channel_gated], dim=-1)

        # finally to output layers
        x_gated = self.dropout_output0(x_gated)
        x_gated = F.gelu(self.linear_out(x_gated))
        x_gated = self.dropout_output1(x_gated)

        # skip connection and finally to output
        x_gated_lin = F.gelu(self.linear_out1(x_gated))
        x_gated_lin = self.dropout_output2(x_gated_lin)
        x_gated = x_gated + x_gated_lin
        out = self.output_layer(x_gated)

        return out

cfg=type('', (), {})()
cfg.input_sample = 100
cfg.n_targ=7
cfg.fs = 100

# define the hyperparameters for neural network training
cfg.winstart = -0.72
cfg.winend = 0
cfg.jitter_amt = 0

cfg.additive_noise_level = 0.0027354917297051813
cfg.scale_augment_low = 0.98
cfg.scale_augment_high = 1.02
cfg.blackout_len = 0.30682868940865543
cfg.blackout_prob = 0.04787032280216536
cfg.random_channel_noise_sigma = 0.028305685438945617

cfg.decode_layers = 2
cfg.decode_nodes = 256
cfg.dropout = 0.25
cfg.conv_dropout = 0.25
cfg.output_dropout = 0.25
cfg.ks = 3
cfg.stride = 3
cfg.conv_dim = 100

cfg.lr = 1e-3
cfg.batch_size = 256
cfg.steps = 150
cfg.n_fold = 10
cfg.grad_clip = 1
cfg.scheduler_milestone = [30, 60]
cfg.patience = 12
cfg.input_dim=256

#x_train = torch.randn(512, 100, 256).to(device).float()

# calling the model (expect input to be in shape (batch, time, channel))
model = GatedTransformerClassifier(input_sample=cfg.input_sample, input_dim=cfg.input_dim,
                                    hidden_dim=cfg.conv_dim, hidden_unit_dim=int(cfg.decode_nodes),
                                    hidden_unit_dim_time=128, num_heads_time=4,
                                    num_heads=16, kernel_size=int(cfg.ks), stride=int(cfg.stride),
                                    num_layers=int(cfg.decode_layers), num_layers_time=1,
                                    dropout=cfg.dropout, conv_dropout=cfg.conv_dropout,
                                    output_dropout=cfg.output_dropout, n_target=cfg.n_targ)
model = model.to(device)
                                    
#out = model(x_train)



#####################  TRAINING LOOP   ##########################################

# training parameters
num_epochs=100
batch_size = 128
gradient_clipping = 2.0
learning_rate=3e-4
opt = optim.Adam(model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss(reduction='mean')
num_batches = math.ceil(Xtrain.shape[2]/batch_size)
patience=6
len = Xtrain.shape[2]
batch_val=128 

# coverting data to torch tensors and pushing to GPU
Ytest = torch.from_numpy(Ytest).float()
Xtest = torch.from_numpy(Xtest).float()
Xtest= torch.permute(Xtest,(2,0,1)) # N,seq_length,Dimensions


# function to convert one-hot representation back to class numbers
def convert_to_ClassNumbers(indata):
    with torch.no_grad():
        outdata = torch.max(indata,1).indices
    
    return outdata
    
# function to get validation loss 
def validation_loss(model,X_test,Y_test,batch_val,val_type):    
    crit_val = nn.CrossEntropyLoss(reduction='sum')
    loss_val=0    
    accuracy=0
    if batch_val > X_test.shape[0]:
        batch_val = X_test.shape[0]
        
    idx=np.arange(0,X_test.shape[0],batch_val)    
    if idx[-1]<X_test.shape[0]:
        idx=np.append(idx,X_test.shape[0])
    else:
        print('something wrong here')
    
    iters=(idx.shape[0]-1)
    
    for i in np.arange(iters):
        x=X_test[idx[i]:idx[i+1],:,:]
        y=Y_test[idx[i]:idx[i+1],:]
        x=x.to(device)
        y=y.to(device)
        if val_type==1: #validation
            model.eval()
            ypred = model(x)
            model.train()
        else:
            ypred = model(x) #usually just the last training batch
        with torch.no_grad():
            loss_val += (crit_val(ypred,y).item())    
        
        ylabels = convert_to_ClassNumbers(y)        
        ypred_labels = convert_to_ClassNumbers(ypred)     
        accuracy += torch.sum(ylabels == ypred_labels).item()
                    
    loss_val=loss_val/X_test.shape[0]
    accuracy = accuracy/X_test.shape[0]
    torch.cuda.empty_cache()
    return loss_val,accuracy



val_loss,val_acc = validation_loss(model,Xtest,Ytest,batch_val,1)
torch.cuda.empty_cache()
print(val_loss,val_acc)

filename='transformer_model.pth'
goat_loss=99999
goat_acc=0
counter=0
for epoch in range(num_epochs):
  #shuffle the data    
  idx = rnd.permutation(len) 
  
    
  for batch in range(num_batches-1):
      # get the batch 
      k = batch*batch_size
      k1 = k+batch_size
      samples = idx[k:k1]
      Xtrain_batch = Xtrain[:,:,samples]
      Ytrain_batch = Ytrain[samples,:]        
      
      #push to gpu
      Xtrain_batch = torch.from_numpy(Xtrain_batch).float()
      Ytrain_batch = torch.from_numpy(Ytrain_batch).float()    
      Xtrain_batch = torch.permute(Xtrain_batch,(2,0,1)) # N,seq_length,Dimensions      
      Xtrain_batch = Xtrain_batch.to(device)
      Ytrain_batch = Ytrain_batch.to(device)
      
      # pass thru network
      opt.zero_grad() 
      Ypred = model(Xtrain_batch)
      loss = criterion(Ypred,Ytrain_batch)   
      with torch.no_grad():
          Ypred_labels = convert_to_ClassNumbers(Ypred)        
          Ylabels =  convert_to_ClassNumbers(Ytrain_batch)        
      
      accuracy = (torch.sum(Ypred_labels == Ylabels).item())/Ypred_labels.shape[0]
      train_acc = accuracy
      train_loss = loss.item()
      print(loss.item(),accuracy*100)
      loss.backward()
      nn.utils.clip_grad_value_(model.parameters(), clip_value=gradient_clipping)
      opt.step()
      
  val_loss,val_acc=validation_loss(model,Xtest,Ytest,batch_val,1)  
  #train_loss,train_acc=validation_loss(model,Xtrain_batch,Ytrain_batch,
  #                                     round(batch_size/2),0)  
  print(val_loss,val_acc*100,train_loss,train_acc*100,)
  
  if val_loss<goat_loss:
      goat_loss = val_loss
      goat_acc = val_acc*100
      counter = 0
      print('Goat loss, saving model')      
      torch.save(model.state_dict(), filename)
  else:
      counter += 1

  if counter>=patience:
      print('Early stoppping point reached')
      print('Best val loss and accuracy is')
      print(goat_loss,goat_acc)
      break


      

      
