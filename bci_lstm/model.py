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
from sklearn.model_selection import train_test_split


# setting up GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load the data from matlab
file_name = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\decimated_lstm_data_below25Hz.mat'
data_dict = mat73.loadmat(file_name)
condn_data_new = data_dict.get('condn_data_new')
Y = data_dict.get('Y')


# artifact correction
for i in np.arange(condn_data_new.shape[2]):
    
    print(i)
    # first 128 features
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
    
    # features 129 to 256
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
# len = np.arange(condn_data_new.shape[2])
# len_cutoff = round(0.85*len[-1])
# idx = np.random.permutation(condn_data_new.shape[2])
# train_idx, test_idx = idx[:len_cutoff] , idx[len_cutoff:]
# Xtrain, Xtest = condn_data_new[:,:,train_idx] , condn_data_new[:,:,test_idx] 
# Ytrain, Ytest = Y[train_idx] , Y[test_idx]

# # looking at class membership
# class_mem = np.empty([0])
# for i in np.arange(7):
#     class_mem = np.append(class_mem, np.sum(Ytrain==(i+1)))

# plt.Figure()
# plt.bar(np.arange(7)+1,class_mem)


# split into testing and training samples with balanced sets using dict
#get the indices for each of the classs first
class_idx = {'Class':[], 'indices':[], 'size':[]}
for i in np.arange(7):
    idx = list(np.where(Y==(i+1)))
    class_idx['Class'].append(i+1)
    class_idx['indices'].append(idx)
    class_idx['size'].append(np.size(idx))

#shuffle training samples with smallest size for balanced distribution
min_size = round(5.2e3)#np.min(class_idx['size'])                    
train_idx = np.empty([0],dtype=int)
test_idx = np.empty([0],dtype=int)
for i in np.arange(7):
    idx = class_idx['indices'][i][0]    
    I = np.random.permutation(idx.shape[0])
    idx= idx[I]
    idx1 = idx[:min_size]
    idx2 = idx[min_size:]
    train_idx = np.append(train_idx, idx1)
    test_idx = np.append(test_idx, idx2)

Xtrain, Xtest = condn_data_new[:,:,train_idx] , condn_data_new[:,:,test_idx] 
Ytrain, Ytest = Y[train_idx] , Y[test_idx]

idx = rnd.permutation(Xtrain.shape[2])
Xtrain = Xtrain[:,:,idx]
Ytrain = Ytrain[idx]

class_mem = np.empty([0])
for i in np.arange(7):
    class_mem = np.append(class_mem, np.sum(Ytrain==(i+1)))

plt.Figure()
plt.bar(np.arange(7)+1,class_mem)


# convert to numpy arrays
Ytrain = np.array(Ytrain)
Ytest = np.array(Ytest)
Xtrain = np.array(Xtrain)
Xtest = np.array(Xtest)


print('done all steps till data augmntation')

# data augmentation on the training samples  -> introduce random noise plus random shift to each 
Xtrain_aug = np.zeros(Xtrain.shape)
len = Xtrain.shape[2]
for i in np.arange(len):
    
    print(i)
    
    tmp = np.squeeze(Xtrain[:,:,i])
    tid = Ytrain[i]
    
    # first 128, high gamma
    tmp1 = tmp[:,:128]
    # add noise
    var_noise = 0.35
    std_dev = np.std(np.concatenate(tmp1))
    add_noise = rnd.randn(tmp1.shape[0],tmp1.shape[1]) * std_dev * var_noise
    tmp1n = tmp1 + add_noise
    #plt.figure();plt.plot(tmp1[:,13]);plt.plot(tmp1n[:,13]);plt.show()    
    # add variable mean offset
    m=np.mean(tmp1,0)
    add_mean = m*0.2
    flip_sign = rnd.rand(add_mean.shape[0])
    flip_sign[flip_sign>0.5]=1
    flip_sign[flip_sign<=0.5]=-1
    add_mean = np.multiply(flip_sign,add_mean) + m
    tmp1m = tmp1n + add_mean
    tmp1m  = (tmp1m-tmp1m.min())/(tmp1m.max()-tmp1m.min())    
    #plt.figure();plt.plot(tmp1[:,13]);plt.plot(tmp1m[:,13]);plt.show()
    
    # next 128, LFOs
    tmp2 = tmp[:,128:]
    # add noise
    var_noise = 0.4
    std_dev = np.std(np.concatenate(tmp2))
    add_noise = rnd.randn(tmp2.shape[0],tmp2.shape[1]) * std_dev * var_noise
    tmp2n = tmp2 + add_noise
    #plt.figure();plt.plot(tmp2[:,13]);plt.plot(tmp2n[:,13]);plt.show()    
    # add variable mean offset
    m=np.mean(tmp2,0)
    add_mean = m*0.5
    flip_sign = rnd.rand(add_mean.shape[0])
    flip_sign[flip_sign>0.5]=1
    flip_sign[flip_sign<=0.5]=-1
    add_mean = np.multiply(flip_sign,add_mean) + m
    tmp2m = tmp2n + add_mean
    tmp2m  = (tmp2m-tmp2m.min())/(tmp2m.max()-tmp2m.min())    
    #plt.figure();plt.plot(tmp2[:,13]);plt.plot(tmp2m[:,13]);plt.show()
    
    tmp=np.concatenate((tmp1m,tmp2m),axis=1)
    
    Ytrain = np.append(Ytrain,tid) 
    Xtrain_aug[:,:,i] =tmp
    #Xtrain[:,:,Xtrain.shape[2]+1] = tmp
    #Xtrain=np.dstack((Xtrain,tmp))
    #Xtrain = np.append(Xtrain,np.atleast_3d(tmp),axis=2)
    #Xtrain = np.concatenate((Xtrain,np.atleast_3d(tmp)),axis=2)


Xtrain = np.concatenate((Xtrain,Xtrain_aug),axis=2)
del Xtrain_aug
del condn_data_new
print('Data augmentation also done')

class_mem = np.empty([0])
for i in np.arange(7):
    class_mem = np.append(class_mem, np.sum(Ytest==(i+1)))

plt.figure()
plt.bar(np.arange(7),class_mem)

###################### LSTM CODE ############################
# creating stacked bidirectional lstm layer class
class LSTM1(nn.Module):
    def __init__(self,num_classes,input_size,hidden_size,proj_size,num_layers,fc_nodes,
                 dropout_val):
        super(LSTM1,self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size       
        
        self.lstm=nn.LSTM(input_size=input_size,hidden_size=hidden_size,
                          num_layers=num_layers,batch_first=True,dropout=dropout_val,
                          bidirectional=True,proj_size=proj_size)
        self.layer_ratio=num_layers*2-2
        if proj_size>0:
            self.mlp_input = proj_size*2
        else:
            self.mlp_input = hidden_size*2              
        self.linear1 = nn.Linear(self.mlp_input,fc_nodes)
        self.linear2 = nn.Linear(fc_nodes,num_classes)
        self.gelu=nn.GELU()
        self.bn1 = nn.BatchNorm1d(fc_nodes)
        
    
    def forward(self,x):
        # take the last hidden state of the last layer all the way through MLP
        output, (hn, cn) = self.lstm(x) 
        hn = hn[self.layer_ratio:,:,:]
        hn = torch.permute(hn,(1,0,2))
        hn = torch.flatten(hn,start_dim=1,end_dim=2)
        #pass thru mlp
        out = self.linear1(hn)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.linear2(out)
        return out

# lstm parameters
num_classes=7
input_size=256
hidden_size=150
proj_size=75
num_layers=2
dropout_val=0.3
fc_nodes=25
model =  LSTM1(num_classes, input_size, hidden_size,proj_size,
               num_layers,fc_nodes, dropout_val) 
model = model.to(device) #push to GPU


#####################  TRAINING LOOP   ##########################################

# training parameters
num_epochs=100
loss_function = nn.CrossEntropyLoss()
learning_rate = 1e-4
batch_size = 128
gradient_clipping = 10
optimizer = optim.Adam(model.parameters(),lr=learning_rate)
num_batches = math.ceil(Xtrain.shape[2]/batch_size)
patience=0



# training loop:
    # define batch size -> get number of batches per epoch
    # define number of total epochs -> randomize data within each epoch
    # within for loop of each epoch, have subfor loop of num batches
    # send the batch throught the network, compute loss, get gradient and update model
    # at the end of epoch, evaluate the validation loss
    # if validation loss at current epoch lower than previous epoch, save model
    # if validation loss increases at current epoch, dont save and increase patience couter by 1
    # if patience counter hits 6 -> consecutive epochs without any improvements, terminate training
    # learning rate scheduler design





for epoch in range(num_epochs):
    #shuffle the data
    idx = rnd.permutation(Xtrain.shape[2])
    Xtrain = Xtrain[:,:,idx]
    Ytrain = Ytrain[idx]
    
    for batch in range(num_batches):
        # get the batch 
        k = batch*batch_size
        k1 = k+batch_size
        Xtrain_batch = Xtrain[:,:,k:k1]
        Ytrain_batch = Ytrain[k:k1]
        
        #push to gpu
        Xtrain_batch = torch.from_numpy(Xtrain_batch)
        Ytrain_batch = torch.from_numpy(Ytrain_batch)
        Xtrain_batch = Xtrain_batch.to(device)
        Ytrain_batch = Ytrain_batch.to(device)
        
        


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




# testing random permutation
data_test = np.empty(0,dtype=int)
for i in range(1000):
    tmp = np.random.choice(10,2)
    data_test = np.append(data_test,(tmp))





class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out
    
    
num_epochs = 1000 #1000 epochs
learning_rate = 0.001 #0.001 lr

input_size = 5 #number of features
hidden_size = 2 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers

num_classes = 1 #number of output classes 


lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, 128) #our lstm class 


 
class LSTM1(nn.Module):
    def __init__(self,num_classes,input_size,hidden_size,num_layers,projection_size,fc_nodes,
                 bidirectional_flag,dropout_val,dim_red):
        super(LSTM1,self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state        
        self.projection_size = projection_size #projection length
        self.fc_nodes = fc_nodes #number of nodes in the forward MLP
        self.bidirectional_flag=bidirectional_flag # flag if the LSTM is bidirectional 
        if self.bidirectional_flag:
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
        hn=hn[self.layer_ratio:,:,:]
        hn=hn.view(-1,hn.shape[1])
        hn=torch.t(hn)
        # pass it thru mlp layers        
        out = self.gelu(hn)
        out = self.linear1(out) #first Dense
        out = self.gelu(out) #relu
        out = self.linear2(out) #Final Output
        return out
