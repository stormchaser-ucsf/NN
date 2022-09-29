# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 19:22:32 2022

@author: nikic
"""

import numpy as np
import numpy.random as rnd
import torch
import torch.nn as nn
import torch.optim as optim
import math
import mat73
import matplotlib.pyplot as plt


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


# data augmentation on the training samples  -> introduce random noise plus random shift to each  training sample
Xtrain_aug = np.zeros(Xtrain.shape)
len = Xtrain.shape[2]
for i in np.arange(len):
    
    print(i)
    
    tmp = np.squeeze(Xtrain[:,:,i])
    tid = Ytrain[i]
    
    # first 128, high gamma
    tmp1 = tmp[:,:128]
    # add noise
    var_noise = 0.7
    std_dev = np.std(np.concatenate(tmp1))
    add_noise = rnd.randn(tmp1.shape[0],tmp1.shape[1]) * std_dev * var_noise
    tmp1n = tmp1 + add_noise
    #plt.figure();plt.plot(tmp1[:,13]);plt.plot(tmp1n[:,13]);plt.show()    
    # add variable mean offset
    m=np.mean(tmp1,0)
    add_mean = m*0.25
    flip_sign = rnd.rand(add_mean.shape[0])
    flip_sign[flip_sign>0.5]=1
    flip_sign[flip_sign<=0.5]=-1
    add_mean = np.multiply(flip_sign,add_mean) + m
    tmp1m = tmp1n + add_mean
    #tmp1m  = (tmp1m-tmp1m.min())/(tmp1m.max()-tmp1m.min())    
    #plt.figure();plt.plot(tmp1[:,13]);plt.plot(tmp1m[:,13]);plt.show()
    
    # next 128, LFOs
    tmp2 = tmp[:,128:]
    # add noise
    var_noise = 0.7
    std_dev = np.std(np.concatenate(tmp2))
    add_noise = rnd.randn(tmp2.shape[0],tmp2.shape[1]) * std_dev * var_noise
    tmp2n = tmp2 + add_noise
    #plt.figure();plt.plot(tmp2[:,13]);plt.plot(tmp2n[:,13]);plt.show()    
    # add variable mean offset
    m=np.mean(tmp2,0)
    add_mean = m*0.35
    flip_sign = rnd.rand(add_mean.shape[0])
    flip_sign[flip_sign>0.5]=1
    flip_sign[flip_sign<=0.5]=-1
    add_mean = np.multiply(flip_sign,add_mean) + m
    tmp2m = tmp2n + add_mean
    #tmp2m  = (tmp2m-tmp2m.min())/(tmp2m.max()-tmp2m.min())    
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
    

# ###################### LSTM CODE ############################
# # creating stacked bidirectional lstm layer class
# class LSTM1(nn.Module):
#     def __init__(self,num_classes,input_size,hidden_size,proj_size,num_layers,fc_nodes,
#                  dropout_val):
#         super(LSTM1,self).__init__()
#         self.num_classes = num_classes
#         self.num_layers = num_layers
#         self.input_size = input_size       
        
#         self.lstm=nn.LSTM(input_size=input_size,hidden_size=hidden_size,
#                           num_layers=num_layers,batch_first=True,dropout=dropout_val,
#                           bidirectional=True,proj_size=proj_size)
#         self.layer_ratio=num_layers*2-2
#         if proj_size>0:
#             self.mlp_input = proj_size*2
#         else:
#             self.mlp_input = hidden_size*2              
#         self.linear1 = nn.Linear(self.mlp_input,fc_nodes)
#         self.linear2 = nn.Linear(fc_nodes,num_classes)
#         self.gelu=nn.GELU()
#         self.bn1 = nn.BatchNorm1d(fc_nodes)
#         self.softmax = nn.Softmax(dim=1)
        
    
#     def forward(self,x):
#         # take the last hidden state of the last layer all the way through MLP
#         output, (hn, cn) = self.lstm(x) 
#         hn = hn[self.layer_ratio:,:,:]
#         hn = torch.permute(hn,(1,0,2))
#         hn = torch.flatten(hn,start_dim=1,end_dim=2)
#         #pass thru mlp
#         out = self.linear1(hn)
#         out = self.bn1(out)
#         out = self.gelu(out)
#         out = self.linear2(out)
#         # get the softmax values of out
#         out_prob = self.softmax(out)
#         return out

# # lstm parameters
# num_classes=7
# input_size=256
# hidden_size=150
# proj_size=0
# num_layers=2
# dropout_val=0.35
# fc_nodes=25
# model =  LSTM1(num_classes, input_size, hidden_size,proj_size,
#                num_layers,fc_nodes, dropout_val) 
# model = model.to(device) #push to GPU


# creating a bidir LSTM -> GRU -> MLP, with GRU having half the number of hidden nodes
class rnn_gru(nn.Module):
    def __init__(self,num_classes,input_size,hidden_size,num_layers,dropout_val,fc_nodes):
        super(rnn_gru,self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size       
        
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=input_size, 
                               kernel_size=2,stride=2)
        
        self.rnn1=nn.LSTM(input_size=input_size,hidden_size=hidden_size,
                          num_layers=num_layers,batch_first=True,bidirectional=True)
        self.rnn2=nn.GRU(input_size=round(hidden_size*2),hidden_size=round(hidden_size/2),
                          num_layers=num_layers,batch_first=True, bidirectional=False)
        
        self.mlp_input = round(hidden_size/2)
        self.linear0 = nn.Linear(self.mlp_input,fc_nodes)
        self.linear1 = nn.Linear(fc_nodes,num_classes)
        self.dropout = nn.Dropout(dropout_val)
        self.gelu = nn.GELU()
    
    def forward(self,x):        
        #x=torch.permute(x,(0,2,1))
        #x=self.conv1(x)
        #x=self.dropout(x)
        #x=torch.permute(x,(0,2,1))
        output1, (hn1,cn1) = self.rnn1(x) 
        output1=self.dropout(output1)
        output2, (hn2) = self.rnn2(output1)
        hn2 = torch.squeeze(hn2)    
        hn2 = self.dropout(hn2)            
        out = self.linear0(hn2)        
        out = self.gelu(out)
        out = self.linear1(out)        
        return out

# lstm parameters
input_size=256 
hidden_size=150
num_layers=1
sequence_length = 100
num_classes=7
dropout_val=0.3
fc_nodes=25

#init model
model = rnn_gru(num_classes,input_size,hidden_size,num_layers,dropout_val,fc_nodes)
model = model.to(device) #push to GPU



#####################  TRAINING LOOP   ##########################################

# training parameters
num_epochs=100
batch_size = 256
gradient_clipping = 10.0
learning_rate=1e-3
opt = optim.Adam(model.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss(reduction='mean')
num_batches = math.ceil(Xtrain.shape[2]/batch_size)
patience=6
len = Xtrain.shape[2]
batch_val=512 
indices = np.arange(0,len,batch_size)

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

filename='lstm_model.pth'
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


      




        


#training loop:
    # define batch size -> get number of batches per epoch
    # define number of total epochs -> randomize data within each epoch
    # within for loop of each epoch, have subfor loop of num batches
    # send the batch throught the network, compute loss, get gradient and update model
    # at the end of epoch, evaluate the validation loss
    # if validation loss at current epoch lower than previous epoch, save model
    # if validation loss increases at current epoch, dont save and increase patience couter by 1
    # if patience counter hits 6 -> consecutive epochs without any improvements, terminate training
    # learning rate scheduler design
        
 
# class LSTM1(nn.Module):
#     def __init__(self,num_classes,input_size,hidden_size,num_layers,projection_size,fc_nodes,
#                  bidirectional_flag,dropout_val,dim_red):
#         super(LSTM1,self).__init__()
#         self.num_classes = num_classes #number of classes
#         self.num_layers = num_layers #number of layers
#         self.input_size = input_size #input size
#         self.hidden_size = hidden_size #hidden state        
#         self.projection_size = projection_size #projection length
#         self.fc_nodes = fc_nodes #number of nodes in the forward MLP
#         self.bidirectional_flag=bidirectional_flag # flag if the LSTM is bidirectional 
#         if self.bidirectional_flag:
#             self.layer_ratio = 2         
#         else:
#             self.layer_ratio = 1
#         if self.projection_size>0:
#             self.hidden_size_output = projection_size
#         else:
#             self.hidden_size_output = hidden_size        
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
#                           num_layers=num_layers, proj_size= projection_size,bidirectional = bidirectional_flag,
#                           batch_first=True,dropout=dropout_val) #lstm
#         self.linear1 =  nn.Linear(self.hidden_size_output, fc_nodes) #fully connected 1
#         self.linear2 = nn.Linear(fc_nodes, num_classes) #fully connected last layer
#         self.gelu = nn.GELU()
#         self.dim_red1 = nn.Linear(128,dim_red)

#     def forward(self,x):              
        
#         # dimensionality reduction on hg and LMp separately if needed and concatenate 
#         # data comes in as batch, sequence length, channels 
        
#         # Propagate input through LSTM
#         output, (hn, cn) = self.lstm(x) 
#         # extract last hidden state and reshape
#         hn=hn[self.layer_ratio:,:,:]
#         hn=hn.view(-1,hn.shape[1])
#         hn=torch.t(hn)
#         # pass it thru mlp layers        
#         out = self.gelu(hn)
#         out = self.linear1(out) #first Dense
#         out = self.gelu(out) #relu
#         out = self.linear2(out) #Final Output
#         return out
