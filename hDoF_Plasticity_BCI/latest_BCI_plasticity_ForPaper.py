# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 20:44:36 2022

@author: nikic
"""


"""
Created on Fri Jul 30 22:47:39 2021

@author: Nikhlesh
"""
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import math
import mat73
import numpy.random as rnd
import numpy.linalg as lin
from sklearn.model_selection  import train_test_split
import os
plt.rcParams['figure.dpi'] = 200
from utils import *
import sklearn as skl
from sklearn.metrics import silhouette_score as sil
from sklearn.metrics import silhouette_samples as sil_samples
from models import plot_latent


# setting up GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# load the data from matlab
file_name = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\condn_data_imagined_day1.mat'
data_dict = mat73.loadmat(file_name)
data_imagined = data_dict.get('condn_data')
#condn_data_imagined = np.array(condn_data_imagined)
condn_data_imagined = np.zeros((0,96))
Y = np.zeros(0)
for i in np.arange(7):
    tmp = np.array(data_imagined[i])
    condn_data_imagined = np.concatenate((condn_data_imagined,tmp),axis=0)
    idx = i*np.ones((tmp.shape[0],1))[:,0]
    Y = np.concatenate((Y,idx),axis=0)

Y_mult = np.zeros((Y.shape[0],7))
for i in range(Y.shape[0]):
    tmp = round(Y[i])
    Y_mult[i,tmp]=1
Y = Y_mult

#condn_data_imagined = scale_zero_one(condn_data_imagined)            
Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_imagined,Y,0.8)
#Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_online,Yonline,0.8)


# create a autoencoder with a classifier layer for separation in latent space
class encoder(nn.Module):
    def __init__(self,input_size,hidden_size,latent_dims,num_classes):
        super(encoder,self).__init__()
        self.hidden_size2 = round(hidden_size/4)
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,self.hidden_size2)
        self.linear3 = nn.Linear(self.hidden_size2,latent_dims)
        self.gelu = nn.ELU()
        self.tanh = nn.Sigmoid()
        self.dropout =  nn.Dropout(p=0.3)
        
    def forward(self,x):
        x=self.linear1(x)
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.linear2(x)        
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.linear3(x)
        #x=self.tanh(x)
        return x

class latent_classifier(nn.Module):
    def __init__(self,latent_dims,num_classes):
        super(latent_classifier,self).__init__()
        self.linear1 = nn.Linear(latent_dims,num_classes)
        self.weights = torch.randn(latent_dims,num_classes).to(device)        
    
    def forward(self,x):
        x=self.linear1(x)        
        #x=torch.matmul(x,self.weights)
        return x
    
class recon_classifier(nn.Module):
    def __init__(self,input_size,num_classes):
        super(recon_classifier,self).__init__()
        self.linear1 = nn.Linear(input_size,num_classes)
        self.weights = torch.randn(input_size,num_classes)
        self.dropout =  nn.Dropout(p=0.3)
    
    def forward(self,x):
        x=self.linear1(x)
        return x

class decoder(nn.Module):
    def __init__(self,input_size,hidden_size,latent_dims,num_classes):
        super(decoder,self).__init__()
        self.hidden_size2 = round(hidden_size/4)
        self.linear1 = nn.Linear(latent_dims,self.hidden_size2)
        self.linear2 = nn.Linear(self.hidden_size2,hidden_size)
        self.linear3 = nn.Linear(hidden_size,input_size)
        self.gelu = nn.ELU()
        self.relu = nn.ReLU()
        self.dropout =  nn.Dropout(p=0.3)
        
    def forward(self,x):
        x=self.linear1(x)
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.linear2(x)
        x=self.gelu(x)
        x=self.dropout(x)
        x=self.linear3(x)
        return x


# combining all into 
class iAutoencoder(nn.Module):
    def __init__(self,input_size,hidden_size,latent_dims,num_classes):
        super(iAutoencoder,self).__init__()
        self.encoder = encoder(input_size,hidden_size,latent_dims,num_classes)
        self.decoder = decoder(input_size,hidden_size,latent_dims,num_classes)
        self.latent_classifier = latent_classifier(latent_dims,num_classes)
        #self.recon_classifier = recon_classifier(input_size,num_classes)
    
    def forward(self,x):
        z=self.encoder(x)
        y=self.latent_classifier(z)
        z=self.decoder(z)
        #y=self.recon_classifier(z)
        return z,y

def convert_to_ClassNumbers(indata):
    with torch.no_grad():
        outdata = torch.max(indata,1).indices
    
    return outdata

# function to validate model 
def validation_loss(model,X_test,Y_test,batch_val,val_type):    
    crit_classif_val = nn.CrossEntropyLoss(reduction='sum') #if mean, it is over all samples
    crit_recon_val = nn.MSELoss(reduction='sum') # if mean, it is over all elements 
    loss_val=0    
    accuracy=0
    recon_error=0
    if batch_val > X_test.shape[0]:
        batch_val = X_test.shape[0]
    
    idx=np.arange(0,X_test.shape[0],batch_val)    
    if idx[-1]<X_test.shape[0]:
        idx=np.append(idx,X_test.shape[0])
    else:
        print('something wrong here')
    
    iters=(idx.shape[0]-1)
    
    for i in np.arange(iters):
        x=X_test[idx[i]:idx[i+1],:]
        y=Y_test[idx[i]:idx[i+1],:]     
        with torch.no_grad():                
            if val_type==1: #validation
                x=torch.from_numpy(x).to(device).float()
                y=torch.from_numpy(y).to(device).float()
                model.eval()
                out,ypred = model(x) 
                loss1 = crit_recon_val(out,x)
                loss2 = crit_classif_val(ypred,y)
                loss_val += loss1.item() + loss2.item()
                model.train()
            else:
                out,ypred = model(x) 
                loss1 = crit_recon_val(out,x)
                loss2 = crit_classif_val(ypred,y)
                loss_val += loss1.item() + loss2.item()
            
            ylabels = convert_to_ClassNumbers(y)        
            ypred_labels = convert_to_ClassNumbers(ypred)     
            accuracy += torch.sum(ylabels == ypred_labels).item()
            recon_error += (torch.sum(torch.square(out-x))).item()   
            
    loss_val=loss_val/X_test.shape[0]
    accuracy = accuracy/X_test.shape[0]
    recon_error = (recon_error/X_test.shape[0])#.cpu().numpy()
    torch.cuda.empty_cache()
    return loss_val,accuracy,recon_error

# model init 
input_size = 96
hidden_size = 32
latent_dims = 3
num_classes = 7
model = iAutoencoder(input_size,hidden_size,latent_dims,num_classes)
model = model.to(device)
# testing it out
input = torch.randn(64,96).to(device)
(recon,decodes) = model(input)


init_loss,init_acc,init_recon = validation_loss(model,Xtest,Ytest,512,1)
print('Initial loss, acc,  recon error')
print(init_loss,init_acc,init_recon)


# TRAINING LOOP
# minimize the recon loss as well as the classification loss
# return the model with lowest validation loss 
num_epochs=200
batch_size=32
num_batches = math.ceil(Xtrain.shape[0]/batch_size)
recon_criterion = nn.MSELoss(reduction='sum')
classif_criterion = nn.CrossEntropyLoss(reduction='sum')
learning_rate = 1e-3
opt = torch.optim.Adam(model.parameters(),lr=learning_rate)
batch_val=512
patience=5
gradient_clipping=5
filename='autoencoder.pth'
print('Starting training')
goat_loss=99999
counter=0
for epoch in range(num_epochs):
  #shuffle the data    
  idx = rnd.permutation(Xtrain.shape[0]) 
  
  if epoch==100:
      for g in opt.param_groups:
          g['lr']=1e-4
    
  for batch in range(num_batches-1):
      # get the batch 
      k = batch*batch_size
      k1 = k+batch_size
      samples = idx[k:k1]
      Xtrain_batch = Xtrain[samples,:]
      Ytrain_batch = Ytrain[samples,:]        
      
      #push to gpu
      Xtrain_batch = torch.from_numpy(Xtrain_batch).float()
      Ytrain_batch = torch.from_numpy(Ytrain_batch).float()          
      Xtrain_batch = Xtrain_batch.to(device)
      Ytrain_batch = Ytrain_batch.to(device)
      
      # forward pass thru network
      opt.zero_grad() 
      recon,decodes = model(Xtrain_batch)
      latent_activity = model.encoder(Xtrain_batch)      
      
      # get loss      
      recon_loss = (recon_criterion(recon,Xtrain_batch))/Xtrain_batch.shape[0]
      classif_loss = (classif_criterion(decodes,Ytrain_batch))/Xtrain_batch.shape[0]      
      loss = recon_loss + classif_loss
      total_loss = loss.item()
      #print(classif_loss.item())
      
      # compute accuracy
      ylabels = convert_to_ClassNumbers(Ytrain_batch)        
      ypred_labels = convert_to_ClassNumbers(decodes)     
      accuracy = (torch.sum(ylabels == ypred_labels).item())/ylabels.shape[0]
      
      # backpropagate thru network 
      loss.backward()
      nn.utils.clip_grad_value_(model.parameters(), clip_value=gradient_clipping)
      opt.step()
  
  # get validation losses
  val_loss,val_acc,val_recon=validation_loss(model,Xtest,Ytest,batch_val,1)    
  #val_loss,val_recon=validation_loss_regression(model,Xtest,Ytest,batch_val,1)    
  
  
  print(f'Epoch [{epoch}/{num_epochs}], Val. Loss {val_loss:.2f}, Train Loss {total_loss:.2f}, Val. Acc {val_acc*100:.2f}, Train Acc {accuracy*100:.2f}')
  #print(f'Epoch [{epoch}/{num_epochs}], Val. Loss {val_loss:.4f}, Train Loss {total_loss:.4f}')
  
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
      print('Best val loss and val acc  are')
      print(goat_loss,goat_acc)
      break




#print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# loading the model back
model_goat = iAutoencoder(input_size,hidden_size,latent_dims,num_classes)
model_goat.load_state_dict(torch.load('autoencoder.pth'))
model_goat=model_goat.to(device)
model_goat.eval()

  
       
        
D = plot_latent(model_goat, Xtest,Ytest,Xtest.shape[0],3)

D = plot_latent(model_goat, condn_data_imagined,Y,2100,3)



# X=rnd.randn(10,12)
# X[[0,3,4,7],:] = X[[0,3,4,7],:] + 25
# X[[1,5,9],:] = X[[1,5,9],:] + 10
# X[[2,6,8],:] = X[[2,6,8],:] - 40
# labels = np.array([1,2,3,1,1,2,3,1,3,2])
# D = sil(X,labels)
# print(1-D)

# D = sil_samples(X,labels)
# print(np.mean(1-D))

D = monte_carlo_mahab(condn_data_imagined,Y,model_goat,0)
D = D[np.triu_indices(D.shape[0])]
D = D[D>0]
imagined_data_D = np.mean(D)
print(imagined_data_D)

D = monte_carlo_mahab_full(condn_data_imagined,Y,0)
plt.imshow(D)
plt.colorbar()
D = D[np.triu_indices(D.shape[0])]
D = D[D>0]
imagined_data_Dfull = np.mean(D)
print(imagined_data_Dfull)


# now load the testing data 
file_name = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\condn_data_online_Day7.mat'
data_online=mat73.loadmat(file_name)
data_online = data_online.get('condn_data')
condn_data_online = np.zeros((0,96))
Yonline = np.zeros(0)
for i in np.arange(7):
    tmp = np.array(data_online[i])
    condn_data_online = np.concatenate((condn_data_online,tmp),axis=0)
    idx = i*np.ones((tmp.shape[0],1))[:,0]
    Yonline = np.concatenate((Yonline,idx),axis=0)

Y_mult = np.zeros((Yonline.shape[0],7))
for i in range(Yonline.shape[0]):
    tmp = round(Yonline[i])
    Y_mult[i,tmp]=1
Yonline = Y_mult

    
D = plot_latent(model_goat, condn_data_online,Yonline,696,3)
D = monte_carlo_mahab(condn_data_online,Yonline,model_goat,0)
D = D[np.triu_indices(D.shape[0])]
D = D[D>0]
online_data_D = np.mean(D)
print(online_data_D)


D = monte_carlo_mahab_full(condn_data_online,Yonline,0)
plt.imshow(D)
plt.colorbar()
D = D[np.triu_indices(D.shape[0])]
D = D[D>0]
online_data_Dfull = np.mean(D)
print(online_data_Dfull)

# # now plot data from online thru the built autoencoder     

# # get the data shape to be batch samples, features
# condn_data = np.empty([0,32])
# l=data_online.size
# Y_online = np.empty([0])
# for i in np.arange(l):
#     tmp=data_online[:,i]
#     tmp=tmp[0]
#     condn_data = np.append(condn_data,tmp,axis=0)
#     idx  = tmp.shape[0]
#     Y_online = np.append(Y_online,i*np.ones([idx,1]))

# data_online = condn_data


# plot_latent(vae, data_online,Y_online,100,3)










