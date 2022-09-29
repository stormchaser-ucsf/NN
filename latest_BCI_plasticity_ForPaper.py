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
from sklearn.model_selection  import train_test_split
plt.rcParams['figure.dpi'] = 200

# setting up GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# load the data from matlab
file_name = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\condn_data_imagined_Day1.mat'
data_dict = mat73.loadmat(file_name)
condn_data_imagined = data_dict.get('condn_data')
condn_data_imagined = np.array(condn_data_imagined)


# get the class labels from the data, and convert to one-hot encoding
len = np.shape(condn_data_imagined)[1]
Y=np.zeros([0])
for i in range(7):
    #len = np.shape(condn_data_imagined)[i]
    #len = condn_data_imagined[i].shape
    Y = np.append(Y,i*np.ones(len))
    
Y_mult = np.zeros((Y.shape[0],7))
for i in range(Y.shape[0]):
    tmp = round(Y[i])
    Y_mult[i,tmp]=1

Y= Y_mult

# convert to 2D matrix from 3D
tmp_data = np.zeros((condn_data_imagined.shape[0]*condn_data_imagined.shape[1],
                    condn_data_imagined.shape[2]))
tmp_data = np.empty([0,96])

for i in np.arange(condn_data_imagined.shape[0]):
    tmp = condn_data_imagined[i,:,:]
    tmp_data = np.concatenate((tmp_data,tmp))

condn_data_imagined = tmp_data


# split into training and validation class 
def training_test_split(condn_data,Y,prop):
    len = np.arange(Y.shape[0])
    len_cutoff = round(prop*len[-1])
    idx = np.random.permutation(Y.shape[0])
    train_idx, test_idx = idx[:len_cutoff] , idx[len_cutoff:]
    Xtrain, Xtest = condn_data_imagined[train_idx,:] , condn_data_imagined[test_idx,:] 
    Ytrain, Ytest = Y[train_idx,:] , Y[test_idx,:]
    return Xtrain,Xtest,Ytrain,Ytest
             
Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_imagined,Y,0.85)

    

# why isnt this working????
# I = np.ones((len,1))
# I[np.round(idx)] =  0    
# I = (np.where(I==1)[0])

# create a autoencoder with a classifier layer for separation in latent space
class encoder(nn.Module):
    def __init__(self,input_size,hidden_size,latent_dims,num_classes):
        super(encoder,self).__init__()
        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size,latent_dims)
        self.gelu = nn.ELU()
        self.tanh = nn.Sigmoid()
        
    def forward(self,x):
        x=self.linear1(x)
        x=self.gelu(x)
        x=self.linear2(x)
        #x=self.gelu(x)
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
    
    def forward(self,x):
        x=self.linear1(x)
        return x

class decoder(nn.Module):
    def __init__(self,input_size,hidden_size,latent_dims,num_classes):
        super(decoder,self).__init__()
        self.linear1 = nn.Linear(latent_dims,hidden_size)
        self.linear2 = nn.Linear(hidden_size,input_size)
        self.gelu = nn.ELU()
        
    def forward(self,x):
        x=self.gelu(x)
        x=self.linear1(x)
        x=self.gelu(x)
        x=self.linear2(x)
        return x


# combining all into 
class iAutoencoder(nn.Module):
    def __init__(self,input_size,hidden_size,latent_dims,num_classes):
        super(iAutoencoder,self).__init__()
        self.encoder = encoder(input_size,hidden_size,latent_dims,num_classes)
        self.decoder = decoder(input_size,hidden_size,latent_dims,num_classes)
        self.latent_classifier = latent_classifier(latent_dims,num_classes)
        self.recon_classifier = recon_classifier(input_size,num_classes)
    
    def forward(self,x):
        z=self.encoder(x)
        y=self.latent_classifier(z)
        z=self.decoder(z)
        #y=self.recon_classifier(z)
        return z,y

# model init 
input_size = 96
hidden_size = 32
latent_dims = 3
num_classes = 7
model = iAutoencoder(input_size,hidden_size,latent_dims,num_classes)
model = model.to(device)
# testing it out
input = torch.randn(64,96).to(device)
(recon,latent) = model(input)

# function to convert one-hot representation back to class numbers
def convert_to_ClassNumbers(indata):
    with torch.no_grad():
        outdata = torch.max(indata,1).indices
    
    return outdata

# validation function 
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
                out,ypred = model(x) #usually just the last training batch
                loss1 = crit_recon_val(out,x)
                loss2 = crit_classif_val(ypred,y)
                loss_val += loss1.item() + loss2.item()
                model.train()
            else:
                out,ypred = model(x) #usually just the last training batch
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

init_loss,init_acc,init_recon = validation_loss(model,Xtest,Ytest,32,1)
print('Initial loss, accuracy, recon error')
print(init_loss,init_acc,init_recon)

# TRAINING LOOP
# minimize the recon loss as well as the classification loss
# return the model with lowest validation loss 

num_epochs=300
batch_size=32
num_batches = math.ceil(Xtrain.shape[0]/batch_size)
recon_criterion = nn.MSELoss(reduction='sum')
classif_criterion = nn.CrossEntropyLoss(reduction='sum')
learning_rate = 1e-3
opt = torch.optim.Adam(model.parameters(),lr=learning_rate)
batch_val=32
patience=8

print('Starting training')
goat_loss=99999
counter=0
for epoch in range(num_epochs):
  #shuffle the data    
  idx = rnd.permutation(Xtrain.shape[0]) 
  
  if epoch==150:
      for g in opt.param_groups:
          g['lr']=1e-4
      
    
  for batch in range(num_batches):
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
      
      # pass thru network
      opt.zero_grad() 
      recon,decodes = model(Xtrain_batch)
      recon_loss = (recon_criterion(recon,Xtrain_batch))/Xtrain_batch.shape[0]
      classif_loss = (classif_criterion(decodes,Ytrain_batch))/Xtrain_batch.shape[0]      
      loss = recon_loss + classif_loss
      #print(classif_loss.item())
      loss.backward()
      #nn.utils.clip_grad_value_(model.parameters(), clip_value=gradient_clipping)
      opt.step()
      
  val_loss,val_acc,val_recon=validation_loss(model,Xtest,Ytest,batch_val,1)  
  train_loss,train_acc,train_recon=validation_loss(model,Xtrain_batch,Ytrain_batch,
                                       round(batch_size/2),0)  
  print(epoch,val_loss,val_acc*100,train_loss,train_acc*100)
  
  if val_loss<goat_loss:
      goat_loss = val_loss
      counter = 0
  else:
      counter += 1

  if counter>=patience:
      print('Early stoppping point reached')
      print('Best val loss is')
      print(goat_loss)
      break

#print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# plotting in latent space
def plot_latent(model, data, Y, num_samples,dim):
    # randomly sample the number of samples 
    idx = rnd.choice(data.shape[0],num_samples)
    data=torch.from_numpy(data).to(device).float()
    Y=torch.from_numpy(Y).to(device).float()
    Y=convert_to_ClassNumbers(Y)
    y=Y[idx]    
    z = data[idx,:]
    z = model.encoder(z)
    z = z.to('cpu').detach().numpy()
    y = y.to('cpu').detach().numpy()    
    plt.figure()
    if dim==3:
        ax=plt.axes(projection="3d")
        p=ax.scatter3D(z[:, 0], z[:, 1],z[:,2], c=y, cmap='tab10')
        p=ax.scatter3D(z[:, 0], z[:, 1],z[:,2], c=y, cmap='tab10')
        plt.colorbar(p)
    if dim==2:
        ax=plt.axes
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        plt.colorbar()
        
plot_latent(model, Xtrain,Ytrain,500,3)
    
   
    

# # now plot data from online thru the built autoencoder     
# file_name = 'F:\DATA\ecog data\ECoG BCI\GangulyServer\Multistate clicker\condn_data_online'
# data_online=sio.loadmat(file_name, mdict=None, appendmat=True)
# data_online=data_online.get('condn_data_online')

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










