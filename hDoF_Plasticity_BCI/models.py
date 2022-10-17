# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 16:57:09 2022

@author: nikic
"""


import torch as torch
import torch.nn as nn
import torch.utils
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.random as rnd
plt.rcParams['figure.dpi'] = 200
from utils import *
from sklearn.metrics import silhouette_score as sil
from sklearn.metrics import silhouette_samples as sil_samples
# setting up GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

def get_model_iAE(input_size,hidden_size,latent_dims,num_classes):
    model = iAutoencoder(input_size,hidden_size,latent_dims,num_classes)
    model = model.to(device)
    return model




class VariationalEncoder(nn.Module):
    def __init__(self,input_size,hidden_dim,latent_dim,dropout):
        super(VariationalEncoder, self).__init__()
        self.hidden_dim2 = round(hidden_dim/4)
        self.linear1 = nn.Linear(input_size,hidden_dim)
        self.linear2 = nn.Linear(hidden_dim,  self.hidden_dim2 )
        self.mu = nn.Linear(self.hidden_dim2 ,latent_dim) 
        self.log_sig = nn.Linear(self.hidden_dim2 ,latent_dim) 
        self.relu = nn.GELU()
        self.dropout_val= dropout
        self.dropout = nn.Dropout(dropout)        
        # gaussian samplimg
        self.Ndist = torch.distributions.Normal(0, 1)     
        self.Ndist.loc = self.Ndist.loc.cuda()
        self.Ndist.scale = self.Ndist.scale.cuda()
        self.kl= 0 #KL loss
    
    def forward(self,x):        
        x=self.linear1(x)
        x=self.relu(x)
        if self.dropout_val >0:
            x=self.dropout(x)
        x=self.linear2(x)
        x=self.relu(x)
        if self.dropout_val >0:
            x=self.dropout(x)
        mu = self.mu(x)
        log_sigma = self.log_sig(x)
        sigma = torch.exp(log_sigma)
        z = mu + sigma*self.Ndist.sample(mu.shape) # reparametrizing trick
        qdist = torch.distributions.Normal(mu,sigma)
        log_qz = qdist.log_prob(z)
        log_pz = self.Ndist.log_prob(z)
        self.kl = (log_qz - log_pz).sum(-1) # summing over all the dimensions
        return z
    
class VariationalDecoder(nn.Module):
    def __init__(self,input_size,hidden_dim,latent_dim,dropout):
        super(VariationalDecoder, self).__init__()
        self.hidden_dim2 = round(hidden_dim/4)
        self.linear1 = nn.Linear(latent_dim,self.hidden_dim2)
        self.linear2 = nn.Linear(self.hidden_dim2,hidden_dim)
        self.mu_p = nn.Linear(hidden_dim, input_size)
        self.logsig_p = nn.Linear(hidden_dim, input_size)        
        self.relu = nn.GELU()
        self.dropout_val= dropout
        self.dropout = nn.Dropout(dropout)        
        # gaussian sampling for reparmaterization trick on GPU
        self.Ndist = torch.distributions.Normal(0, 1)
        self.Ndist.loc = self.Ndist.loc.to(device)
        self.Ndist.scale = self.Ndist.scale.to(device)
        self.logprob_x = 0 #log prob of data
    
    def forward(self,x):        
        x=self.linear1(x)
        x=self.relu(x)
        if self.dropout_val >0:
            x=self.dropout(x)
        x=self.linear2(x)
        x=self.relu(x)
        if self.dropout_val >0:
            x=self.dropout(x)
        mu_p = self.mu_p(x)
        logsig_p = self.logsig_p(x)
        sigma_p = torch.exp(logsig_p)
        xhat = mu_p + sigma_p*self.Ndist.sample(mu_p.shape) # reconstruction        
        return xhat,mu_p,sigma_p
    
class latentClassifier(nn.Module):
    def __init__(self,latent_dim,num_classes):
        super(latentClassifier,self).__init__()
        self.linear = nn.Linear(latent_dim,num_classes)
    
    def forward(self,x):
        x = self.linear(x)
        return x

# putting it all together 
# class vae(nn.Module):
#     def __init__(self,input_size,hidden_dim,latent_dim,num_classes,dropout):
#         super(vae,self).__init__()
#         self.encoder = VariationalEncoder(input_size,hidden_dim,latent_dim,dropout)
#         self.decoder = VariationalDecoder(input_size,hidden_dim,latent_dim,dropout)
#         #self.classifier = latentClassifier(latent_dim,num_classes) 
#         self.logprob_x=0
#         self.kl_loss=0
    
#     def forward(self,x):
#         z=self.encoder(x)
#         self.kl_loss = self.encoder.kl
#         #y=self.classifier(z)
#         xhat,mu_p,sig_p=self.decoder(z)
#         pdist = torch.distributions.Normal(mu_p, sig_p)
#         pdist.loc = pdist.loc.to(device)
#         pdist.scale = pdist.scale.to(device)
#         self.logprob_x = pdist.log_prob(x).sum(-1) # summing over all dimensions
#         vae_loss = self.kl_loss - self.logprob_x                
#         return xhat,vae_loss





# function to convert one-hot representation back to class numbers
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



# function to validate model 
def validation_loss_cluster(model,X_test,Y_test,batch_val,val_type):       
    crit_recon_val = nn.MSELoss(reduction='sum') # if mean, it is over all elements 
    loss_val=0    
    cluster_fitness=0
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
                out = model(x) 
                out_latent = model.encoder(x)
                loss1 = crit_recon_val(out,x)
                loss2 = 1 - sil_samples(out_latent.detach().cpu().numpy(),
                        convert_to_ClassNumbers(y).detach().cpu().numpy())
                loss2 = torch.tensor(loss2).to(device).float()                
                loss_val += loss1.item() + loss2.sum().item()
                model.train()
            else: #do nothing
                loss1=0
                loss2 = 0                
                loss_val += loss1.item() + loss2.sum().item()         
            
            cluster_fitness += loss2.sum().item()            
            recon_error += (torch.sum(torch.square(out-x))).item()   
            
    loss_val=loss_val/X_test.shape[0]
    cluster_fitness = cluster_fitness/X_test.shape[0]
    recon_error = (recon_error/X_test.shape[0])#.cpu().numpy()
    torch.cuda.empty_cache()
    return loss_val,cluster_fitness,recon_error



# function to validate model 
def validation_loss_regression(model,X_test,Y_test,batch_val,val_type):        
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
                out = model(x) 
                loss1 = crit_recon_val(out,x)                
                loss_val += loss1.item() 
                model.train()
            else:
                out = model(x) 
                loss1 = crit_recon_val(out,x)                
                loss_val += loss1.item()             
            
            recon_error += (torch.sum(torch.square(out-x))).item()   
            
    loss_val=loss_val/X_test.shape[0]    
    recon_error = (recon_error/X_test.shape[0])#.cpu().numpy()
    torch.cuda.empty_cache()
    return loss_val,recon_error

# plotting in latent space
def plot_latent(model, data, Y, num_samples,dim):
    # randomly sample the number of samples 
    idx = rnd.choice(data.shape[0],num_samples)
    data=torch.from_numpy(data).to(device).float()
    Y=torch.from_numpy(Y).to(device).float()
    Y=convert_to_ClassNumbers(Y)
    y=Y[idx]    
    z = data[idx,:]
    model = model.eval()
    z = model.encoder(z)
    z = z.to('cpu').detach().numpy()
    y = y.to('cpu').detach().numpy()    
    D = sil(z,y)
    plt.figure()
    if dim==3:
        ax=plt.axes(projection="3d")
        p=ax.scatter3D(z[:, 0], z[:, 1],z[:,2], c=y, cmap='tab10')        
        plt.colorbar(p)
    if dim==2:
        ax=plt.axes        
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        plt.colorbar()
    
    return D



def training_loop_iAE(model,num_epochs,batch_size,opt,batch_val,
                      patience,gradient_clipping,filename,
                      Xtrain,Ytrain,Xtest,Ytest,
                      input_size,hidden_size,latent_dims,num_classes):    
    
    num_batches = math.ceil(Xtrain.shape[0]/batch_size)
    recon_criterion = nn.MSELoss(reduction='sum')
    classif_criterion = nn.CrossEntropyLoss(reduction='sum')
    print('Starting training')
    goat_loss=99999
    counter=0
    for epoch in range(num_epochs):
      #shuffle the data    
      idx = rnd.permutation(Xtrain.shape[0]) 
      idx_split = np.array_split(idx,num_batches)
      
      if epoch==100:
          for g in opt.param_groups:
              g['lr']=1e-4
        
      for batch in range(num_batches):
          # get the batch           
          samples = idx_split[batch]
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
    
    # loading the model back
    model_goat = iAutoencoder(input_size,hidden_size,latent_dims,num_classes)
    model_goat.load_state_dict(torch.load(filename))
    model_goat=model_goat.to(device)
    model_goat.eval()
    return model_goat



# def training_loop_VAE(model,num_epochs,batch_size,opt,batch_val,
#                       patience,gradient_clipping,filename,
#                       Xtrain,Ytrain,Xtest,Ytest,
#                       vae,input_size,hidden_dim,latent_dim,num_classes,dropout):    
    
#     num_batches = math.ceil(Xtrain.shape[0]/batch_size)    
#     classif_criterion = nn.CrossEntropyLoss(reduction='sum')
#     print('Starting training')
#     goat_loss=99999
#     counter=0
#     kl_loss=np.array([])
#     recon_loss = np.array([])
#     total_loss = np.array([])
#     for epoch in range(num_epochs):
#       #shuffle the data    
#       idx = rnd.permutation(Xtrain.shape[0]) 
#       idx_split = np.array_split(idx,num_batches)
      
#       if epoch==100:
#           for g in opt.param_groups:
#               g['lr']=1e-4
        
#       for batch in range(num_batches):
#           # get the batch           
#           samples = idx_split[batch]
#           Xtrain_batch = Xtrain[samples,:]
#           Ytrain_batch = Ytrain[samples,:]        
          
#           #push to gpu
#           Xtrain_batch = torch.from_numpy(Xtrain_batch).float()
#           Ytrain_batch = torch.from_numpy(Ytrain_batch).float()          
#           Xtrain_batch = Xtrain_batch.to(device)
#           Ytrain_batch = Ytrain_batch.to(device)
          
#           # forward pass thru network
#           opt.zero_grad() 
#           recon,vae_loss = model(Xtrain_batch)
#           #latent_activity = model.encoder(Xtrain_batch)      
          
#           # get loss                
#           #classif_loss = (classif_criterion(decodes,Ytrain_batch))/Xtrain_batch.shape[0]      
#           loss = vae_loss.mean() 
#           tmp_kl_loss = model.encoder.kl.mean().item()
#           tmp_recon_loss = model.logprob_x.mean().item()
#           kl_loss = np.append(kl_loss,tmp_kl_loss)
#           recon_loss = np.append(recon_loss,tmp_recon_loss)
#           total_loss = np.append(total_loss,loss.item())
#           #print(classif_loss.item())
          
#           # compute accuracy
#           # ylabels = convert_to_ClassNumbers(Ytrain_batch)        
#           # ypred_labels = convert_to_ClassNumbers(decodes)     
#           # accuracy = (torch.sum(ylabels == ypred_labels).item())/ylabels.shape[0]
          
#           # backpropagate thru network 
#           loss.backward()
#           nn.utils.clip_grad_value_(model.parameters(), clip_value=gradient_clipping)
#           opt.step()
      
#       # get validation losses
#       val_loss=validation_loss_vae(model,Xtest,Ytest,batch_val,1)    
    
      
#       print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss {loss:.2f}, Val. Loss {val_loss:.2f}, Kl loss {tmp_kl_loss:.2f}, Recon loss {tmp_recon_loss:.2f}')
      
#       if val_loss<goat_loss:
#         goat_loss = val_loss            
#         counter = 0
#         print('Goat loss, saving model')      
#         torch.save(model.state_dict(), filename)
#       else:
#         counter += 1
 
#       if counter>=patience:
#         print('Early stoppping point reached')
#         print('Best val loss')
#         print(goat_loss)
#         break
 
#     # loading the model back
#     model_goat = vae(input_size,hidden_dim,latent_dim,num_classes,dropout)
#     model_goat.load_state_dict(torch.load(filename))
#     model_goat=model_goat.to(device)
#     model_goat.eval()
#     return model_goat


# # function to validate model 
# def validation_loss_vae(model,X_test,Y_test,batch_val,val_type):            
#     loss_val=0    
#     if batch_val > X_test.shape[0]:
#         batch_val = X_test.shape[0]
    
#     idx=np.arange(0,X_test.shape[0],batch_val)    
#     if idx[-1]<X_test.shape[0]:
#         idx=np.append(idx,X_test.shape[0])
#     else:
#         print('something wrong here')
    
#     iters=(idx.shape[0]-1)
    
#     for i in np.arange(iters):
#         x=X_test[idx[i]:idx[i+1],:]
#         y=Y_test[idx[i]:idx[i+1],:]     
#         with torch.no_grad():                
#             if val_type==1: #validation
#                 x=torch.from_numpy(x).to(device).float()
#                 y=torch.from_numpy(y).to(device).float()
#                 model.eval()
#                 out_recon,out_loss = model(x)                 
#                 loss_val += out_loss.sum().item()
#                 model.train()
#             else:
#                 out = model(x) 
#                 loss1 = crit_recon_val(out,x)                
#                 loss_val += loss1.item()             
            
#     loss_val=loss_val/X_test.shape[0]        
#     torch.cuda.empty_cache()
#     return loss_val