# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 22:47:39 2021

@author: Nikhlesh
"""
# testing variational autoencoders on the MNIST database using pytorch

# importing everything
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200
import numpy as np
import numpy.random as rnd
from utils import *

# setting up GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# full implementation of VAE using distributions, on MNIST and then BCI data, 
# and then with a classifier layer in the middle 


# encoder, decoder, classifier, parameters of the latent representation

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
class vae(nn.Module):
    def __init__(self,input_size,hidden_dim,latent_dim,num_classes,dropout):
        super(vae,self).__init__()
        self.encoder = VariationalEncoder(input_size,hidden_dim,latent_dim,dropout)
        self.decoder = VariationalDecoder(input_size,hidden_dim,latent_dim,dropout)
        self.classifier = latentClassifier(latent_dim,num_classes) 
        self.logprob_x=0
        self.kl_loss=0
    
    def forward(self,x):
        z=self.encoder(x)
        self.kl_loss = self.encoder.kl
        #y=self.classifier(z)
        xhat,mu_p,sig_p=self.decoder(z)
        pdist = torch.distributions.Normal(mu_p, sig_p)
        pdist.loc = pdist.loc.to(device)
        pdist.scale = pdist.scale.to(device)
        self.logprob_x = pdist.log_prob(x).sum(-1) # summing over all dimensions
        vae_loss = self.kl_loss - self.logprob_x                
        return xhat,vae_loss
    
input_size=96
hidden_dim=32
latent_dim=3
num_classes=7
dropout=0

vae_model = vae(input_size,hidden_dim,latent_dim,num_classes,dropout).to(device)
input_data = torch.randn(128,96).to(device)
xhat,vae_loss = vae_model(input_data)

condn_data_imagined = np.array(rnd.randn(30000,96))
Y=np.zeros((int(3e4),1))
idx = np.arange(0,1e4,dtype=int)
condn_data_imagined[idx,:] = condn_data_imagined[idx,:] + 0
idx = np.arange(1e4,2e4,dtype=int)
condn_data_imagined[idx,:] = condn_data_imagined[idx,:] - 1
Y[idx]=1
idx = np.arange(2e4,3e4,dtype=int)
condn_data_imagined[idx,:] = condn_data_imagined[idx,:] + 1
Y[idx]=2
plt.stem(np.mean(condn_data_imagined,axis=1))
plt.figure();plt.stem(Y)

# train and testing split
Y_mult = np.zeros((Y.shape[0],3))
for i in range(Y.shape[0]):
    tmp = int(Y[i])
    Y_mult[i,tmp]=1
Y = Y_mult

#condn_data_imagined = scale_zero_one(condn_data_imagined)            
Xtrain,Xtest,Ytrain,Ytest = training_test_split(condn_data_imagined,Y,0.8)

# run it through the vae training loop 
num_epochs=25
batch_size=128
learning_rate = 1e-3
batch_val=512
patience=6
gradient_clipping=10
filename='vae_test.pth'
model = vae_model
opt = torch.optim.Adam(model.parameters(),lr=learning_rate)

def training_loop_VAE(model,num_epochs,batch_size,opt,batch_val,
                      patience,gradient_clipping,filename,
                      Xtrain,Ytrain,Xtest,Ytest):    
    
    num_batches = math.ceil(Xtrain.shape[0]/batch_size)    
    classif_criterion = nn.CrossEntropyLoss(reduction='sum')
    print('Starting training')
    goat_loss=99999
    counter=0
    kl_loss=np.array([])
    recon_loss = np.array([])
    total_loss = np.array([])
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
          recon,vae_loss = model(Xtrain_batch)
          #latent_activity = model.encoder(Xtrain_batch)      
          
          # get loss                
          #classif_loss = (classif_criterion(decodes,Ytrain_batch))/Xtrain_batch.shape[0]      
          loss = vae_loss.mean()
          tmp_kl_loss = model.encoder.kl.mean().item()
          tmp_recon_loss = model.logprob_x.mean().item()
          kl_loss = np.append(kl_loss,tmp_kl_loss)
          recon_loss = np.append(recon_loss,tmp_recon_loss)
          total_loss = np.append(total_loss,loss.item())
          #print(classif_loss.item())
          
          # compute accuracy
          # ylabels = convert_to_ClassNumbers(Ytrain_batch)        
          # ypred_labels = convert_to_ClassNumbers(decodes)     
          # accuracy = (torch.sum(ylabels == ypred_labels).item())/ylabels.shape[0]
          
          # backpropagate thru network 
          loss.backward()
          nn.utils.clip_grad_value_(model.parameters(), clip_value=gradient_clipping)
          opt.step()
      
      # get validation losses
      #val_loss,val_acc,val_recon=validation_loss_vae(model,Xtest,Ytest,batch_val,1)    
      #val_loss,val_recon=validation_loss_regression(model,Xtest,Ytest,batch_val,1)    
      
      print(f'Epoch [{epoch}/{num_epochs}], Train Loss {loss:.2f}, Kl loss {tmp_kl_loss:.2f}, Recon loss {tmp_recon_loss:.2f}')
      #print(f'Epoch [{epoch}/{num_epochs}], Val. Loss {val_loss:.2f}, Train Loss {total_loss:.2f}, Val. Acc {val_acc*100:.2f}, Train Acc {accuracy*100:.2f}')
      #print(f'Epoch [{epoch}/{num_epochs}], Val. Loss {val_loss:.4f}, Train Loss {total_loss:.4f}')
      
      # if val_loss<goat_loss:
      #     goat_loss = val_loss
      #     goat_acc = val_acc*100      
      #     counter = 0
      #     print('Goat loss, saving model')      
      #     torch.save(model.state_dict(), filename)
      # else:
      #     counter += 1
    
      # if counter>=patience:
      #     print('Early stoppping point reached')
      #     print('Best val loss and val acc  are')
      #     print(goat_loss,goat_acc)
      #     break
    
    # loading the model back
    model_goat = iAutoencoder(input_size,hidden_size,latent_dims,num_classes)
    model_goat.load_state_dict(torch.load('autoencoder.pth'))
    model_goat=model_goat.to(device)
    model_goat.eval()
    return model_goat



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


        
D = plot_latent(model, Xtest,Ytest,Xtest.shape[0],3)

D = plot_latent(model, condn_data_imagined,Y,2100,3)





# # playing around with Kl stuff in pytorch 
# p = torch.distributions.Normal(0,1)
# q = torch.distributions.Normal(0,2)


# prob_p=np.array([])
# prob_q=np.array([])
# k_values=np.array([])

# for i in range(100):
#     z = q.rsample()   
#     log_pz = p.log_prob(z)
#     log_qz = q.log_prob(z)
#     prob_p =  np.append(prob_p,torch.exp(log_pz).numpy())
#     prob_q =  np.append(prob_q,torch.exp(log_qz).numpy())
#     d = log_qz - log_pz
#     k_values =  np.append(k_values,d.numpy())
    
# plt.figure()
# plt.hist(k_values)
# plt.figure()
# plt.hist(prob_p)
# plt.figure()
# plt.hist(prob_q)

