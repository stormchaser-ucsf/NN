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

# setting up GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# the first dimension in torch is always the batch size or the batch input data
# thus why data is flattened along the first dimension


# create encoder module with linear layers
class encoder(nn.Module):
    def __init__(self,latent_dims):
        super(encoder,self).__init__()
        self.linear1 = nn.Linear(784,256)
        self.linear11 = nn.Linear(256,64)
        self.linear2 = nn.Linear(64,latent_dims)
       
    def forward(self,x):
        x = torch.flatten(x,start_dim=1)
        x = self.linear1(x)
        x = F.elu(x)
        x = F.elu(self.linear11(x))
        x = self.linear2(x)
        return x

class decoder(nn.Module):
    def __init__(self,latent_dims):
        super(decoder,self).__init__()
        self.linear1 = nn.Linear(latent_dims,64)
        self.linear11 = nn.Linear(64,256)
        self.linear2 = nn.Linear(256,784)
       
    def forward(self,z):        
        z = self.linear1(z)
        z = F.elu(z)
        z = F.elu(self.linear11(z))
        z = self.linear2(z)
        z = z.reshape((-1, 1, 28, 28))    
        return z


class mlp_classifier(nn.Module):
    def __init__(self,num_classes,latent_dims):
        super(mlp_classifier,self).__init__()
        self.linear1 = nn.Linear(784,num_classes)
   
    def forward(self,x):
        x = torch.flatten(x,start_dim=1)
        x = self.linear1(x)
        return x



class autoencoder(nn.Module):
    def __init__(self, latent_dims,num_classes):
        super(autoencoder,self).__init__()
        self.encoder = encoder(latent_dims)
        self.decoder = decoder(latent_dims)
        self.classifier = mlp_classifier(num_classes,latent_dims)
       
    def forward(self,x):
        z=self.encoder(x)
        #z1 = self.classifier(z)
        z=self.decoder(z)       
        z1 = self.classifier(z)
        return z,z1
   

latent_dims = 3
num_classes=10
linearAE = autoencoder(latent_dims,num_classes).to(device) # GPU
       
   
# training loop to train on MNIST dataset
data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data',
               transform=torchvision.transforms.ToTensor(),
               download=True),
        batch_size=512,
        shuffle=True)



num_epochs=175
n_total_steps = len(data)
criterion = nn.MSELoss(reduction='sum')
criterion2 = nn.CrossEntropyLoss()

def train(linearAE, data, epochs =num_epochs):
    opt  = torch.optim.Adam(linearAE.parameters())
    for epoch in range(epochs):
        for i,(x,y) in enumerate(data):
            x=x.to(device) # push it to GPU
            y=y.to(device) # push it to GPU
            opt.zero_grad() # flush gradients
            xhat,xclass = linearAE(x)
            #loss = ((x - xhat)**2).sum()
            loss1=criterion(x,xhat)
            loss2=criterion2(xclass,y)
            loss=loss2+0*loss1
            loss.backward()
            opt.step()
            #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    return linearAE


linearAE = train(linearAE, data)

# get one training batch of samples
x,y,=next(iter(data))

# # pytorch engineer method of training
# criterion = nn.MSELoss()
# opt  = torch.optim.Adam(linearAE.parameters())
# n_total_steps = len(data)
# for epoch in range(num_epochs):
#     for i,(x,y) in enumerate(data):
#         x=x.to(device)
       
#         # foward pass
#         xhat = linearAE(x)
#         #loss = ((x - xhat)**2).sum()
#         loss = criterion(x,xhat)
       
#         #backward pass and optimize
#         opt.zero_grad() # flush gradients
#         loss.backward()
#         opt.step()
#         if (i+1) % 100 == 0:
#             print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

#### end pytroch engineer method of training        





# plotting
def plot_latent(linearAE, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = linearAE.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 1], z[:, 0], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
plot_latent(linearAE, data)



# x=torch.randn(100,28,28)
# z = linearAE.encoder(x.to(device))
# z = z.to('cpu').detach().numpy()
# plt.scatter(z[:, 0], z[:, 1],z[:, 2])




# # changing the forward pass to be reflective of the VAE framework
# class VariationalEncoder(nn.Module):
#     def __init__(self,latent_dims):
#         super(VariationalEncoder, self).__init__()
#         self.linear1 = nn.Linear(784,256)
#         self.linear2 = nn.Linear(256,latent_dims) # for the mean
#         self.linear3 = nn.Linear(256,latent_dims) # for the std
       
#         # gaussian samplimg
#         self.N = torch.distributions.Normal(0, 1)
#         self.N.loc = self.N.loc.cuda() #sampling on GPU loc-mean
#         self.N.scale = self.N.loc.scale.cuda() #sampling on GPU scale -std
#         self.kl= 0 #KL loss
   
#     def forward(self,x):
#         x=torch.flatten(x,start_dim=1) #first dim is batch
#         x=self.linear1(x)
#         x=F.relu(x)
#         mu = self.linear2(x)
#         sigma = self.linear3(x)
#         sigma = torch.exp(sigma) # exp of std
#         z = mu + sigma*self.N.sample(mu.shape) # sampling from the normal distribution
#         self.kl = (sigma**2+mu**2 - torch.log(sigma) -1/2).sum
#         return z

# # combining both into one
# class VariationalAutoencoder(nn.Module):
#     def __init__(self,latent_dims):
#         super(VariationalAutoencoder,self).__init__()
#         self.encoder = VariationalEncoder(latent_dims)
#         self.decoder = decoder(latent_dims)
   
#     def forward(self,x):
#         z=self.encoder(x)
#         z=self.decoder(z)
#         return z

# # the loss function in the training algorithm
# num_epochs=25
# n_total_steps = len(data)
# criterion = nn.MSELoss(reduction='sum')
# def train(ae, data, epochs =num_epochs):
#     opt  = torch.optim.Adam(ae.parameters())
#     for epoch in range(epochs):
#         for i,(x,y) in enumerate(data):
#             x=x.to(device) # push it to GPU
#             opt.zero_grad() # flush gradients
#             xhat,xclass = ae(x)
#             #loss = ((x - xhat)**2).sum()
#             loss=criterion(x,xhat) + ae.encoder.kl            
#             loss.backward()
#             opt.step()
#             #print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
#             if (i+1) % 100 == 0:
#                 print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

#     return ae


# latent_dims = 3
# vae = VariationalAutoencoder(latent_dims).to(device) # GPU

# vae = train(vae, data)

# x,y=next(iter(data))

# # ################## extra stuff ##################
# # # plotting images using matplotlib
# # fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1) # two axes on figure
# # x=torch.randn(28,28)
# # ax1.imshow(x)

# # # understanding the flatten and the reshape functions
# # x = torch.flatten(x)
# # z=x
# # z=z.reshape(28,28)
# # ax2.imshow(z)

# # x=np.random.randn(28,28)
# # y=np.random.randn(28,28)
# # z=((x-y)**2)
# # print(z)



# # nn.Linear






