# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 22:39:09 2022

@author: Nikhlesh
"""

import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.autograd import Variable 


# setting up GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

df = pd.read_csv("C:/Users/nikic/Documents/GitHub/NN/SBUX.csv", index_col = "Date", parse_dates=True)

plt.style.use("ggplot")
#df["Volume"].plot(label="CLOSE", title="Star Bucks Stock Volume")
df["Volume"].plot(title="Star Bucks Stock Volume")




X = df.iloc[:, :-1]
y = df.iloc[:, 5:6] 

mm = MinMaxScaler()
ss = StandardScaler()

X_ss = ss.fit_transform(X)
y_mm = mm.fit_transform(y) 


#first 200 for training
X_train = X_ss[:200, :]
X_test = X_ss[200:, :]

y_train = y_mm[:200, :]
y_test = y_mm[200:, :] 

print("Training Shape", X_train.shape, y_train.shape)
print("Testing Shape", X_test.shape, y_test.shape) 

X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test)) 


#reshaping to rows, timestamps, features
X_train_tensors_final = torch.reshape(X_train_tensors,
                                      (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))


X_test_tensors_final = torch.reshape(X_test_tensors,  
                                     (X_test_tensors.shape[0], 1, X_test_tensors.shape[1])) 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

class LSTM1(nn.Module):
    def __init__(self,num_classes,input_size,hidden_size,num_layers,seq_length,projection_size,fc_nodes,bidirectional_flag):
        super(LSTM1,self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
      #  self.projection_size = projection_size #projection length
        self.fc_nodes = fc_nodes #number of nodes in the forward MLP
        self.bidirectional_flag=bidirectional_flag # flag if the LSTM is bidirectional 
        if self.bidirectional :
            self.layer_ratio = 2         
        else:
            self.layer_ratio = 1
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, proj_size= projection_size,bidirectional = bidirectional_flag,
                          batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, fc_nodes) #fully connected 1
        self.fc = nn.Linear(fc_nodes, num_classes) #fully connected last layer
        self.relu = nn.ReLU()

    def forward(self,x):
        h_0 = Variable(torch.zeros(self.layer_ratio*self.num_layers, x.size(0), self.hidden_size)) #layers, batch size, hidden size
        c_0 = Variable(torch.zeros(self.layer_ratio*self.num_layers, x.size(0), self.hidden_size)) #layers, batch size, hidden size
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out

num_epochs = 10000 #1000 epochs
learning_rate = 0.001 #0.001 lr
input_size = 5 #number of features
hidden_size = 2 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers
num_classes = 1 #number of output classes 



lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1]) #our lstm class 
lstm1 = lstm1.to(device)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate) 
X_train_tensors_final = X_train_tensors_final.to(device)
y_train_tensors = y_train_tensors.to(device)




for epoch in range(num_epochs):
  outputs = lstm1.forward(X_train_tensors_final) #forward pass
  optimizer.zero_grad() #caluclate the gradient, manually setting to 0
 
  # obtain the loss function
  loss = criterion(outputs, y_train_tensors)
 
  loss.backward() #calculates the loss of the loss function
 
  optimizer.step() #improve from loss, i.e backprop
  if epoch % 100 == 0:
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 


df_X_ss = ss.transform(df.iloc[:, :-1]) #old transformers
df_y_mm = mm.transform(df.iloc[:, -1:]) #old transformers

df_X_ss = Variable(torch.Tensor(df_X_ss)) #converting to Tensors
df_y_mm = Variable(torch.Tensor(df_y_mm))
#reshaping the dataset
df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1])) 


train_predict = lstm1(df_X_ss)#forward pass
data_predict = train_predict.data.numpy() #numpy conversion
dataY_plot = df_y_mm.data.numpy()

data_predict = mm.inverse_transform(data_predict) #reverse transformation
dataY_plot = mm.inverse_transform(dataY_plot)


plt.figure(figsize=(10,6)) #plotting
plt.axvline(x=200, c='r', linestyle='--') #size of the training set
plt.plot(dataY_plot)
plt.plot(data_predict, label='Predicted Data') #predicted plot
plt.title('Time-Series Prediction')
plt.legend()
plt.show() 


num_classes=7
input_size=256
hidden_size=32
proj_size=15
if proj_size>0:
    hidden_state_size = proj_size
else:
    hidden_state_size = hidden_size
sequence_length=100
num_layers=2
batch_size=64
bidir_flag=True
if bidir_flag:
    D=2
else:
    D=1
fc_nodes=25

# init the LSTM        
rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
              batch_first=True,proj_size =proj_size,bidirectional =bidir_flag) #input dim, hidden size, num layers (stacked lstm)
input = torch.randn(batch_size, sequence_length, input_size) # batch_size, sequence length, features (input_dimension)
h0 = torch.zeros(D*num_layers, batch_size, hidden_state_size) # 1*num_layers (not bidirectional), batch, hidden size
c0 = torch.zeros(D*num_layers, batch_size, hidden_size) # same as above

#output, (hn, cn) = rnn(input, (h0, c0)) 
output, (hn, cn) = rnn(input) 
# out -> batch, seq. length, hidden size of last LSTM layer, concatenated if bilstm
# hn -> D*num_layers, batch size, out_size. essentially contains the final hidden state alone, and not for each time pt in sequence
# cn -> D*num_layers, batch_size, cell hidden state containing the final cell state
tmp = hn[2:,:,:] # get only the activation of the last layer to use in the MLP
tmp = tmp.view(-1,batch_size)

# just get the final hidden state from output
tmp = output[:,-1,:]
# pass through linear layer
linear1 = nn.Linear(hidden_state_size*D, fc_nodes)
gelu = nn.GELU()
linear2 = nn.Linear(fc_nodes,num_classes)
tmp=linear2(gelu(linear1(tmp)))

tmp2 = output
tmp2=tmp2.detach().numpy()
tmp2a = tmp2[:,-1,0:15]
tmp2b = tmp2[:,0,15:30]
tmp2 = np.concatenate((tmp2a,tmp2b),axis=1)

tmp1=hn
tmp1=tmp1.detach().numpy()
tmp1=tmp1[2:,:,:]
tmp1a = tmp1[0,:,:]
tmp1b = tmp1[1,:,:]


plt.plot(tmp1[0,:])
plt.figure();
plt.plot(tmp2[0,:])

# using pytorch functions
tmp1=hn
tmp1=tmp1[2:,:,:]
tmp1=torch.permute(tmp1, (1,0,2))
tmp1 = torch.flatten(tmp1,start_dim=1,end_dim=2)
tmp1=tmp1.detach().numpy()


plt.plot(tmp1[11,:])
plt.figure();
plt.plot(tmp2[11,:])

# =============================================================================
# # general rule of thumb -> hidden state for bidirectional LSTM contains the final forward and reverse hidden states.
# # The output in bidirectional LSTMs contains final forward hidden state and the initial reverse hidden state. So when using
# # stacks of biLSTMs, have to parse out the hidden state and feed it to the MLP -> classifier. In pytroch seems to be difference 
# between hidden state and output states
# 
# =============================================================================

#use a function to implement a bidirectional lstm
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

num_classes=7
input_size=256
hidden_size=150
proj_size=75
num_layers=2
dropout_val=0.3
fc_nodes=25
model =  LSTM1(num_classes, input_size, hidden_size,proj_size,
               num_layers,fc_nodes, dropout_val) 
model = model.to(device)


# push to gpu

input=input.to(device)
opt = torch.optim.Adam(model.parameters())
opt.zero_grad() 
out = model(input)
        



# creating a bidir LSTM -> GRU -> MLP, with GRU having half the number of hidden nodes
class rnn_gru(nn.Module):
    def __init__(self,num_classes,input_size,hidden_size,num_layers,dropout_val):
        super(rnn_gru,self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size       
        
        self.gru1=nn.GRU(input_size=input_size,hidden_size=hidden_size,
                          num_layers=num_layers,batch_first=True,dropout=dropout_val,
                          bidirectional=True)
        self.gru2=nn.GRU(input_size=round(hidden_size*2),hidden_size=round(hidden_size/2),
                          num_layers=num_layers,batch_first=True, bidirectional=False)
        
        self.mlp_input = round(hidden_size/2              )
        self.linear1 = nn.Linear(self.mlp_input,num_classes)
    
    def forward(self,x):        
        output1, (hn1) = self.gru1(x) 
        output2, (hn2) = self.gru2(output1)
        hn2 = torch.squeeze(hn2)        
        out = self.linear1(hn2)        
        return out

input_size=256
hidden_size=150
num_layers=1
batch_size=64
sequence_length = 100
num_classes=7
dropout_val=0.3


rnn1 = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
              batch_first=True,bidirectional = True,dropout=dropout_val) 
rnn2 = nn.GRU(input_size=round(hidden_size*2), hidden_size=round(hidden_size/2), num_layers=num_layers,
              batch_first=True,bidirectional = False) 
linear1 = nn.Linear(round(hidden_size/2),num_classes)
input = torch.randn(batch_size, sequence_length, input_size) # batch_size, sequence length, features (input_dimension)

output1, (hn1) = rnn1(input) 
output2, (hn2) = rnn2(output1)
hn2  = torch.squeeze(hn2)
out = linear1(hn2)

# conv1d
input1= torch.permute(input, (0,2,1))
conv_layer = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=3,stride=2)
out1 = conv_layer(input1)
out1 = torch.permute(out1,(0,2,1))

# out -> batch, seq. length, hidden size of last LSTM layer, concatenated if bilstm
# hn -> D*num_layers, batch size, out_size. essentially contains the final hidden state alone, and not for each time pt in sequence
# cn -> D*num_layers, batch_size, cell hidden state containing the final cell state

model = rnn_gru(num_classes,input_size,hidden_size,num_layers,dropout_val)
out = model(input)
