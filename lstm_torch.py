# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 22:39:09 2022

@author: Nikhlesh
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.autograd import Variable 


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



input_size=10
hidden_size=20
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

# init the LSTM        
rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
              batch_first=True,proj_size =proj_size,bidirectional =bidir_flag) #input dim, hidden size, num layers (stacked lstm)
input = torch.randn(batch_size, sequence_length, input_size) # batch_size, sequence length, features (input_dimension)
h0 = torch.zeros(D*num_layers, batch_size, hidden_state_size) # 1*num_layers (not bidirectional), batch, hidden size
c0 = torch.zeros(D*num_layers, batch_size, hidden_size) # same as above

output, (hn, cn) = rnn(input, (h0, c0)) 
# out -> batch, seq. length, hidden size of last LSTM layer, concatenated if bilstm
# hn -> D*num_layers, batch size, out_size. essentially contains the final hidden state alone, and not for each time pt in sequence
# cn -> D*num_layers, batch_size, cell hidden state containing the final cell state
tmp = hn[2:,:,:] # get only the activation of the last layer to use in the MLP
tmp = tmp.view(-1,batch_size)

# =============================================================================
# # general rule of thumb -> hidden state for bidirectional LSTM contains the final forward and reverse hidden states.
# # The output in bidirectional LSTMs contains final forward hidden state and the initial reverse hidden state. So when using
# # stacks of biLSTMs, have to parse out the hidden state and feed it to the MLP -> classifier. In pytroch seems to be difference 
# between hidden state and output states
# 
# =============================================================================


