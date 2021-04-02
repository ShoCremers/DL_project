#!/usr/bin/env python
# coding: utf-8

# In[62]:


# Setup dependencies (as taken from assignment 6)
import os
import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models

from torchsummary import summary
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

#Additional Setup to use Tensorboard
# get_ipython().system('pip install -q tensorflow')
# get_ipython().run_line_magic('load_ext', 'tensorboard')


# In[137]:


df = pd.read_csv('day_ahead.csv')
df = df.set_index('datetime')
df.index = pd.to_datetime(df.index)
# df=df["Day-ahead Price [EUR/MWh]"].to_frame()
df


# # Create Torch dataset

# In[138]:


# sequence length (edit the value for different sequence length)
seq = 24 


# In[139]:


delta = pd.Timedelta(seq, unit ='h')
# define 1 hour object for convenience when using datetime as index in the dataframe to not include the last item
hours_12 = pd.Timedelta(12, unit ='h') # used mostly for empty 12 hours 
hour = pd.Timedelta(1, unit ='h')
day = pd.Timedelta(1, unit ='d')


# In[140]:


### creating training dataset
train_y_start = dt.datetime(2015, 1, 5, 0, 0) + (delta+hours_12).ceil('1d')
#train_x_start = train_y_start - delta - hours_12
train_end = dt.datetime(2020, 11, 30, 23, 0)

train_x = []
train_y = []
while train_y_start + day - hour <= train_end:
    train_x_start = train_y_start - delta - hours_12
    
    
    #print(train_x_start, train_y_start)
    train_x.append(df[train_x_start:train_x_start+delta - hour].values)
    train_y.append(df[train_y_start:train_y_start+day - hour]['Day-ahead Price [EUR/MWh]'].values)
    
    train_y_start += day
    
train_x = np.asarray(train_x)
train_y = np.asarray(train_y)
print(train_x.shape)
print(train_y.shape)
print(train_x)
print(train_y)


# In[141]:


### creating testing dataset
test_y_start = dt.datetime(2020, 12, 1, 0, 0)
test_end = dt.datetime(2020, 12, 31, 23, 0)

test_x = []
test_y = []
while test_y_start + day - hour <= test_end:
    test_x_start = test_y_start - delta - hours_12
    
    test_x.append(df[test_x_start:test_x_start+delta - hour].values)
    test_y.append(df[test_y_start:test_y_start+day - hour]['Day-ahead Price [EUR/MWh]'].values)
    
    test_y_start += day

test_x = np.asarray(test_x)
test_y = np.asarray(test_y)
print(test_x.shape)
print(test_y.shape)
print(test_x[0])


# In[142]:
if torch.cuda.is_available():
    dev = 'cuda:0'
    print('You have CUDA device.')
else:
    dev = 'cpu'
    print('Switch to GPU runtime to speed up computation.')

# create tensor objects
x_train = torch.from_numpy(train_x).float().to(dev)
y_train = torch.from_numpy(train_y).float().to(dev)
x_test = torch.from_numpy(test_x).float().to(dev)
y_test = torch.from_numpy(test_y).float().to(dev)
# train_loader = DataLoader(x_train,y_train, batch_size=128, shuffle=False)
# val_loader = DataLoader(x_test,y_test batch_size=128, shuffle=True)


# # Define BLSTM model

# In[143]:


class BLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, quantiles):
        super(BLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
#         self.fc = nn.Linear(hidden_dim*2, output_dim) # multiply hidden_dim by 2 because bidirectional
        
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.out_shape = len(quantiles)
        
        final_layers = [
            nn.Linear(hidden_dim*2, output_dim) for _ in range(len(self.quantiles))
        ]
        self.final_layers = nn.ModuleList(final_layers)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).requires_grad_() #hidden layer output
        # Initialize cell state
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_dim).requires_grad_() 
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        # Index hidden state of last time step
#         _out = self.fc(out[:, -1, :])
        
        return torch.stack([layer(out[:, -1, :]) for layer in self.final_layers], dim=1)
        
        
#         return out


# ## Training

# In[144]:


num_train = x_train.shape[0]
input_dim = x_train.shape[2]
output_dim = 24 
hidden_dim = 20 # no. of neurons in hidden layer
num_layers = 3 # no of hidden layers 
num_epochs = 2
# print(x_train[i].unsqueeze(0))


# In[ ]:





# In[145]:


# criterion = nn.MSELoss(reduction='mean')
# quatile loss implementation
class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, preds, target):
        assert not target.requires_grad
#         print("inside QuantileLoss")
#         print(preds.size(0), target.size(0))
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(
                torch.max(
                   (q-1) * errors, 
                   q * errors
            ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss

quantiles = [.01,0.05, 0.10,0.25, .5, 0.75, 0.90, 0.95, .99]
criterion = QuantileLoss(quantiles)

model = BLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, quantiles=quantiles).to(dev)
# for practice use MSE, in real experiment use NLLLOSS for parametric
print(model)

#criterion = nn.NLLLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)


# In[146]:


# training loop
for t in range(num_epochs): 
    err = 0
    for i in range(num_train):
        y_train_pred = model(x_train[i].unsqueeze(0))
        
#         print(y_train_pred.shape)
        loss = criterion(torch.transpose(y_train_pred[0],0,1), y_train[i])
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        err += loss.item()
        #print("item ", t, "MSE: ", loss.item())
        
    print("Epoch ", t, "MSE: ", err/num_train)


# In[147]:



# Make the prediction on the meshed x-axis
model.eval()
with torch.no_grad():
    preds=model(x_test)


# In[148]:


print(preds.shape)


# In[149]:


test_df = df[dt.datetime(2020, 12, 1, 0, 0):dt.datetime(2020, 12, 31, 23, 0)][['Day-ahead Price [EUR/MWh]']]
# print(preds.shape[0])
# for i in range(preds.shape[1]):
#     y=preds[:,i,:]
#     test_df[str(i)]=y.flatten()
# test_df.plot()    

# y_lower, y_pred, y_upper = preds[:, 0,:], preds[:, 1,:], preds[:, 2,:],
# print(y_lower.shape)


# test_df['q1'] = y_lower.flatten()
# test_df['q2'] = y_pred.flatten()
# test_df['q3'] = y_upper.flatten()
# test_df.plot();


# Plot the function, the prediction and the 90% confidence interval based on
# # the MSE
# fig = plt.figure()
# plt.plot(xx, f(xx), 'g:', label=u'$f(x) = x\,\sin(x)$')
# plt.plot(X, y, 'b.', markersize=10, label=u'Observations')
# plt.plot(xx, y_pred, 'r-', label=u'Prediction')
# plt.plot(xx, y_upper, 'k-')
# plt.plot(xx, y_lower, 'k-')
# plt.fill(np.concatenate([xx, xx[::-1]]),
#          np.concatenate([y_upper, y_lower[::-1]]),
#          alpha=.5, fc='b', ec='None', label='90% prediction interval')
# plt.xlabel('$x$')
# plt.ylabel('$f(x)$')
# plt.ylim(-10, 20)
# plt.legend(loc='upper left')
# plt.show()


# In[150]:


fig = plt.figure(figsize=(20,10))
plt.plot(test_df.index, test_df['Day-ahead Price [EUR/MWh]'].values, 'r', label='expected')

plt.plot(test_df.index, preds[:,0,:].flatten(), 'b,',markersize=0,label='0.01')
plt.plot(test_df.index, preds[:,8,:].flatten(), 'b.',markersize=0,label='0.99')

plt.plot(test_df.index, preds[:,1,:].flatten(), 'r.',markersize=0,label='0.05')
plt.plot(test_df.index, preds[:,7,:].flatten(), 'r.',markersize=0,label='0.95')

# plt.plot(test_df.index, y_pred, 'r-', label=u'Prediction')
# plt.plot(test_df.index, y_upper, 'k-')
# plt.plot(test_df.index, y_lower, 'k-')
plt.fill(np.concatenate([test_df.index, test_df.index[::-1]]),
         np.concatenate([preds[:,8,:].flatten(), preds[:,0,:].flatten()]),
         alpha=.25, fc='grey', ec='None', label='98% prediction interval')

plt.fill(np.concatenate([test_df.index, test_df.index[::-1]]),
         np.concatenate([preds[:,7,:].flatten(), preds[:,1,:].flatten()]),
         alpha=.5, fc='grey', ec='None', label='90% prediction interval')

plt.fill(np.concatenate([test_df.index, test_df.index[::-1]]),
         np.concatenate([preds[:,6,:].flatten(), preds[:,2,:].flatten()]),
         alpha=.75, fc='grey', ec='None', label='80% prediction interval')

plt.fill(np.concatenate([test_df.index, test_df.index[::-1]]),
         np.concatenate([preds[:,5,:].flatten(), preds[:,3,:].flatten()]),
         alpha=0.9, fc='grey', ec='None', label='50% prediction interval')

plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
plt.savefig("result.png")


