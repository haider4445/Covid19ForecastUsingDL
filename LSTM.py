import torch
from torch import nn, optim
from datetime import timedelta
from time import time
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

class COVID_LSTM(nn.Module):
    def __init__(self,n_hidden,n_features):
        super().__init__()
        # LSTM takes input as (batch, seq_len=30, input_size=1):
        self.lstm1 = nn.LSTM(input_size =n_features, hidden_size =n_hidden,batch_first =True)
        self.lstm2 = nn.LSTM(input_size =n_hidden, hidden_size =n_hidden,batch_first =True)
        self.lstm3 = nn.LSTM(input_size =n_hidden, hidden_size =n_hidden,batch_first =True)
        self.seq = nn.Sequential(
            nn.Linear(n_hidden,n_features),
            nn.Tanh()
        )
    def forward(self,X):
        out,_ = self.lstm1(X)
        out,_ = self.lstm2(out)
        out,_ = self.lstm3(out)
        out = self.seq(out)
        return out     # out = [Batch size, sequence_length, n_features = 1]
    
    def train_model(self,train_dataloader,criterion,optimizer,n_epochs):
        time_all = time()
        losses = []
        all_epochs_loss = []
        self.train()
        for epoch in range(n_epochs):
            time0 = time()
            one_epoch_loss = []
            for idx,(X,Y_real) in enumerate(train_dataloader):  
                optimizer.zero_grad() 
                predicted = self(X.to(device))[:,-1,:]
                loss = criterion(predicted,Y_real.to(device))
                loss.backward()            
                optimizer.step()
                one_epoch_loss.append(loss.item())

            print("Epoch {} Loss is {}".format(epoch+1,np.mean(one_epoch_loss)))

    #         print("Epoch {} - Time (in minutes) is {}".format(epoch+1,timedelta(seconds=(time()-time0))))
            all_epochs_loss.append(np.mean(one_epoch_loss))

        print("Total Time (in minutes) is {}".format( timedelta(seconds=(time()-time_all))))
        return all_epochs_loss 