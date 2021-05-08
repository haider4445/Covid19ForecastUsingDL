import torch
from torch import nn, optim
from datetime import timedelta
from time import time
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x).to(device)

    
class TimeSeriesTransformer(nn.Module):

    def __init__(self, n_features=2, d_model=128, n_heads=8, n_hidden=128, n_layers=8, dropout=0):
        super().__init__()
        self.model_type = 'Time Series Transformer Model'
        self.InputLinear = nn.Linear(n_features, d_model)
        
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, n_heads, n_hidden, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        
        self.d_model = d_model
        self.n_features = n_features
        
        self.OutputLinear = nn.Linear(d_model, n_features) # The output of the encoder is similar to the input of the encoder, both are (B,S,d_model)

        self.init_weights()
        self.activation = nn.Tanh()
            

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(-1e6)).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def init_weights(self):
        initrange = 0.1
        self.InputLinear.weight.data.uniform_(-initrange, initrange)
        self.OutputLinear.bias.data.zero_()
        self.OutputLinear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask,padding_mask):
        src = self.InputLinear(src) * math.sqrt(self.d_model)
        src = self.positional_encoding(src)
        output = self.transformer_encoder(src, src_mask,padding_mask)
        output = self.OutputLinear(output)
        output = self.activation(output) # output[...,:9] --> Actual 9 values
        return output
    
    def train_model(self,train_dataloader,criterion,optimizer,n_epochs):
        time_all = time()
        losses = []
        all_epochs_loss = []
        self.train()
        S = train_dataloader.dataset.tensors[0].shape[1] # Sequence Length
        src_mask = self.generate_square_subsequent_mask(S)
        for epoch in range(n_epochs):
            time0 = time()
            one_epoch_loss = []
            for idx,(X,Y_real) in enumerate(train_dataloader):  
                optimizer.zero_grad() 
                predicted = self(X.permute(1,0,2).to(device),None,None)  # [S,B,E]
                loss = criterion(predicted.permute(1,0,2)[:,-1],Y_real.to(device))
                loss.backward()  
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)

                optimizer.step()
                one_epoch_loss.append(loss.item())

            print("Epoch {} Loss is {}".format(epoch+1,np.mean(one_epoch_loss)))

    #         print("Epoch {} - Time (in minutes) is {}".format(epoch+1,timedelta(seconds=(time()-time0))))
            all_epochs_loss.append(np.mean(one_epoch_loss))

        print("Total Time (in minutes) is {}".format( timedelta(seconds=(time()-time_all))))
        return all_epochs_loss 