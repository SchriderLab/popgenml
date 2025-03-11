# -*- coding: utf-8 -*-

from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence

# a basic MLP module
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim = 256, n_blk = 3, norm = nn.BatchNorm1d,
                 activation = nn.ReLU, dropout = 0.0):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, dim), norm(dim), activation(inplace=True), nn.Dropout(dropout)]
        for _ in range(n_blk - 2):
            layers += [nn.Linear(dim, dim), norm(dim), activation(inplace=True), nn.Dropout(dropout)]
        layers += [nn.Linear(dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))
    
class RNNEstimator(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size = 128, global_shape = 84, n_heads = 5, num_layers = 1, dropout = 0.0):
        super().__init__()
        
        self.in_norm = nn.LayerNorm(in_dim)
        self.g_norm = nn.LayerNorm(global_shape)
        
        self.h_norm = nn.LayerNorm(hidden_size + global_shape)
        
        self.affine = nn.Linear(in_dim, in_dim)
        
        self.gru = nn.GRU(in_dim * 2, hidden_size, num_layers = num_layers, batch_first = True)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(in_dim, n_heads, dim_feedforward = hidden_size, batch_first = True, dropout = dropout), num_layers = 6)
        
        self.out = nn.Sequential(*[nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.LeakyReLU(), 
                                  nn.Linear(hidden_size, hidden_size), nn.BatchNorm1d(hidden_size), nn.LeakyReLU(),
                                  nn.Linear(hidden_size, out_dim)])
                
        self.drop = nn.Dropout(0.5)
        
    def forward(self, x, ls, drop = None):
        x0 = self.encoder(x)
  
        x = torch.cat([x, x0], -1)
  
        x = pack_padded_sequence(x, ls, batch_first = True)
  
        x, h = self.gru(x)
        h = h[0]
        
        if drop:
            h = drop(h)
        
        return self.out(h)
    
  

    