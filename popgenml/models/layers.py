# -*- coding: utf-8 -*-

from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class MLP(nn.Module):
    """
    A customizable Multi-Layer Perceptron (MLP) neural network.

    This module creates a sequential stack of linear layers, followed by
    normalization, activation, and dropout layers. It automatically flattens 
    the input tensor to 2D before passing it through the network.

    Args:
        input_dim (int): The size of the input feature dimension.
        output_dim (int): The size of the final output dimension.
        dim (int, optional): The number of features in the hidden layers. Defaults to 256.
        n_blk (int, optional): The total number of linear layers (blocks) in the network. 
            Must be at least 2 (input block + output block). Defaults to 3.
        norm (callable, optional): The normalization layer class to use. Defaults to nn.BatchNorm1d.
        activation (callable, optional): The activation function class to use. Defaults to nn.ReLU.
        dropout (float, optional): The dropout probability. Defaults to 0.0.
    """
    def __init__(self, input_dim, output_dim, dim=256, n_blk=3, norm=nn.BatchNorm1d,
                 activation=nn.ReLU, dropout=0.0):
        super(MLP, self).__init__()
        
        # 1. Input layer block
        layers = [nn.Linear(input_dim, dim), norm(dim), activation(inplace=True), nn.Dropout(dropout)]
        
        # 2. Hidden layer blocks (total blocks - 2 accounts for input and output layers)
        for _ in range(n_blk - 2):
            layers += [nn.Linear(dim, dim), norm(dim), activation(inplace=True), nn.Dropout(dropout)]
            
        # 3. Output layer (no activation or normalization applied here)
        layers += [nn.Linear(dim, output_dim)]
        
        # Unpack the list of layers into a Sequential container
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the forward pass of the MLP.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ...).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Flatten all dimensions except batch size (dim 0) before passing to linear layers
        return self.model(x.view(x.size(0), -1))
    

class RNNEstimator(nn.Module):
    """
    A sequence-to-vector estimator that combines a Transformer Encoder and a GRU.

    This model processes sequential data by first passing it through a Transformer Encoder 
    to capture global context/attention. The original input and the transformer output 
    are concatenated and passed into a GRU. The final hidden state of the GRU is then 
    passed through an MLP head to produce the final estimation.

    Note: 
        `global_shape`, `g_norm`, `h_norm`, and `affine` are initialized but currently 
        unused in the forward pass. `self.drop` is also initialized but overridden 
        if a dropout function is passed to `forward()`.

    Args:
        in_dim (int): The number of expected features in the input `x`.
        out_dim (int): The size of the final output dimension.
        hidden_size (int, optional): The number of features in the GRU hidden state and 
            Transformer feedforward network. Defaults to 128.
        global_shape (int, optional): Size of an auxiliary global feature (currently unused). Defaults to 84.
        n_heads (int, optional): The number of heads in the multiheadattention models. Defaults to 5.
        num_layers (int, optional): The number of sub-encoder-layers and GRU layers. Defaults to 6.
        dropout (float, optional): The dropout value for the Transformer Encoder. Defaults to 0.0.
    """
    def __init__(self, in_dim, out_dim, hidden_size=128, global_shape=84, n_heads=5, num_layers=6, dropout=0.0):
        super().__init__()
        
        # Initialization of normalization and affine layers (Currently unused in forward pass)
        self.in_norm = nn.LayerNorm(in_dim)
        self.g_norm = nn.LayerNorm(global_shape)
        self.h_norm = nn.LayerNorm(hidden_size + global_shape)
        self.affine = nn.Linear(in_dim, in_dim)
        
        # The input to the GRU will be the concatenated original input + transformer output (in_dim * 2)
        self.gru = nn.GRU(in_dim * 2, hidden_size, num_layers=num_layers, batch_first=True)
        
        # Transformer Encoder to capture self-attention across the sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim, 
            nhead=n_heads, 
            dim_feedforward=hidden_size, 
            batch_first=True, 
            dropout=dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # MLP output head to map the GRU hidden state to the final output dimension
        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), 
            nn.BatchNorm1d(hidden_size), 
            nn.LeakyReLU(), 
            nn.Linear(hidden_size, hidden_size), 
            nn.BatchNorm1d(hidden_size), 
            nn.LeakyReLU(),
            nn.Linear(hidden_size, out_dim)
        )
                
        # Class-level dropout (unused in forward pass unless explicitly called)
        self.drop = nn.Dropout(0.5)
        
    def forward(self, x, ls, drop=None):
        """
        Defines the forward pass of the RNNEstimator.

        Args:
            x (torch.Tensor): Padded input sequence tensor of shape (batch_size, seq_len, in_dim).
            ls (torch.Tensor or list): Unpadded sequence lengths used for packing the sequence.
            drop (callable, optional): An optional dropout function/module to apply to the GRU 
                hidden state before the final MLP head. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_dim).
        """
        # 1. Pass input through the Transformer to get self-attended features
        x0 = self.encoder(x)
  
        # 2. Concatenate the original input features with the Transformer output features
        # Shape becomes (batch_size, seq_len, in_dim * 2)
        x = torch.cat([x, x0], dim=-1)
  
        # 3. Pack the padded sequence for efficient RNN processing (ignores padded zeros)
        x = pack_padded_sequence(x, ls, batch_first=True)
  
        # 4. Pass through the GRU
        # x contains the packed output, h contains the final hidden states for all layers
        x, h = self.gru(x)
        
        # Extract the hidden state from the FIRST layer of the GRU.
        # Note: h has shape (num_layers, batch_size, hidden_size). 
        # Typically, h[-1] (the last layer's hidden state) is used. 
        # If intending to use the final layer, consider changing h[0] to h[-1].
        h = h[0]
        
        # 5. Apply optional runtime dropout to the hidden state
        if drop:
            h = drop(h)
        
        # 6. Project to the final output dimension
        return self.out(h)
    
  

    