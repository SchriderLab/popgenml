# -*- coding: utf-8 -*-

from torch import nn
import torch

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
    
class ResNet1d(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        
        self.conv0 = nn.Sequential(nn.Conv1d(in_dim, in_dim, 1, stride = 1), nn.BatchNorm1d(in_dim))
        
        self.layers = nn.ModuleList()
        self.layers_ = nn.ModuleList()
        channels = [in_dim, 64, 128, 256]
        
        for k in range(len(channels) - 1):
            self.layers.append(nn.Sequential(nn.Conv1d(channels[k], channels[k], 3, padding = 1), nn.BatchNorm1d(channels[k])))
            self.layers_.append(nn.Sequential(nn.Conv1d(channels[k], channels[k + 1], 3, padding = 1), nn.BatchNorm1d(channels[k + 1]), nn.ReLU()))

    def forward(self, x):
        x = self.conv0(x)
        
        for k in range(len(self.layers)):
            x0 = self.layers[k](x)
            x = self.layers_[k](x0 + x)
            
        return x
        
class TransformerConv(nn.Module):
    def __init__(self, in_dim, out_dim = 3, hidden_size = 512, dropout = 0.):
        super().__init__()
        
        self.conv = ResNet1d(in_dim)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(256, 8, dim_feedforward = hidden_size, batch_first = True, dropout = dropout), num_layers = 3)
        self.gru = nn.GRU(512, 512, batch_first = True)
        
        self.mlp = MLP(512, out_dim, 512)
        
    def forward(self, x):
        x = self.conv(x).transpose(1, 2)
        
        x0 = self.encoder(x)
        x, h = self.gru(torch.cat([x, x0], -1))
        h = h[0]
        
        return self.mlp(h)
    
if __name__ == '__main__':
    import torch
    
    model = TransformerConv(61)
    
    x = torch.randn((8, 61, 128))
    print(model(x).shape)    

    