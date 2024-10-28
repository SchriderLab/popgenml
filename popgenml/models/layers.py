# -*- coding: utf-8 -*-

from torch import nn

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