
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from layers.mlp_readout_layer import MLPReadout

class MLP(nn.Module):
    def __init__(self, net_params):
        super(MLP, self).__init__()
        n_layers = net_params['n_layers']
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']

        self.num_layers = n_layers
        self.mlp = MLPReadout(in_dim, out_dim, n_layers)
        

    def forward(self, g, inputs):
        h = self.mlp(inputs)
        return h

    def loss(self, pred, label):
        # calculating label weights for weighted loss computation
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss