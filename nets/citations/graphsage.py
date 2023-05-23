"""
Inductive Representation Learning on Large Graphs
Paper: http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
Code: https://github.com/williamleif/graphsage-simple
Simple reference implementation of GraphSAGE.
"""
from torch import dropout
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch.conv import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, net_params):
        super(GraphSAGE, self).__init__()

        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_layers = net_params['n_layers']
        dropout = net_params['dropout']
        aggregator_type = net_params['readout']

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu

        # input layer
        self.layers.append(SAGEConv(in_dim, hidden_dim, aggregator_type))
        # hidden layers
        for i in range(n_layers - 2):
            self.layers.append(SAGEConv(in_dim, hidden_dim, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(hidden_dim, out_dim, aggregator_type)) # activation None

    def forward(self, graph, inputs):
        h = self.dropout(inputs)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != 0:
                h = self.activation(h)
                h = self.dropout(h)
        return h