import torch.nn as nn
import torch.nn.functional as F
import torch
import dgl.function as fn
from layers.gcn_layer import GCNLayer
import dgl.nn as dglnn


class GCNNet(nn.Module):
    """ """

    def __init__(self, net_params):
        super(GCNNet, self).__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_layers = net_params['n_layers']
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GraphConv(in_dim, hidden_dim, activation=F.relu))
        for i in range(n_layers - 2):
            self.layers.append(dglnn.GraphConv(hidden_dim, hidden_dim, activation=F.relu))
        self.layers.append(dglnn.GraphConv(hidden_dim, out_dim))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h

    def loss(self, pred, label):
        # calculating label weights for weighted loss computation
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss


