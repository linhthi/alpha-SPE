# Original code: https://github.com/dmlc/dgl/blob/master/examples/pytorch/gat/gat.py
"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn import GATConv


class GAT(nn.Module):
    def __init__(self, net_params):
        super(GAT, self).__init__()
        n_layers = net_params['n_layers']
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        num_heads = net_params['heads']
        n_out_heads = net_params['n_out_heads']
        feat_drop = net_params['feat_drop']
        attn_drop = net_params['attn_drop']
        negative_slope = net_params['negative_slope']
        residual = net_params['residual']
        heads = ([num_heads] * (n_layers-1)) + [n_out_heads]

        self.num_layers = n_layers
        self.gat_layers = nn.ModuleList()
        self.activation = F.elu
        if n_layers > 1:
        # input projection (no residual)
            self.gat_layers.append(GATConv(
                in_dim, hidden_dim, heads[0],
                feat_drop, attn_drop, negative_slope, False, self.activation))
            # hidden layers
            for l in range(1, n_layers-1):
                # due to multi-head, the in_dim = num_hidden * num_heads
                self.gat_layers.append(GATConv(
                    hidden_dim * heads[l-1], hidden_dim, heads[l],
                    feat_drop, attn_drop, negative_slope, residual, self.activation))
            # output projection
            self.gat_layers.append(GATConv(
                hidden_dim * heads[-2], out_dim, heads[-1],
                feat_drop, attn_drop, negative_slope, residual, None))
        else:
            self.gat_layers.append(GATConv(
                in_dim, out_dim, heads[0],
                feat_drop, attn_drop, negative_slope, residual, None))

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h)
            h = h.flatten(1) if l != self.num_layers - 1 else h.mean(1)
        return h

    def loss(self, pred, label):
        # calculating label weights for weighted loss computation
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss