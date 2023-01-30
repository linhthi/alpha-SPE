# Reference: https://github.com/DevinKreuzer/SAN/blob/main/nets/SBMs_node_classification/SAN.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import numpy as np

"""
    Transformer using SPE
    
"""
from layers.gt import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout
from layers.mlp import MLP

class SPE_TransformerNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()


        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        self.n_classes = net_params['n_classes']
        
        self.alpha = net_params['alpha']
        
        GT_layers = net_params['GT_layers']
        GT_hidden_dim = net_params['GT_hidden_dim']
        GT_out_dim = net_params['GT_out_dim']
        GT_n_heads = net_params['GT_n_heads']

        LSE_dim = net_params['LSE_dim']
        LSE_n_heads = net_params['LSE_n_heads']
        LSE_layers = net_params['LSE_layers']
        
        self.residual = net_params['residual']
        self.readout = net_params['readout']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        spe_hidden_dim = net_params['spe_hidden_dim']
        k = net_params['k']
        hidden_dim = net_params['hidden_dim']
        self.alpha = net_params['alpha']
        self.m = net_params['m']

        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']

        self.device = net_params['device']
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        
        self.embedding_h = nn.Linear(in_dim_node, GT_hidden_dim-spe_hidden_dim)
        self.embedding_se = MLP(in_dim=k, hidden_dim=hidden_dim, out_dim=spe_hidden_dim, n_layers=4, dropout=dropout)
        self.embedding_pe = MLP(in_dim=self.m, hidden_dim=hidden_dim, out_dim=spe_hidden_dim, n_layers=4, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=LSE_dim, nhead=LSE_n_heads)
        self.SE_Transformer = nn.TransformerEncoder(encoder_layer, num_layers=LSE_layers)
        self.layers = nn.ModuleList([ GraphTransformerLayer(GT_hidden_dim, GT_hidden_dim, GT_n_heads, dropout, self.layer_norm, self.batch_norm, self.residual) for _ in range(GT_layers-1) ])
        
        self.layers.append(GraphTransformerLayer(GT_hidden_dim, GT_out_dim, GT_n_heads, dropout, self.layer_norm, self.batch_norm, self.residual))

        self.MLP_layer = MLPReadout(GT_out_dim, self.n_classes)


    def forward(self, g, h, e):
        
        EigVals, EigVecs = g.ndata['EigVals'], g.ndata['EigVecs']
        # mEigVals = EigVals[:self.m]
        mEigVecs = EigVecs[:, :self.m].to(self.device)
        # print(mEigVecs.shape, mEigVals.)

        h_se = g.ndata['SE']
        h_se = self.embedding_se(h_se)
        h_pe = mEigVecs
        h_pe = self.embedding_pe(h_pe)
        h_spe = (1-self.alpha)*h_pe + self.alpha*h_se

        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        h = torch.cat([h, h_spe], dim=1)
        
        
        # GraphTransformer Layers
        for conv in self.layers:
            h = conv(g, h)
        
        g.ndata['h'] = h

        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        # output
        h_out = self.MLP_layer(hg)

        return h_out
    
    
    def loss(self, pred, label):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(pred, label)
        return loss



        