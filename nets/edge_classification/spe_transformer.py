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
from scipy.sparse.linalg import norm
from scipy import sparse as sp

class STransformer(nn.Module):

    def __init__(self, net_params):
        super().__init__()


        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        self.n_classes = net_params['n_classes']
        
        self.learn_alpha = net_params['learn_alpha']
        
        if not net_params['learn_alpha']:
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
        self.spe_hidden_dim = spe_hidden_dim
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
        self.embedding_se = MLP(in_dim=k, hidden_dim=hidden_dim, out_dim=GT_hidden_dim, n_layers=3, dropout=dropout)
        self.embedding_pe = MLP(in_dim=self.m*2, hidden_dim=hidden_dim, out_dim=GT_hidden_dim, n_layers=3, dropout=dropout)

        if self.learn_alpha:
            #self.w_alpha = nn.Conv1d(2, 1, kernel_size=1, bias=False)
            self.w_alpha = nn.Parameter(torch.rand(1))

        encoder_layer = nn.TransformerEncoderLayer(d_model=LSE_dim, nhead=LSE_n_heads)
        self.SE_Transformer = nn.TransformerEncoder(encoder_layer, num_layers=LSE_layers)
        self.layers = nn.ModuleList([ GraphTransformerLayer(GT_hidden_dim, GT_hidden_dim, GT_n_heads, dropout, self.layer_norm, self.batch_norm, self.residual) for _ in range(GT_layers-1) ])
        
        self.layers.append(GraphTransformerLayer(GT_hidden_dim, GT_out_dim, GT_n_heads, dropout, self.layer_norm, self.batch_norm, self.residual))

        self.MLP_layer = MLPReadout(GT_out_dim*2, self.n_classes)
        self.g = None


    def forward(self, g, h=None):
        
        EigVals, EigVecs = g.EigVals, g.EigVecs
        # mEigVecs = EigVecs[:, :self.m].to(self.device) # previous version
        m = self.m
        # print(EigVals)
        k = len(EigVals)
        mEigVals = torch.Tensor(EigVals[:m])
        mEigVals = mEigVals.repeat(k, 1) 
        mEigVecs = EigVecs[:, :m]
        PE_raw = torch.cat([mEigVals, mEigVecs], dim=1).to(self.device)
        # print(mEigVecs.shape, mEigVals.)

        h_se = g.ndata['SE']
        h_se = self.embedding_se(h_se)
        h_pe = PE_raw
        h_pe = self.embedding_pe(h_pe)
        if not self.learn_alpha:
            h_spe = (1-self.alpha)*h_pe + self.alpha*h_se
        else:
            w = self.w_alpha
            h_spe = w*h_se + (1 - w)*h_pe

        g.ndata['SPE'] = h_spe
        if h is None:
            h = h_spe
        else:
            h = self.embedding_h(h)
            h = self.in_feat_dropout(h)
            h = torch.cat([h, h_spe], dim=1)
        
        
        # GraphTransformer Layers
        for conv in self.layers:
            h = conv(g, h)
            
        # output
        h_out = self.MLP_layer(h)
        self.g = g

        return h_out
    
    
    def edge_predictor(self, h_i, h_j):
        x = torch.cat([h_i, h_j], dim=1)
        x = self.MLP_layer(x)
        
        return torch.sigmoid(x)
    
    def loss(self, pos_out, neg_out):
        pos_loss = -torch.log(pos_out + 1e-15).mean()  # positive samples
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()  # negative samples
        loss = pos_loss + neg_loss
        
        return loss



        
