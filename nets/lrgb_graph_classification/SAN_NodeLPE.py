# SAN: LPE Transformer over nodes
# URL: https://github.com/DevinKreuzer/SAN/blob/main/nets/SBMs_node_classification/SAN_NodeLPE.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import numpy as np

"""
    Graph Transformer
    
"""
from layers.gt import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout

class SAN_NodeLPE(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params['in_dim'] # node_dim (feat is an integer)
        # self.n_classes = net_params['n_classes']
        
        full_graph = net_params['full_graph']
        gamma = net_params['gamma']
        self.m = net_params['m']
        
        LPE_layers = net_params['LPE_layers']
        LPE_dim = net_params['LPE_dim']
        LPE_n_heads = net_params['LPE_n_heads']
        
        GT_layers = net_params['GT_layers']
        GT_hidden_dim = net_params['GT_hidden_dim']
        GT_out_dim = net_params['GT_out_dim']
        GT_n_heads = net_params['GT_n_heads']
        
        self.residual = net_params['residual']
        self.readout = net_params['readout']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']

        self.readout = net_params['readout']
        self.layer_norm = net_params['layer_norm']
        self.batch_norm = net_params['batch_norm']

        self.device = net_params['device']
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        #Remove some embedding dimensions to make room for concatenating laplace encoding
        self.embedding_h = nn.Linear(in_dim_node, GT_hidden_dim-LPE_dim)
        # self.embedding_e = nn.Linear(2, GT_hidden_dim)
        self.linear_A = nn.Linear(2, LPE_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=LPE_dim, nhead=LPE_n_heads)
        self.PE_Transformer = nn.TransformerEncoder(encoder_layer, num_layers=LPE_layers)
        
        # self.layers = nn.ModuleList([ GraphTransformerLayer(gamma, GT_hidden_dim, GT_hidden_dim, GT_n_heads, full_graph, dropout, self.layer_norm, self.batch_norm, self.residual) for _ in range(GT_layers-1) ])
        
        # self.layers.append(GraphTransformerLayer(gamma, GT_hidden_dim, GT_out_dim, GT_n_heads, full_graph, dropout, self.layer_norm, self.batch_norm, self.residual))
        self.layers = nn.ModuleList([ GraphTransformerLayer(GT_hidden_dim, GT_hidden_dim, GT_n_heads, dropout, self.layer_norm, self.batch_norm, self.residual) for _ in range(GT_layers-1) ])
        
        self.layers.append(GraphTransformerLayer(GT_hidden_dim, GT_out_dim, GT_n_heads, dropout, self.layer_norm, self.batch_norm, self.residual))


        self.MLP_layer = MLPReadout(GT_out_dim, 11)


    def forward(self, g, h, e=None):
        EigVals, EigVecs = g.ndata['EigVals'], g.ndata['EigVecs']
        m = self.m
        # EigVals = EigVals.unsqueeze(0)
        s = EigVals.shape[0]
        EigVals, EigVecs = EigVals[: m], EigVecs[:, :m]
        EigVals = EigVals.repeat(s,1).unsqueeze(2)
        # print(EigVals.shape, EigVecs.shape)
        
        # input embedding
        # h = torch.LongTensor(h)
        h = self.embedding_h(h)
        # e = self.embedding_e(e) 
          
        PosEnc = torch.cat((EigVecs.unsqueeze(2), EigVals), dim=2).float() # (Num nodes) x (Num Eigenvectors) x 2
        empty_mask = torch.isnan(PosEnc) # (Num nodes) x (Num Eigenvectors) x 2
        
        PosEnc[empty_mask] = 0 # (Num nodes) x (Num Eigenvectors) x 2
        PosEnc = torch.transpose(PosEnc, 0 ,1).float() # (Num Eigenvectors) x (Num nodes) x 2
        PosEnc = self.linear_A(PosEnc) # (Num Eigenvectors) x (Num nodes) x PE_dim
        
        
        #1st Transformer: Learned PE
        PosEnc = self.PE_Transformer(src=PosEnc, src_key_padding_mask=empty_mask[:,:,0]) 
        
        #remove masked sequences
        PosEnc[torch.transpose(empty_mask, 0 ,1)[:,:,0]] = float('nan') 
        
        #Sum pooling
        PosEnc = torch.nansum(PosEnc, 0, keepdim=False)
        
        #Concatenate learned PE to input embedding
        h = torch.cat((h, PosEnc), 1)
        
        h = self.in_feat_dropout(h)
        
        # GraphTransformer Layers
        for conv in self.layers:
            h = conv(g, h)
        
        g.ndata['h'] = h
        
        # output
        if self.readout == 'sum':
            h_out = dgl.sum_nodes(g, 'h')
        elif self.readout == 'max':
            h_out = dgl.max_nodes(g, 'h')
        elif self.readout == 'mean':
            h_out = dgl.mean_nodes(g, 'h')

        h_out = self.MLP_layer(h_out)

        return h_out
    
    
    def loss(self, pred, label):
        criterion = nn.L1Loss()
        loss = criterion(pred, label)
        return loss



        
