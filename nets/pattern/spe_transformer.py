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
        
        
        # self.embedding_h = nn.Linear(in_dim_node, GT_hidden_dim-spe_hidden_dim)
        self.embedding_h = nn.Embedding(in_dim_node, GT_hidden_dim - spe_hidden_dim)
        self.embedding_se = MLP(in_dim=k, hidden_dim=hidden_dim, out_dim=spe_hidden_dim, n_layers=4, dropout=dropout)
        self.embedding_pe = MLP(in_dim=self.m*2, hidden_dim=hidden_dim, out_dim=spe_hidden_dim, n_layers=4, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=LSE_dim, nhead=LSE_n_heads)
        self.SE_Transformer = nn.TransformerEncoder(encoder_layer, num_layers=LSE_layers)
        self.layers = nn.ModuleList([ GraphTransformerLayer(GT_hidden_dim, GT_hidden_dim, GT_n_heads, dropout, self.layer_norm, self.batch_norm, self.residual) for _ in range(GT_layers-1) ])
        
        self.layers.append(GraphTransformerLayer(GT_hidden_dim, GT_out_dim, GT_n_heads, dropout, self.layer_norm, self.batch_norm, self.residual))

        self.MLP_layer = MLPReadout(GT_out_dim, self.n_classes)
        self.g = None


    def forward(self, g, h):
        
        h = h.type(torch.LongTensor).to(self.device)
        # EigVals, EigVecs = g.EigVals, g.EigVecs
        EigVals, EigVecs = g.ndata['EigVals'], g.ndata['EigVecs']
        # print(EigVals.shape, EigVecs.shape)
        # mEigVecs = EigVecs[:, :self.m].to(self.device) # previous version
        m = self.m
        # print(EigVals)
        k = len(EigVals)
        # print(EigVecs.shape)
        mEigVals = torch.Tensor(EigVals[:m])
        mEigVals = mEigVals.repeat(k, 1) 
        mEigVecs = EigVecs
        # print(mEigVecs.shape)
        PE_raw = torch.cat([mEigVals, mEigVecs], dim=1).to(self.device)
        # print(mEigVecs.shape, mEigVals.)

        h_se = g.ndata['SE']
        h_se = self.embedding_se(h_se)
        h_pe = PE_raw
        h_pe = self.embedding_pe(h_pe)
        h_spe = (1-self.alpha)*h_pe + self.alpha*h_se
        g.ndata['SPE'] = h_spe

        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        h = torch.cat([h, h_spe], dim=1)
        
        
        # GraphTransformer Layers
        for conv in self.layers:
            # print(h.shape)
            # print(g.num_nodes())
            h = conv(g, h)
            
        # output
        h_out = self.MLP_layer(h)
        self.g = g

        return h_out
    
    
    def loss(self, pred, label):

        # calculating label weights for weighted loss computation
        V = label.size(0)
        label_count = torch.bincount(label)
        label_count = label_count[label_count.nonzero()].squeeze()
        cluster_sizes = torch.zeros(self.n_classes).long().to(self.device)
        cluster_sizes[torch.unique(label)] = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes>0).float()
        
        # weighted cross-entropy for unbalanced classes
        criterion = nn.CrossEntropyLoss(weight=weight)
        loss_a = criterion(pred, label)

        g = self.g
        n = g.number_of_nodes()
        A = g.adjacency_matrix(scipy_fmt="csr")
        N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
        L = sp.eye(n) - N * A * N
        # Loss location encoding
        p = g.ndata['SPE']
        pT = torch.transpose(p, 1, 0)
        loss_b_1 = torch.trace(torch.mm(torch.mm(pT, torch.Tensor(L.todense()).to(self.device)), p))

        # Correct batch-graph wise loss_b_2 implementation; using a block diagonal matrix
        # bg = dgl.unbatch(g)
        # batch_size = len(bg)
        # P = sp.block_diag(p)
        # PTP_In = P.T * P - sp.eye(P)
        # loss_b_2 = torch.tensor(norm(PTP_In, 'fro')**2).float().to(self.device)

        # loss_b = ( loss_b_1 + 100 * loss_b_2 ) / ( self.spe_hidden_dim * n) 
        loss_b = loss_b_1

        # /del bg, P, PTP_In, loss_b_1, loss_b_2
        del loss_b_1

        return loss_a + 0.01*loss_b



        