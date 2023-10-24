import torch.nn as nn
import torch.nn.functional as F
import torch
import dgl.function as fn
from layers.gcn_layer import GCNLayer
import dgl.nn as dglnn
from layers.mlp import MLP


class GCNNet(nn.Module):
    """ """

    def __init__(self, net_params):
        super(GCNNet, self).__init__()
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        n_layers = net_params['n_layers']
        dropout = net_params['dropout']
        in_feat_dropout = net_params['in_feat_dropout']

        self.learn_alpha = net_params['learn_alpha']
        if not net_params['learn_alpha']:
            self.alpha = net_params['alpha']

        k = net_params['k']
        self.alpha = net_params['alpha']
        self.m = net_params['m']

        spe_hidden_dim = net_params['spe_hidden_dim']
        self.spe_hidden_dim = spe_hidden_dim

        self.device = net_params['device']
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)

        self.embedding_h = nn.Linear(in_dim, hidden_dim - spe_hidden_dim)
        self.embedding_pe = MLP(in_dim=self.m * 2, hidden_dim=hidden_dim, out_dim=spe_hidden_dim, n_layers=4, dropout=dropout)
        self.embedding_se = MLP(in_dim=k, hidden_dim=hidden_dim, out_dim=spe_hidden_dim, n_layers=4, dropout=dropout)

        if self.learn_alpha:
            self.w_alpha = nn.Linear(2 * spe_hidden_dim, 1)

        self.layers = nn.ModuleList()
        # self.layers.append(dglnn.GraphConv(in_dim, hidden_dim, activation=F.relu))
        for i in range(n_layers - 1):
            self.layers.append(dglnn.GraphConv(hidden_dim, hidden_dim, activation=F.relu))
        self.layers.append(dglnn.GraphConv(hidden_dim, out_dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, h):
        EigVals, EigVecs = g.ndata['EigVals'], g.ndata['EigVecs']

        m = self.m
        k = len(EigVals)

        mEigVals = torch.Tensor(EigVals[:m])
        mEigVals = mEigVals.repeat(k, 1) 
        mEigVecs = torch.Tensor(EigVecs[:, :m])

        # Positional Embedding
        PE_raw = torch.cat([mEigVals, mEigVecs], dim=1).to(self.device)
        h_pe = self.embedding_pe(PE_raw)

        # Structural Embedding
        h_se = g.ndata['SE']
        h_se = self.embedding_se(h_se)

        # Structural Positional Embedding
        if not self.learn_alpha:
            h_spe = (1-self.alpha)*h_pe + self.alpha*h_se
        elif self.learn_alpha == 'concat':
            h_se1 = torch.unsqueeze(h_se, 1)
            h_pe1 = torch.unsqueeze(h_pe, 1)

            h_tmp = torch.cat([h_se1, h_pe1], dim=1)
            h_tmp = self.w_alpha(h_tmp)

            h_spe = torch.squeeze(h_tmp, 1)
        else:
            w = self.w_alpha
            h_spe = w*h_se + (1 - w)*h_pe

        g.ndata['SPE'] = h_spe

        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        h = torch.cat([h, h_spe], dim=1)

        # GCN Layers
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


