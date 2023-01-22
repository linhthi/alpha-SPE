import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, act_layer=nn.ReLU,
                 dropout=0.5, evaluate=False):
        super(MLP, self).__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_dim, hidden_dim))
        self.acts = nn.ModuleList()
        self.acts.append(act_layer())
        for _ in range(n_layers - 2):
            self.lins.append(nn.Linear(hidden_dim, hidden_dim))
            self.acts.append(act_layer())

        self.lins.append(nn.Linear(hidden_dim, out_dim))

        self.dropout = dropout
        self.evaluate = evaluate

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, **kwargs):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.acts[i](x)
            # x = nn.ReLU()(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        if self.evaluate:
            x = x.log_softmax(dim=-1)

        return x