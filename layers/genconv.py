# Reference: https://github.com/dmlc/dgl/blob/master/examples/pytorch/deepergcn/layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import BondEncoder

import dgl.function as fn
from dgl.nn.functional import edge_softmax


class GENConv(nn.Module):
    r"""

    Description
    -----------
    Generalized Message Aggregator was introduced in "DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>"

    Parameters
    ----------
    in_dim: int
        Input size.
    out_dim: int
        Output size.
    aggregator: str
        Type of aggregation. Default is 'softmax'.
    beta: float
        A continuous variable called an inverse temperature. Default is 1.0.
    learn_beta: bool
        Whether beta is a learnable variable or not. Default is False.
    p: float
        Initial power for power mean aggregation. Default is 1.0.
    learn_p: bool
        Whether p is a learnable variable or not. Default is False.
    msg_norm: bool
        Whether message normalization is used. Default is False.
    learn_msg_scale: bool
        Whether s is a learnable scaling factor or not in message normalization. Default is False.
    mlp_layers: int
        The number of MLP layers. Default is 1.
    eps: float
        A small positive constant in message construction function. Default is 1e-7.
    """

    def __init__(self, net_params):
        in_dim = net_params['in_dim']
        out_dim = net_params['out_dim']
        aggregator="softmax",
        beta = net_params['beta'] # default 1.0,
        learn_beta = net_params['learn_beta'] #False,
        p = net_params['p'] # 1.0,
        learn_p = net_params['learn_p'] # False,
        msg_norm = net_params['msg_norm'] # False,
        learn_msg_scale = net_params['learn_msg_scale'] # False,
        mlp_layers = net_params['mlp_layers'] # 1,
        eps = net_params['eps'] # 1e-7,
        super(GENConv, self).__init__()

        self.aggr = aggregator
        self.eps = eps

        channels = [in_dim]
        for _ in range(mlp_layers - 1):
            channels.append(in_dim * 2)
        channels.append(out_dim)

        self.mlp = MLP(channels)
        self.msg_norm = MessageNorm(learn_msg_scale) if msg_norm else None

        self.beta = (
            nn.Parameter(torch.Tensor([beta]), requires_grad=True)
            if learn_beta and self.aggr == "softmax"
            else beta
        )
        self.p = (
            nn.Parameter(torch.Tensor([p]), requires_grad=True)
            if learn_p
            else p
        )

        self.edge_encoder = BondEncoder(in_dim)

    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            # Node and edge feature size need to match.
            g.ndata["h"] = node_feats
            g.edata["h"] = self.edge_encoder(edge_feats)
            g.apply_edges(fn.u_add_e("h", "h", "m"))

            if self.aggr == "softmax":
                g.edata["m"] = F.relu(g.edata["m"]) + self.eps
                g.edata["a"] = edge_softmax(g, g.edata["m"] * self.beta)
                g.update_all(
                    lambda edge: {"x": edge.data["m"] * edge.data["a"]},
                    fn.sum("x", "m"),
                )

            elif self.aggr == "power":
                minv, maxv = 1e-7, 1e1
                torch.clamp_(g.edata["m"], minv, maxv)
                g.update_all(
                    lambda edge: {"x": torch.pow(edge.data["m"], self.p)},
                    fn.mean("x", "m"),
                )
                torch.clamp_(g.ndata["m"], minv, maxv)
                g.ndata["m"] = torch.pow(g.ndata["m"], self.p)

            else:
                raise NotImplementedError(
                    f"Aggregator {self.aggr} is not supported."
                )

            if self.msg_norm is not None:
                g.ndata["m"] = self.msg_norm(node_feats, g.ndata["m"])

            feats = node_feats + g.ndata["m"]

            return self.mlp(feats)

class MessageNorm(nn.Module):
    r"""
    Description
    -----------
    Message normalization was introduced in "DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>"
    Parameters
    ----------
    learn_scale: bool
        Whether s is a learnable scaling factor or not. Default is False.
    """

    def __init__(self, learn_scale=False):
        super(MessageNorm, self).__init__()
        self.scale = nn.Parameter(
            torch.FloatTensor([1.0]), requires_grad=learn_scale
        )

    def forward(self, feats, msg, p=2):
        msg = F.normalize(msg, p=2, dim=-1)
        feats_norm = feats.norm(p=p, dim=-1, keepdim=True)
        return msg * feats_norm * self.scale

class MLP(nn.Sequential):
    r"""
    Description
    -----------
    From equation (5) in "DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>"
    """

    def __init__(self, channels, act="relu", dropout=0.0, bias=True):
        layers = []

        for i in range(1, len(channels)):
            layers.append(nn.Linear(channels[i - 1], channels[i], bias))
            if i < len(channels) - 1:
                layers.append(nn.BatchNorm1d(channels[i], affine=True))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*layers)