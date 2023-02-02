"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.citations.gcn_net import GCNNet
from nets.citations.Transformer import Transformer
from nets.citations.spe_transformer import SPETransformer as STransformer
from nets.citations.SAN_NodeLPE import SAN_NodeLPE
from nets.citations.SAN_EdgeLPE import SAN_EdgeLPE
from nets.citations.graphsage import GraphSAGE
from nets.citations.gat import GAT
from nets.citations.mlp import MLP

def NodeLPE(net_params):
    return SAN_NodeLPE(net_params)

def EdgeLPE(net_params):
    return SAN_EdgeLPE(net_params)

def NoLPE(net_params):
    return Transformer(net_params)

def GCN(net_params):
    return GCNNet(net_params)

def GraphSAGE(net_params):
    return GraphSAGE(net_params)

def GATNet(net_params):
    return GAT(net_params)

def MLPNet(net_params):
    return MLP(net_params)

def SPETransformer(net_params):
    return STransformer(net_params)

def gnn_model(LPE, net_params):
    model = {
        'edge': EdgeLPE,
        'nodeLPE': NodeLPE,
        'none': NoLPE,
        'gcn': GCN,
        'graphsage': GraphSAGE, 
        'gat': GATNet,
        'mlp': MLPNet,
        'spe': SPETransformer,
    }
        
    return model[LPE](net_params)