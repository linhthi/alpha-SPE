"""
    Utility file to select GraphNN model as
    selected by the user
"""
 
from nets.lrgb_graph_classification.spe_transformer import SPE_TransformerNet
from nets.lrgb_graph_classification.SAN_NodeLPE import SAN_NodeLPE
from nets.lrgb_graph_classification.gcn_net import GCNNet

def SPE(net_params):
    return SPE_TransformerNet(net_params)

def SAN_NodePE(net_params):
    return SAN_NodeLPE(net_params)

def GCN(net_params):
    return GCNNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'SPE': SPE,
        'SAN_NodeLPE': SAN_NodePE,
        'GCN': GCN,
    }
        
    return models[MODEL_NAME](net_params)
