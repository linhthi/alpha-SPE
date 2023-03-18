"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.edge_classification.gat_net import GATNet
from nets.edge_classification.gcn_net import GCNNet
from nets.edge_classification.spe_transformer import SPE_TransformerNet


def GCN(net_params):
    return GCNNet(net_params)

def GAT(net_params):
    return GATNet(net_params)
def SPE(net_params):
    return SPE_TransformerNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GCN': GCN,
        'GAT': GAT,
        'SPE': SPE
    }
        
    return models[MODEL_NAME](net_params)