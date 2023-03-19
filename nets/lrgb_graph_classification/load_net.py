"""
    Utility file to select GraphNN model as
    selected by the user
"""
 
from nets.lrgb_graph_classification.spe_transformer import SPE_TransformerNet


def SPE(net_params):
    return SPE_TransformerNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'SPE': SPE
    }
        
    return models[MODEL_NAME](net_params)
