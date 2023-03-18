import dgl
import numpy as np
from sklearn.model_selection import train_test_split
from dgl.data import BitcoinOTCDataset, CoraGraphDataset
import scipy.sparse as sp
import torch

# Load the dataset and extract the first graph
dataset = BitcoinOTCDataset()
g = dataset[0]

# Split edge set for training and testing
u, v = g.edges()
print(u, v)

eids = np.arange(g.number_of_edges())
eids = np.random.permutation(eids)
test_size = int(len(eids) * 0.1)
valid_size = int(len(eids) * 0.1)
train_size = g.number_of_edges() - test_size - valid_size
test_u, test_v = u[eids[:test_size]], v[eids[:test_size]]
valid_u, valid_v = u[eids[test_size:test_size+valid_size]], v[eids[test_size:test_size+valid_size]]
train_u, train_v = u[eids[test_size+valid_size:]], v[eids[test_size+valid_size:]]
print(train_u.shape, train_v)

train_g = dgl.graph((train_u, train_v), num_nodes=g.number_of_nodes())
train_g.edata['h'] = g.edata['h'][g.edge_ids(train_u, train_v)]

test_g = dgl.graph((test_u, test_v), num_nodes=g.number_of_nodes())
test_g.edata['h'] = g.edata['h'][g.edge_ids(test_u, test_v)]

valid_g = dgl.graph((valid_u, valid_v), num_nodes=g.number_of_nodes())
valid_g.edata['h'] = g.edata['h'][g.edge_ids(valid_u, valid_v)]

print(valid_g.edges(), valid_g.edata['h'])
