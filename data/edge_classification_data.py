import time
import os
import pickle
import numpy as np

import dgl
import torch
import torch.nn.functional as F

from scipy import sparse as sp
import numpy as np
import networkx as nx

import hashlib
import data.precompute_features as pf
from dgl import LaplacianPE, RandomWalkPE

from dgl.data import BitCoinOTCDataset, RedditDataset


class EdgeLinkPredictionDataset(torch.utils.data.Dataset):

    def __init__(self, name):
        """
            Loading datasets
        """
        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        self.graphs = None
        if name == 'BitcoinOTC':
            self.graphs = BitCoinOTCDataset()
        elif name == 'Reddit':
            self.graphs = RedditDataset()

    def collate(self, graph):
        graph.ndata['SE'] = pf.add_structural_feats(graph)
        FullEigVals, FullEigVecs = pf.laplace_decomp(graph, graph.num_nodes())
        graph.ndata['FullEigVals'] = torch.Tensor(FullEigVals)
        graph.ndata['FullEigVecs'] = torch.Tensor(FullEigVecs)
        return graph
    
    def split_data(g, test_size=0.1, valid_size=0.1):
    # Split edge set for training and testing
        u, v = g.edges()

        eids = np.arange(g.number_of_edges())
        eids = np.random.permutation(eids)
        test_size = int(len(eids) * 0.1)
        valid_size = int(len(eids) * 0.1)
        train_size = g.number_of_edges() - test_size - valid_size
        test_u, test_v = u[eids[:test_size]], v[eids[:test_size]]
        valid_u, valid_v = u[eids[test_size:test_size+valid_size]], v[eids[test_size:test_size+valid_size]]
        train_u, train_v = u[eids[test_size+valid_size:]], v[eids[test_size+valid_size:]]

        train_g = dgl.graph((train_u, train_v), num_nodes=g.number_of_nodes())
        train_g.edata['h'] = g.edata['h'][g.edge_ids(train_u, train_v)]

        test_g = dgl.graph((test_u, test_v), num_nodes=g.number_of_nodes())
        test_g.edata['h'] = g.edata['h'][g.edge_ids(test_u, test_v)]

        valid_g = dgl.graph((valid_u, valid_v), num_nodes=g.number_of_nodes())
        valid_g.edata['h'] = g.edata['h'][g.edge_ids(valid_u, valid_v)]
        return train_g, valid_g, test_g
